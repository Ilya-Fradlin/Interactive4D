import torch
import argparse
import itertools
import torch.nn as nn
import MinkowskiEngine.MinkowskiOps as me
from MinkowskiEngine.MinkowskiPooling import MinkowskiAvgPooling
from models.modules.common import conv
from models.position_embedding import PositionEmbeddingCoordsSine
from models.modules.helpers_3detr import GenericMLP
from torch.cuda.amp import autocast
from models import Res16UNet34C
from models.modules.attention import CrossAttentionLayer, SelfAttentionLayer, FFNLayer
from models.position_embedding import PositionalEncoding1D, PositionEmbeddingCoordsSine, PositionalEncoding3D


class Interactive4D(nn.Module):
    def __init__(
        self,
        num_heads,
        num_decoders,
        hidden_dim,
        dim_feedforward,
        shared_decoder,
        num_bg_queries,
        dropout,
        pre_norm,
        aux,
        voxel_size,
        sample_sizes,
        sweep_size,
    ):
        super().__init__()

        backbone_cfg = argparse.ArgumentParser()
        backbone_cfg.dilations = [1, 1, 1, 1]
        backbone_cfg.conv1_kernel_size = 5
        backbone_cfg.bn_momentum = 0.02
        if sweep_size == 1:
            in_channels = 2
        else:
            in_channels = 3  # scan index is added as an additional feature
        self.sweep_size = sweep_size
        self.backbone = Res16UNet34C(in_channels=in_channels, out_channels=19, config=backbone_cfg)
        self.num_heads = num_heads
        self.num_decoders = num_decoders
        self.mask_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.shared_decoder = shared_decoder
        self.num_bg_queries = num_bg_queries
        self.dropout = dropout
        self.pre_norm = pre_norm
        self.aux = aux
        self.voxel_size = voxel_size
        self.sample_sizes = sample_sizes

        sizes = self.backbone.PLANES[-5:]

        self.lin_squeeze_head = conv(self.backbone.PLANES[7], self.mask_dim, kernel_size=1, stride=1, bias=True, D=3)

        self.bg_query_feat = nn.Embedding(num_bg_queries, self.mask_dim)
        self.bg_query_pos = nn.Embedding(num_bg_queries, self.mask_dim)

        self.mask_embed_head = nn.Sequential(nn.Linear(self.mask_dim, self.mask_dim), nn.ReLU(), nn.Linear(self.mask_dim, self.mask_dim))

        self.pos_enc = PositionEmbeddingCoordsSine(pos_type="fourier", d_pos=hidden_dim, gauss_scale=1.0, normalize=True)

        self.pooling = MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=3)

        self.masked_transformer_decoder = nn.ModuleList()

        # Click-to-scene attention
        self.c2s_attention = nn.ModuleList()

        # Click-to-click attention
        self.c2c_attention = nn.ModuleList()

        # FFN
        self.ffn_attention = nn.ModuleList()

        # Scene-to-click attention
        self.s2c_attention = nn.ModuleList()

        num_uniq_decoders = self.num_decoders if not self.shared_decoder else 1

        for _ in range(num_uniq_decoders):
            tmp_c2s_attention = nn.ModuleList()
            tmp_s2c_attention = nn.ModuleList()
            tmp_c2c_attention = nn.ModuleList()
            tmp_ffn_attention = nn.ModuleList()

            tmp_c2s_attention.append(
                CrossAttentionLayer(
                    d_model=self.mask_dim,
                    nhead=self.num_heads,
                    dropout=self.dropout,
                    normalize_before=self.pre_norm,
                )
            )

            tmp_s2c_attention.append(
                CrossAttentionLayer(
                    d_model=self.mask_dim,
                    nhead=self.num_heads,
                    dropout=self.dropout,
                    normalize_before=self.pre_norm,
                )
            )

            tmp_c2c_attention.append(
                SelfAttentionLayer(
                    d_model=self.mask_dim,
                    nhead=self.num_heads,
                    dropout=self.dropout,
                    normalize_before=self.pre_norm,
                )
            )

            tmp_ffn_attention.append(
                FFNLayer(
                    d_model=self.mask_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=self.dropout,
                    normalize_before=self.pre_norm,
                )
            )

            self.c2s_attention.append(tmp_c2s_attention)
            self.s2c_attention.append(tmp_s2c_attention)
            self.c2c_attention.append(tmp_c2c_attention)
            self.ffn_attention.append(tmp_ffn_attention)

        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.time_encode = PositionalEncoding1D(hidden_dim, 2000)
        self.scan_num_encode = PositionalEncoding1D(hidden_dim, 5000)

        # Learned object-id embeddings
        self.object_embedding = nn.Embedding(400, hidden_dim)

    def forward_backbone(self, x, raw_coordinates=None, is_eval=False):
        device = x.device
        all_features = self.backbone(x)
        pcd_features = all_features[-1]
        pcd_features = self.lin_squeeze_head(pcd_features)

        with torch.no_grad():
            coordinates = me.SparseTensor(
                features=raw_coordinates,
                coordinate_manager=all_features[-1].coordinate_manager,
                coordinate_map_key=all_features[-1].coordinate_map_key,
                device=device,
            )
            coords = [coordinates]
            for _ in reversed(range(len(all_features) - 1)):
                coords.append(self.pooling(coords[-1]))

            coords.reverse()

        pos_encodings_pcd = self.get_pos_encs(coords)

        return pcd_features, all_features, coordinates, pos_encodings_pcd

    def forward_mask(self, pcd_features, aux, coordinates, pos_encodings_pcd, click_idx=None, click_time_idx=None, scan_numbers=None):

        batch_size = pcd_features.C[:, 0].max() + 1

        predictions_mask = [[] for i in range(batch_size)]

        bg_learn_queries = self.bg_query_feat.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        bg_learn_query_pos = self.bg_query_pos.weight.unsqueeze(0).repeat(batch_size, 1, 1)

        for b in range(batch_size):

            mins = coordinates.decomposed_features[b].min(dim=0)[0].unsqueeze(0)
            maxs = coordinates.decomposed_features[b].max(dim=0)[0].unsqueeze(0)

            click_idx_sample = click_idx[b]
            click_time_idx_sample = click_time_idx[b]

            bg_click_idx = click_idx_sample["0"]

            fg_obj_num = len(click_idx_sample.keys()) - 1

            fg_query_num_split = [len(click_idx_sample[str(i)]) for i in range(1, fg_obj_num + 1)]
            fg_query_num = sum(fg_query_num_split)

            fg_clicks_coords = torch.vstack([coordinates.decomposed_features[b][click_idx_sample[str(i)], :] for i in range(1, fg_obj_num + 1)]).unsqueeze(0)
            fg_query_pos = self.pos_enc(fg_clicks_coords.float(), input_range=[mins, maxs])

            fg_clicks_time_idx = list(itertools.chain.from_iterable([click_time_idx_sample[str(i)] for i in range(1, fg_obj_num + 1)]))
            # clip the time index to 1999
            fg_clicks_time_idx = [1999 if x >= 2000 else x for x in fg_clicks_time_idx]
            fg_query_time = self.time_encode[fg_clicks_time_idx].T.unsqueeze(0).to(fg_query_pos.device)

            # generate obj_id embeddings
            current_obj_id = 1
            obj_id_list = []
            for count in fg_query_num_split:
                obj_id_list.extend([current_obj_id] * count)
                current_obj_id += 1
            obj_id_tensor = torch.tensor(obj_id_list, dtype=torch.long).to(fg_query_pos.device)

            fg_query_obj = self.object_embedding(obj_id_tensor).T.unsqueeze(0)
            fg_query_pos = fg_query_pos + fg_query_time + fg_query_obj

            if len(bg_click_idx) != 0:
                bg_click_coords = coordinates.decomposed_features[b][bg_click_idx].unsqueeze(0)
                bg_query_pos = self.pos_enc(bg_click_coords.float(), input_range=[mins, maxs])  # [num_queries, 128]
                bg_query_time = self.time_encode[click_time_idx_sample["0"]].T.unsqueeze(0).to(bg_query_pos.device)
                bg_query_pos = bg_query_pos + bg_query_time

                bg_query_pos = torch.cat([bg_learn_query_pos[b].T.unsqueeze(0), bg_query_pos], dim=-1)
            else:
                bg_query_pos = bg_learn_query_pos[b].T.unsqueeze(0)

            fg_query_pos = fg_query_pos.permute((2, 0, 1))[:, 0, :]  # [num_queries, 128]
            bg_query_pos = bg_query_pos.permute((2, 0, 1))[:, 0, :]  # [num_queries, 128]

            bg_query_num = bg_query_pos.shape[0]
            # with torch.no_grad():
            fg_queries = torch.vstack([pcd_features.decomposed_features[b][click_idx_sample[str(i)], :] for i in range(1, fg_obj_num + 1)])

            if len(bg_click_idx) != 0:
                # with torch.no_grad():
                bg_queries = pcd_features.decomposed_features[b][bg_click_idx, :]
                bg_queries = torch.cat([bg_learn_queries[b], bg_queries], dim=0)
            else:
                bg_queries = bg_learn_queries[b]

            # generate bg (obj 0) obj embedding
            bg_obj_id = torch.zeros(bg_queries.shape[0], dtype=torch.long, device=bg_query_pos.device)
            bg_query_obj = self.object_embedding(bg_obj_id)
            bg_queries += bg_query_obj

            src_pcd = pcd_features.decomposed_features[b]

            if self.sweep_size > 1:
                # Add scan encoding for 4d setup for the attention mechanism
                src_pcd_scan_num_encoding = self.scan_num_encode[scan_numbers.cpu().long()].to(src_pcd.device)
                src_pcd += src_pcd_scan_num_encoding

            refine_time = 0

            for decoder_counter in range(self.num_decoders):
                if self.shared_decoder:
                    decoder_counter = 0
                hlevel = 4
                pos_enc = pos_encodings_pcd[hlevel][0][b]  # [num_points, 128]

                if refine_time == 0:
                    attn_mask = None

                output = self.c2s_attention[decoder_counter][0](
                    torch.cat([fg_queries, bg_queries], dim=0),  # [num_queries, 128]
                    src_pcd,  # [num_points, 128]
                    memory_mask=attn_mask,
                    memory_key_padding_mask=None,
                    pos=pos_enc,  # [num_points, 128]
                    query_pos=torch.cat([fg_query_pos, bg_query_pos], dim=0),  # [num_queries, 128]
                )  # [num_queries, 128]

                output = self.c2c_attention[decoder_counter][0](
                    output,  # [num_queries, 128]
                    tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=torch.cat([fg_query_pos, bg_query_pos], dim=0),  # [num_queries, 128]
                )  # [num_queries, 128]

                # FFN
                queries = self.ffn_attention[decoder_counter][0](output)  # [num_queries, 128]

                src_pcd = self.s2c_attention[decoder_counter][0](
                    src_pcd,
                    queries,  # [num_queries, 128]
                    memory_mask=None,
                    memory_key_padding_mask=None,
                    pos=torch.cat([fg_query_pos, bg_query_pos], dim=0),  # [num_queries, 128]
                    query_pos=pos_enc,  # [num_points, 128]
                )  # [num_points, 128]

                fg_queries, bg_queries = queries.split([fg_query_num, bg_query_num], 0)

                outputs_mask, attn_mask = self.mask_module(fg_queries, bg_queries, src_pcd, ret_attn_mask=True, fg_query_num_split=fg_query_num_split)
                predictions_mask[b].append(outputs_mask)

                refine_time += 1

        predictions_mask = [list(i) for i in zip(*predictions_mask)]

        out = {"pred_masks": predictions_mask[-1], "backbone_features": pcd_features}

        if self.aux:
            out["aux_outputs"] = self._set_aux_loss(predictions_mask)

        return out

    def mask_module(self, fg_query_feat, bg_query_feat, mask_features, ret_attn_mask=True, fg_query_num_split=None):

        fg_query_feat = self.decoder_norm(fg_query_feat)
        fg_mask_embed = self.mask_embed_head(fg_query_feat)

        fg_prods = mask_features @ fg_mask_embed.T
        fg_prods = fg_prods.split(fg_query_num_split, dim=1)

        fg_masks = []
        for fg_prod in fg_prods:
            fg_masks.append(fg_prod.max(dim=-1, keepdim=True)[0])

        fg_masks = torch.cat(fg_masks, dim=-1)

        bg_query_feat = self.decoder_norm(bg_query_feat)
        bg_mask_embed = self.mask_embed_head(bg_query_feat)
        bg_masks = (mask_features @ bg_mask_embed.T).max(dim=-1, keepdim=True)[0]

        output_masks = torch.cat([bg_masks, fg_masks], dim=-1)

        if ret_attn_mask:

            output_labels = output_masks.argmax(1)

            bg_attn_mask = ~(output_labels == 0)  # Masking all the points which were *not* predicted as background
            bg_attn_mask = bg_attn_mask.unsqueeze(0).repeat(bg_query_feat.shape[0], 1)
            # prevent a scenario where a query would be completely masked out and have nothing
            # to attend to, which could cause issues in the attention mechanism.
            bg_attn_mask[torch.where(bg_attn_mask.sum(-1) == bg_attn_mask.shape[-1])] = False

            fg_attn_mask = []
            for fg_obj_id in range(1, fg_masks.shape[-1] + 1):
                fg_obj_mask = ~(output_labels == fg_obj_id)  # Masking all the points which were *not* predicted as the obj_id
                fg_obj_mask = fg_obj_mask.unsqueeze(0).repeat(fg_query_num_split[fg_obj_id - 1], 1)
                # prevent a scenario where a query would be completely masked out and have nothing
                # to attend to, which could cause issues in the attention mechanism.
                fg_obj_mask[torch.where(fg_obj_mask.sum(-1) == fg_obj_mask.shape[-1])] = False
                fg_attn_mask.append(fg_obj_mask)

            fg_attn_mask = torch.cat(fg_attn_mask, dim=0)

            attn_mask = torch.cat([fg_attn_mask, bg_attn_mask], dim=0)

            return output_masks, attn_mask

        return output_masks

    def get_pos_encs(self, coords):
        pos_encodings_pcd = []

        for i in range(len(coords)):
            pos_encodings_pcd.append([[]])
            for coords_batch in coords[i].decomposed_features:
                scene_min = coords_batch.min(dim=0)[0][None, ...]
                scene_max = coords_batch.max(dim=0)[0][None, ...]

                with autocast(enabled=False):
                    tmp = self.pos_enc(coords_batch[None, ...].float(), input_range=[scene_min, scene_max])

                pos_encodings_pcd[-1][0].append(tmp.squeeze(0).permute((1, 0)))

        return pos_encodings_pcd

    def get_random_samples(self, pcd_sizes, curr_sample_size, device):
        rand_idx = []
        mask_idx = []
        for pcd_size in pcd_sizes:
            if pcd_size <= curr_sample_size:
                # we do not need to sample
                # take all points and pad the rest with zeroes and mask it
                idx = torch.zeros(curr_sample_size, dtype=torch.long, device=device)
                midx = torch.ones(curr_sample_size, dtype=torch.bool, device=device)
                idx[:pcd_size] = torch.arange(pcd_size, device=device)
                midx[:pcd_size] = False  # attend to first points
            else:
                # we have more points in pcd as we like to sample
                # take a subset (no padding or masking needed)
                idx = torch.randperm(pcd_size, device=device)[:curr_sample_size]
                midx = torch.zeros(curr_sample_size, dtype=torch.bool, device=device)

            rand_idx.append(idx)
            mask_idx.append(midx)
        return rand_idx, mask_idx

    @torch.jit.unused
    def _set_aux_loss(self, outputs_seg_masks):
        return [{"pred_masks": a} for a in outputs_seg_masks[:-1]]
