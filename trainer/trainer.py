import wandb
import math
import copy
import random
import warnings
import MinkowskiEngine as ME
import torch
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.warnings import PossibleUserWarning

import utils.misc as utils
import datasets.datasets_info as datasets_info
from utils.utils import generate_wandb_objects3d, save_clicks_to_json, save_predictions
from utils.seg import mean_iou, mean_iou_validation, mean_iou_scene, cal_click_loss_weights, get_objects_iou, get_class_name, get_obj_ids_per_scan, get_things_stuff_miou
from utils.click_simulation import extend_clicks, get_simulated_clicks
from datasets.utils import VoxelizeCollate
from datasets.lidar_semantickitti import SemanticKittiDataset
from datasets.lidar_nuscenes import NuscenesDataset
from datasets.lidar_kitti360 import Kitti360Dataset
from models import Interactive4D
from models.criterion import SetCriterion
from models.metrics.utils import IoU_at_numClicks, NumClicks_for_IoU, NumClicks_for_IoU_class, mIoU_per_class_metric, mIoU_metric, losses_metric


class ObjectSegmentation(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Setup Model
        self.interactive4d = Interactive4D(
            num_heads=8,
            num_decoders=3,
            hidden_dim=128,
            dim_feedforward=1024,
            shared_decoder=False,
            num_bg_queries=10,
            dropout=0.0,
            pre_norm=False,
            aux=True,
            voxel_size=config.data.dataloader.voxel_size,
            sample_sizes=[4000, 8000, 16000, 32000],
            sweep_size=self.config.data.datasets.sweep,
        )

        self.dataset_type = self.config.general.dataset
        if self.config.general.dataset == "semantickitti":
            self.label_mapping = datasets_info.semantickitti_label_mapping
        elif self.config.general.dataset == "nuScenes_general":
            self.label_mapping = datasets_info.nuScenes_general_label_mapping
        elif self.config.general.dataset == "nuScenes_challenge":
            self.label_mapping = datasets_info.nuScenes_challenge_label_mapping
        elif self.config.general.dataset == "kitti360":
            self.label_mapping = datasets_info.label_name_mapping_kitti360
        else:
            raise ValueError(f"Unknown dataset type: {self.config.general.dataset}")

        # Setup Loss
        weight_dict = {
            "loss_bce": self.config.loss.bce_loss_coef,
            "loss_dice": self.config.loss.dice_loss_coef,
        }
        if config.loss.aux:
            aux_weight_dict = {}
            for i in range(self.interactive4d.num_decoders):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        self.criterion = SetCriterion(losses=["bce", "dice"], weight_dict=weight_dict)

        # adjust max_num_clicks to the max number of clicks
        self.config.general.max_num_clicks *= self.config.data.datasets.sweep
        self.config.general.max_clicks_per_obj *= self.config.data.datasets.sweep

        # adjust clicks of interest according to the max number of clicks
        self.clicks_of_interest = list(range(1, self.config.general.max_num_clicks + 1))

        # Setup clicking strategies
        self.rank_error_strategy = config.general.rank_error_strategy
        self.initial_clicking_strategy = config.general.initial_clicking_strategy
        self.refinement_clicking_strategy = config.general.refinement_clicking_strategy

        # Initiate the Training metrics
        self.log("mIoU_monitor", 0, sync_dist=True, logger=False)
        self.losses_metric = losses_metric()
        self.mIoU_metric = mIoU_metric()
        self.mIoU_per_class_metric = mIoU_per_class_metric(training=True)

        # Initiate the Validation metrics
        self.iou_at_numClicks = IoU_at_numClicks(num_clicks=self.clicks_of_interest)
        self.iou_at_numClicks_weighted = IoU_at_numClicks(num_clicks=self.clicks_of_interest)
        self.numClicks_for_IoU = NumClicks_for_IoU(iou_thresholds=self.config.general.iou_targets)
        self.numClicks_for_IoU_obj = NumClicks_for_IoU(iou_thresholds=self.config.general.iou_targets)
        # iou@K for each class
        self.iou_at_numClicks_class = {}
        for class_type in self.label_mapping.values():
            self.iou_at_numClicks_class[class_type] = IoU_at_numClicks(num_clicks=self.clicks_of_interest)
        self.numClicks_for_IoU_class = NumClicks_for_IoU_class(iou_thresholds=self.config.general.iou_targets, label_mapping=self.label_mapping)
        self.iou_tracking = {click_of_interest: {} for click_of_interest in self.clicks_of_interest}
        self.validation_metric_logger = utils.MetricLogger(delimiter="  ")

        # Filter out the known warning messages
        warnings.filterwarnings("ignore", message=".*when logging on epoch level in distributed setting to accumulate the metric across devices.*", category=PossibleUserWarning)
        warnings.filterwarnings("ignore", message=".*but the value needs to be floating point.*")

        self.save_hyperparameters()

    def setup(self, stage):
        self.train_dataset = SemanticKittiDataset(
            data_dir=self.config.data.datasets.data_dir,
            sweep=self.config.data.datasets.sweep,
            volume_augmentations_path=self.config.data.datasets.volume_augmentations_path,
            mode="train",
            center_coordinates=self.config.data.datasets.center_coordinates,
            window_overlap=self.config.data.datasets.window_overlap,
        )
        if self.config.general.dataset == "semantickitti":
            self.validation_dataset = SemanticKittiDataset(
                data_dir=self.config.data.datasets.data_dir,
                sweep=self.config.data.datasets.sweep,
                mode="validation",
                center_coordinates=self.config.data.datasets.center_coordinates,
                window_overlap=self.config.data.datasets.window_overlap,
            )
        elif "nuScenes" in self.config.general.dataset:
            self.validation_dataset = NuscenesDataset(
                data_dir=self.config.data.datasets.data_dir,
                mode="validation",
                sweep=self.config.data.datasets.sweep,
                center_coordinates=self.config.data.datasets.center_coordinates,
                dataset_type=self.config.general.dataset,
                window_overlap=self.config.data.datasets.window_overlap,
            )
        elif self.config.general.dataset == "kitti360":
            self.validation_dataset = Kitti360Dataset(
                data_dir=self.config.data.datasets.data_dir,
                mode="validation",
                sweep=self.config.data.datasets.sweep,
                center_coordinates=self.config.data.datasets.center_coordinates,
                window_overlap=self.config.data.datasets.window_overlap,
            )
        else:
            raise ValueError(f"Unknown dataset type: {self.config.general.dataset}")

    def train_dataloader(self):
        effective_batch_size = self.config.data.dataloader.batch_size * self.config.trainer.num_devices * self.config.trainer.num_nodes
        print(f"num devices (self.trainer.num_devices): {self.trainer.num_devices}")
        print(f"train_dataloader - batch_size: {self.config.data.dataloader.batch_size}, effective_batch_size: {effective_batch_size}, num_workers: {self.config.data.dataloader.num_workers}")
        c_fn = VoxelizeCollate(mode="train", voxel_size=self.config.data.dataloader.voxel_size, sweep_size=self.config.data.datasets.sweep)
        return DataLoader(self.train_dataset, shuffle=True, pin_memory=self.config.data.dataloader.pin_memory, num_workers=self.config.data.dataloader.num_workers, batch_size=self.config.data.dataloader.batch_size, collate_fn=c_fn)

    def val_dataloader(self):
        effective_batch_size = self.config.data.dataloader.test_batch_size * self.config.trainer.num_devices * self.config.trainer.num_nodes
        print(f"val_dataloader - batch_size: {self.config.data.dataloader.test_batch_size}, effective_batch_size: {effective_batch_size}, num_workers: {self.config.data.dataloader.num_workers}")
        c_fn = VoxelizeCollate(mode="validation", voxel_size=self.config.data.dataloader.voxel_size, sweep_size=self.config.data.datasets.sweep)
        return DataLoader(self.validation_dataset, shuffle=False, pin_memory=self.config.data.dataloader.pin_memory, num_workers=self.config.data.dataloader.num_workers, batch_size=self.config.data.dataloader.test_batch_size, collate_fn=c_fn)

    def configure_optimizers(self):
        # Optimizer: Adjust the learning rate based on the number of GPUs
        self.config.optimizer.lr = self.config.optimizer.lr * self.config.trainer.num_nodes
        optimizer = AdamW(params=self.parameters(), lr=self.config.optimizer.lr)
        # Scehduler:
        steps_per_epoch = math.ceil(len(self.train_dataloader()) / (self.config.trainer.num_devices * self.config.trainer.num_nodes))
        print(f"optimizer_lr: {self.config.optimizer.lr}, scheduler_steps_per_epoch:{steps_per_epoch}")
        lr_scheduler = OneCycleLR(max_lr=self.config.optimizer.lr, epochs=self.config.trainer.max_epochs, steps_per_epoch=steps_per_epoch, optimizer=optimizer)
        scheduler_config = {"scheduler": lr_scheduler, "interval": "step"}

        return [optimizer], [scheduler_config]

    def training_step(self, batch, batch_idx):
        data, target = batch

        coords = data["coordinates"]
        raw_coords = data["raw_coordinates"]
        feats = data["features"]
        labels = target["labels"]
        click_idx = data["click_idx"]
        obj2label = [mapping[0] for mapping in target["obj2labels"]]
        batch_indicators = coords[:, 0]
        batch_size = batch_indicators.max() + 1

        self.log("scene_size/number_of_points_training", data["number_of_points"], on_step=True)
        self.log("scene_size/number_of_voxels_training", data["number_of_voxels"], prog_bar=True, on_step=True)

        # Check if there is more than just the background in the scene (after quantization)
        click_idx, obj2label, labels = self.verify_labels_post_quantization(labels, click_idx, obj2label, batch_size)
        for idx in range(batch_size):
            if len(labels[idx].unique()) < 2:
                # If there is only the background in the scene, skip the scene
                print("after quantization, only background in the scene")
                print(f"From Rank: {self.global_rank}, The corrupted scene is: '{data['scene_names'][idx]}'!")
                return None

        data = ME.SparseTensor(coordinates=coords, features=feats, device=self.device)
        pcd_features, aux, coordinates, pos_encodings_pcd = self.interactive4d.forward_backbone(data, raw_coordinates=raw_coords)

        click_time_idx = copy.deepcopy(click_idx)

        #########  1. pre interactive sampling  #########
        click_idx, click_time_idx = self.pre_interactive_sampling(
            pcd_features=pcd_features,
            aux=aux,
            coordinates=coordinates,
            raw_coords=raw_coords,
            batch_indicators=batch_indicators,
            pos_encodings_pcd=pos_encodings_pcd,
            labels=labels,
            click_idx=click_idx,
            click_time_idx=click_time_idx,
            scan_numbers=data.F[:, 0],
            rank_strategy=self.rank_error_strategy,
            initial_strategy=self.initial_clicking_strategy,
            refinement_strategy=self.refinement_clicking_strategy,
        )
        self.interactive4d.train()

        #########  2. real forward pass  #########
        outputs = self.interactive4d.forward_mask(pcd_features, aux, coordinates, pos_encodings_pcd, click_idx=click_idx, click_time_idx=click_time_idx, scan_numbers=data.F[:, 0])

        ######### 3. loss back propagation #########
        click_weights = cal_click_loss_weights(batch_indicators, raw_coords, torch.cat(labels), click_idx, self.config.loss.w_min, self.config.loss.w_max, self.config.loss.delta)
        loss_dict = self.criterion(outputs, labels, obj2label, click_weights)
        weight_dict = self.criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            exit(1)

        with torch.no_grad():
            pred_logits = outputs["pred_masks"]
            pred = [p.argmax(-1) for p in pred_logits]
            general_miou, label_miou_dict = mean_iou(pred, labels, obj2label, self.dataset_type)
            label_miou_dict = {"trainer/" + k: v for k, v in label_miou_dict.items()}

        self.log("trainer/loss", losses, prog_bar=True, on_step=True)
        self.log("trainer/mIoU", general_miou, prog_bar=True, on_step=True)
        self.losses_metric.update(losses)
        self.mIoU_metric.update(general_miou)
        self.mIoU_per_class_metric.update(label_miou_dict)

        return losses

    def on_train_epoch_end(self):
        miou_epoch = self.mIoU_metric.compute()
        miou_per_class_epoch = self.mIoU_per_class_metric.compute()
        losses_epoch = self.losses_metric.compute()
        print(f"End Epoch mIoU: {miou_epoch},  loss: {losses_epoch}, class_mIoU: {miou_per_class_epoch}", flush=True)

        self.log_dict(miou_per_class_epoch)
        self.log("mIoU_epoch", miou_epoch)
        self.log("loss_epoch", losses_epoch)

        self.mIoU_metric.reset()
        self.mIoU_per_class_metric.reset()
        self.losses_metric.reset()

    def validation_step(self, batch, batch_idx):
        data, target = batch
        scene_names, coords, raw_coords, full_raw_coords, feats = data["scene_names"], data["coordinates"], data["raw_coordinates"], data["full_coordinates"], data["features"]
        labels, labels_full = target["labels"], [torch.from_numpy(l).to(coords) for l in target["labels_full"]]
        num_points_split, num_obj_split, click_idx, inverse_maps, unique_maps = data["num_points"], data["num_obj"], data["click_idx"], target["inverse_maps"], target["unique_maps"]
        batch_indicators, batch_size = coords[:, 0], coords[:, 0].max() + 1

        self.log("scene_size/number_of_points_validation", float(data["number_of_points"]), on_step=True, batch_size=batch_size)
        self.log("scene_size/number_of_voxels_validation", float(data["number_of_voxels"]), prog_bar=True, on_step=True, batch_size=batch_size)

        error_cluster_dict = {}

        obj2label = [mapping[0] for mapping in target["obj2labels"]]
        num_obj = [math.ceil(sum(obj_count) / self.config.data.datasets.sweep) for obj_count in num_obj_split]
        current_num_clicks = 0

        # Remove objects which are not in the scene (due to quantization)
        click_idx, obj2label, labels = self.verify_labels_post_quantization(labels, click_idx, obj2label, batch_size)
        click_time_idx = copy.deepcopy(click_idx)

        # Check if there is more than just the background in the scene
        for idx in range(batch_size):
            if len(labels[idx].unique()) < 2:
                # If there is only the background in the scene, skip the scene
                print("after quantization, only background in the scene")
                return

        ###### interactive evaluation ######
        data = ME.SparseTensor(coordinates=coords, features=feats, device=self.device)
        pcd_features, aux, coordinates, pos_encodings_pcd = self.interactive4d.forward_backbone(data, raw_coordinates=raw_coords)

        iou_targets = self.config.general.iou_targets
        if iou_targets[-1] != 9999:
            iou_targets.append(9999)  # serving as a stop condition
        max_num_clicks = num_obj[0] * self.config.general.max_num_clicks
        next_iou_target_indices = {idx: 0 for idx in range(batch_size)}

        obj_ids_per_scan = get_obj_ids_per_scan(labels_full, num_points_split)
        next_iou_target_indices_obj = {}
        considered_clicks_per_obj = {}
        for idx in range(batch_size):
            next_iou_target_indices_obj[idx] = {}
            considered_clicks_per_obj[idx] = {}
            for obj_id in click_idx[idx].keys():
                considered_clicks_per_obj[idx][obj_id] = {}
                for threshold in iou_targets:
                    considered_clicks_per_obj[idx][obj_id][threshold] = 0
            for sweep_number in range(self.config.data.datasets.sweep):
                next_iou_target_indices_obj[idx][sweep_number] = {}
                for obj_id in obj_ids_per_scan[idx][sweep_number]:
                    obj_id_str = str(obj_id)
                    if obj_id_str == "0":
                        continue
                    next_iou_target_indices_obj[idx][sweep_number][obj_id_str] = 0

        while current_num_clicks <= max_num_clicks:
            if current_num_clicks == 0:
                pred = [torch.zeros(l.shape).to(coords) for l in labels]
            else:
                outputs = self.interactive4d.forward_mask(pcd_features, aux, coordinates, pos_encodings_pcd, click_idx=click_idx, click_time_idx=click_time_idx, scan_numbers=data.F[:, 0])
                pred_logits = outputs["pred_masks"]
                pred = [p.argmax(-1) for p in pred_logits]

            if current_num_clicks != 0:
                click_weights = cal_click_loss_weights(batch_indicators, raw_coords, torch.cat(labels), click_idx, self.config.loss.w_min, self.config.loss.w_max, self.config.loss.delta)
                loss_dict = self.criterion(outputs, labels, obj2label, click_weights)
                weight_dict = self.criterion.weight_dict
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

                loss_dict_reduced = utils.reduce_dict(loss_dict)
                loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
                loss_dict_reduced_unscaled = {f"{k}_unscaled": v for k, v in loss_dict_reduced.items()}

            updated_pred = []

            for idx in range(batch_size):
                sample_mask = batch_indicators == idx
                sample_pred = pred[idx]

                if current_num_clicks != 0:
                    # update prediction with sparse gt
                    for obj_id, cids in click_idx[idx].items():
                        sample_pred[cids] = int(obj_id)
                    updated_pred.append(sample_pred)

                sample_labels = labels[idx]
                sample_raw_coords = raw_coords[sample_mask]
                sample_pred_full = sample_pred[inverse_maps[idx]]
                sample_labels_full = labels_full[idx]

                ##################################################################
                #################  Split the superimposed scans  #################
                ##################################################################
                start_index = 0
                assert sum(num_points_split[idx]) == len(sample_labels_full)
                for sweep_number, split_size in enumerate(num_points_split[idx]):
                    end_index = start_index + split_size
                    assert (end_index - start_index) == split_size
                    scan_sample_labels_full = sample_labels_full[start_index:end_index]
                    scan_sample_pred_full = sample_pred_full[start_index:end_index]

                    # mean_iou_scene here is calculated for the entire scene (with all the points! not just the ones responsible for the quantized values)
                    sample_iou, per_obj_iou = mean_iou_scene(scan_sample_pred_full, scan_sample_labels_full)
                    self.log(f"validation/full_mIoU_sweep_{sweep_number}", sample_iou, on_step=True, on_epoch=True, batch_size=batch_size, prog_bar=True)

                    # Logging IoU@1, IoU@2, IoU@3, IoU@4, IoU@5
                    # And logging tracking_IoU@XX
                    average_clicks_per_obj = current_num_clicks / num_obj[idx]
                    if average_clicks_per_obj in self.clicks_of_interest:
                        # save the predictions into a file in the same format as the labels for pq / lstq metric calculation
                        if self.config.logging.save_predictions:
                            save_predictions(scan_sample_pred_full, obj2label[idx], scene_names[idx], sweep_number, int(average_clicks_per_obj), self.config.general.prediction_dir, self.config.general.dataset)
                            if sweep_number == 0:
                                # save the clicks into a json file
                                current_scan_id = scene_names[idx][sweep_number].split("/")[-1].split(".")[0]
                                save_clicks_to_json(int(average_clicks_per_obj), click_idx[idx], click_time_idx[idx], unique_maps[idx], current_scan_id, self.config.data.datasets.sweep, self.config.general.prediction_dir)

                        self.iou_at_numClicks.update(iou=sample_iou.item(), noc=average_clicks_per_obj)
                        for obj_id in click_idx[idx].keys():
                            if obj_id == "0" or int(obj_id) not in per_obj_iou:
                                continue
                            self.iou_at_numClicks_weighted.update(iou=sample_iou.item(), noc=average_clicks_per_obj)
                            class_type = get_class_name(self.config.general.dataset, obj2label, idx, str(obj_id), self.label_mapping)
                            # iou_tensor = torch.tensor(per_obj_iou[int(obj_id)], dtype=torch.float32, device=self.device)
                            # average_clicks_per_obj_tensor = torch.tensor(average_clicks_per_obj, dtype=torch.float32, device=self.device)
                            self.iou_at_numClicks_class[class_type].update(iou=per_obj_iou[int(obj_id)], noc=average_clicks_per_obj)
                            current_panoptic_label = obj2label[idx][obj_id]
                            self.iou_tracking[int(average_clicks_per_obj)][current_panoptic_label] = per_obj_iou[int(obj_id)]

                    # Logging NoC@ , NoC@65, NoC@80, NoC@85, NoC@90
                    if iou_targets[next_iou_target_indices[idx]] < sample_iou:
                        while iou_targets[next_iou_target_indices[idx]] < sample_iou:
                            self.numClicks_for_IoU.update(iou=iou_targets[next_iou_target_indices[idx]], noc=average_clicks_per_obj)
                            next_iou_target_indices[idx] += 1
                            if next_iou_target_indices[idx] == len(iou_targets) - 1:
                                break

                    # Logging NoC_obj@50, NoC_obj@65, NoC_obj@80, NoC_obj@85, NoC_obj@90
                    for obj_id in click_idx[idx].keys():
                        if obj_id == "0" or int(obj_id) not in per_obj_iou:
                            continue
                        current_obj_click_count = len(click_idx[idx][obj_id])
                        if iou_targets[next_iou_target_indices_obj[idx][sweep_number][obj_id]] < per_obj_iou[int(obj_id)]:
                            while iou_targets[next_iou_target_indices_obj[idx][sweep_number][obj_id]] < per_obj_iou[int(obj_id)]:
                                current_threshold = iou_targets[next_iou_target_indices_obj[idx][sweep_number][obj_id]]
                                effective_clicks_per_obj = current_obj_click_count - considered_clicks_per_obj[idx][obj_id][current_threshold]
                                required_noc = min(effective_clicks_per_obj, self.config.general.max_num_clicks)
                                self.numClicks_for_IoU_obj.update(iou=iou_targets[next_iou_target_indices_obj[idx][sweep_number][obj_id]], noc=required_noc)
                                class_type = get_class_name(self.config.general.dataset, obj2label, idx, obj_id, self.label_mapping)
                                self.numClicks_for_IoU_class.update(iou=iou_targets[next_iou_target_indices_obj[idx][sweep_number][obj_id]], noc=required_noc, class_type=class_type)
                                next_iou_target_indices_obj[idx][sweep_number][obj_id] += 1
                                considered_clicks_per_obj[idx][obj_id][current_threshold] = current_obj_click_count
                                if next_iou_target_indices_obj[idx][sweep_number][obj_id] == len(iou_targets) - 1:
                                    break

                    start_index = end_index

                ##################################################################
                ############### continue in the superimposed setup  ##############
                ##################################################################

                if updated_pred == []:
                    objects_info = get_objects_iou(pred, labels)
                else:
                    objects_info = get_objects_iou(updated_pred, labels)

                if average_clicks_per_obj < 10:
                    new_clicks, new_clicks_num, new_click_pos, new_click_time, error_cluster_dict = get_simulated_clicks(
                        sample_pred,
                        sample_labels,
                        sample_raw_coords,
                        current_num_clicks,
                        objects_info=objects_info[idx],
                        current_click_idx=click_idx[idx],
                        training=False,
                        error_cluster_dict=error_cluster_dict,
                        max_clicks_per_obj=self.config.general.max_clicks_per_obj,
                        rank_strategy=self.rank_error_strategy,
                        initial_strategy=self.initial_clicking_strategy,
                        refinement_strategy=self.refinement_clicking_strategy,
                    )
                else:
                    # To speed up evaluation, after the 10 click per object we sample multiple clicks and no longer one by one
                    new_clicks, new_clicks_num, new_click_pos, new_click_time, error_cluster_dict = get_simulated_clicks(
                        sample_pred,
                        sample_labels,
                        sample_raw_coords,
                        current_num_clicks,
                        objects_info=objects_info[idx],
                        current_click_idx=click_idx[idx],
                        training=True,
                        error_cluster_dict=error_cluster_dict,
                        max_clicks_per_obj=self.config.general.max_clicks_per_obj,
                        rank_strategy=self.rank_error_strategy,
                        initial_strategy=self.initial_clicking_strategy,
                        refinement_strategy=self.refinement_clicking_strategy,
                    )

                ### add new clicks ###
                if new_clicks is not None:
                    click_idx[idx], click_time_idx[idx] = extend_clicks(click_idx[idx], click_time_idx[idx], new_clicks, new_click_time)

            if current_num_clicks != 0:
                # mean_iou here is calculated just with the points responsible for the quantized values!
                # the full_iou which is calculated for all points was calculated for the individual scans
                general_miou, label_miou_dict, objects_info = mean_iou_validation(updated_pred, labels, obj2label, self.label_mapping, self.dataset_type)

                label_miou_dict = {"validation/" + k: v for k, v in label_miou_dict.items()}
                self.log("validation/quantized_mIoU", general_miou, on_step=True, on_epoch=True, batch_size=batch_size, prog_bar=True)
                self.log_dict(label_miou_dict, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
                self.validation_metric_logger.update(mIoU_quantized=general_miou)
                self.validation_metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()), **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)

            if current_num_clicks == 0 or average_clicks_per_obj >= 10:
                new_clicks_num = num_obj[idx]
            else:
                new_clicks_num = 1
            current_num_clicks += new_clicks_num

        pred = 0

        # Update NoC@target for all the targets that have not been reached
        for idx in range(batch_size):
            while next_iou_target_indices[idx] != len(iou_targets) - 1:
                self.numClicks_for_IoU.update(iou=iou_targets[next_iou_target_indices[idx]], noc=self.config.general.max_num_clicks)
                next_iou_target_indices[idx] += 1

        # Update NoC_obj@target for all the targets that have not been reached
        for idx in range(batch_size):
            start_index = 0
            for sweep_number, split_size in enumerate(num_points_split[idx]):
                end_index = start_index + split_size
                scan_sample_labels_full = sample_labels_full[start_index:end_index]
                scan_sample_pred_full = sample_pred_full[start_index:end_index]
                for obj_id in click_idx[idx].keys():
                    if obj_id == "0" or int(obj_id) not in obj_ids_per_scan[idx][sweep_number]:
                        continue
                    while next_iou_target_indices_obj[idx][sweep_number][obj_id] != len(iou_targets) - 1:
                        self.numClicks_for_IoU_obj.update(iou=iou_targets[next_iou_target_indices_obj[idx][sweep_number][obj_id]], noc=self.config.general.max_num_clicks)
                        class_type = get_class_name(self.config.general.dataset, obj2label, idx, obj_id, self.label_mapping)
                        self.numClicks_for_IoU_class.update(iou=iou_targets[next_iou_target_indices_obj[idx][sweep_number][obj_id]], noc=self.config.general.max_num_clicks, class_type=class_type)
                        next_iou_target_indices_obj[idx][sweep_number][obj_id] += 1
                start_index = end_index

        # logging visualization into wandb
        if (self.config.logging.visualization_frequency is not None) and (batch_idx % self.config.logging.visualization_frequency == 0):
            # choose a random scene to visualize from the batch
            chosen_scene_idx = random.randint(0, batch_size - 1)
            scene_name = scene_names[chosen_scene_idx][0]
            scene_name = scene_name.split("/")[-1].split(".")[0]
            sample_raw_coords_full = full_raw_coords[chosen_scene_idx]
            sample_mask = batch_indicators == chosen_scene_idx
            sample_raw_coords = raw_coords[sample_mask]
            sample_pred = updated_pred[chosen_scene_idx]
            sample_labels = labels[chosen_scene_idx]
            sample_click_idx = click_idx[chosen_scene_idx]

            gt_scene, gt_full_scene, pred_scene, pred_full_scene = generate_wandb_objects3d(sample_raw_coords, sample_raw_coords_full, sample_labels, sample_labels_full, sample_pred, sample_pred_full, sample_click_idx, objects_info)
            wandb.log({f"point_scene/ground_truth_ quantized_{scene_name}": gt_scene})
            wandb.log({f"point_scene/ground_truth_full_{scene_name}": gt_full_scene})
            wandb.log({f"point_scene/prediction_{scene_name}_full_sample_iou_{sample_iou:.2f}": pred_full_scene})
            wandb.log({f"point_scene/prediction_{scene_name}_quantized_iou_{general_miou:.2f}": pred_scene})

    def on_validation_epoch_end(self):
        print("\n")
        print("--------- Evaluating Validation Performance  -----------")
        warnings.filterwarnings("ignore", message="The ``compute`` method of metric NumClicks_for_IoU was called before the ``update`` method", category=UserWarning)
        results_dict = {}
        results_dict["mIoU_quantized"] = self.validation_metric_logger.meters["mIoU_quantized"].global_avg
        # Evaluate the NoC@IoU Metric
        metrics_dictionary, iou_thresholds = self.numClicks_for_IoU.compute()
        metrics_dictionary_obj, iou_thresholds_obj = self.numClicks_for_IoU_obj.compute()
        metrics_dictionary_class, iou_thresholds_class = self.numClicks_for_IoU_class.compute()
        for iou in iou_thresholds:
            noc = metrics_dictionary[iou]["noc"]
            count = metrics_dictionary[iou]["count"]
            results_dict[f"scenes_reached_{iou}_iou"] = count.item()
            if count == 0:
                results_dict[f"NoC@{iou}"] = 0  # or return a default value or raise an error
            else:
                results_dict[f"NoC@{iou}"] = (noc / count).item()

        for iou in iou_thresholds_obj:
            noc = metrics_dictionary_obj[iou]["noc"]
            count = metrics_dictionary_obj[iou]["count"]
            results_dict[f"scenes_reached_{iou}_iou"] = count.item()
            if count == 0:
                results_dict[f"NoC_obj@{iou}"] = 0  # or return a default value or raise an error
            else:
                results_dict[f"NoC_obj@{iou}"] = (noc / count).item()

        class_noc = {}
        for iou in iou_thresholds_class:
            for class_type in self.label_mapping.values():
                noc = metrics_dictionary_class[class_type][iou]["noc"]
                count = metrics_dictionary_class[class_type][iou]["count"]
                if count == 0:
                    tmp = 0
                else:
                    tmp = (noc / count).item()
                class_noc[f"validation/noc_class/NoC_{class_type}@{iou}"] = tmp

        # Evaluate the IoU@NoC Metric
        metrics_dictionary, evaluated_num_clicks = self.iou_at_numClicks.compute()
        metrics_dictionary_weighted, evaluated_num_clicks_weighted = self.iou_at_numClicks_weighted.compute()
        for noc in evaluated_num_clicks:
            iou = metrics_dictionary[noc]["iou"]
            count = metrics_dictionary[noc]["count"]
            iou_weighted = metrics_dictionary_weighted[noc]["iou"]
            count_weighted = metrics_dictionary_weighted[noc]["count"]
            if count == 0:
                results_dict[f"IoU@{noc}"] = 0  # or return a default value or raise an error
            else:
                results_dict[f"IoU@{noc}"] = (iou / count).item()
            if count_weighted == 0:
                results_dict[f"IoU_weighted@{noc}"] = 0  # or return a default value or raise an error
            else:
                results_dict[f"IoU_weighted@{noc}"] = (iou_weighted / count_weighted).item()

        # evaluate the IoU@NoC per class
        class_IoU_weighted_results = {}
        class_IoU_weighted_results_extended = {}
        for class_type in self.label_mapping.values():
            class_IoU_weighted_results_extended[class_type] = {}
            self.iou_at_numClicks_class[class_type].to(self.device)
            metrics_dictionary, evaluated_num_clicks = self.iou_at_numClicks_class[class_type].compute()
            for noc in evaluated_num_clicks:
                iou = metrics_dictionary[noc]["iou"]
                count = metrics_dictionary[noc]["count"]
                if count == 0:
                    class_IoU_weighted_results[f"IoU_{class_type}@{noc}"] = -1  # or return a default value or raise an error
                    class_IoU_weighted_results_extended[class_type][f"IoU@{noc}"] = None
                    # print(f"The count is 0 for the class type: {class_type}")
                else:
                    class_IoU_weighted_results[f"validation/IoU_per_class/IoU_{class_type}@{noc}"] = (iou / count).item()
                    class_IoU_weighted_results_extended[class_type][f"IoU@{noc}"] = (iou / count).item()

        classwise_mIoU_score = get_things_stuff_miou(self.dataset_type, class_IoU_weighted_results_extended, self.label_mapping, self.clicks_of_interest)

        # print(class_IoU_weighted_results)
        # print("\n")
        # print(results_dict)
        self.log_dict(class_IoU_weighted_results)
        self.log_dict(classwise_mIoU_score)

        stats = {k: meter.global_avg for k, meter in self.validation_metric_logger.meters.items()}
        stats.update(results_dict)
        if "IoU@6" in stats.keys():
            self.log_dict(
                {
                    "validation/epoch": self.current_epoch,
                    "validation/loss_epoch": stats["loss"],
                    "validation/loss_bce_epoch": stats["loss_bce"],
                    "validation/loss_dice_epoch": stats["loss_dice"],
                    "validation/mIoU_quantized_epoch": stats["mIoU_quantized"],
                    "validation/Interactive_metrics/NoC_50_scene": stats["NoC@50"],
                    "validation/Interactive_metrics/NoC_obj_50_scene": stats["NoC_obj@50"],
                    "validation/Interactive_metrics/NoC_65_scene": stats["NoC@65"],
                    "validation/Interactive_metrics/NoC_obj_65_scene": stats["NoC_obj@65"],
                    "validation/Interactive_metrics/NoC_80_scene": stats["NoC@80"],
                    "validation/Interactive_metrics/NoC_obj_80_scene": stats["NoC_obj@80"],
                    "validation/Interactive_metrics/NoC_85_scene": stats["NoC@85"],
                    "validation/Interactive_metrics/NoC_obj@85": stats["NoC_obj@85"],
                    "validation/Interactive_metrics/NoC_90_Scene": stats["NoC@90"],
                    "validation/Interactive_metrics/NoC_obj_90_Scene": stats["NoC_obj@90"],
                    "validation/Interactive_metrics/IoU_1": stats["IoU@1"],
                    "validation/Interactive_metrics/IoU_2": stats["IoU@2"],
                    "validation/Interactive_metrics/IoU_3": stats["IoU@3"],
                    "validation/Interactive_metrics/IoU_4": stats["IoU@4"],
                    "validation/Interactive_metrics/IoU_5": stats["IoU@5"],
                    "validation/Interactive_metrics/IoU_6": stats["IoU@6"],
                    "validation/Interactive_metrics/IoU_7": stats["IoU@7"],
                    "validation/Interactive_metrics/IoU_8": stats["IoU@8"],
                    "validation/Interactive_metrics/IoU_9": stats["IoU@9"],
                    "validation/Interactive_metrics/IoU_10": stats["IoU@10"],
                }
            )
            # Initialize a dictionary to store the metrics
            metrics_to_log_iou_weighted = {}
            # Iterate through the stats dictionary
            for key, value in stats.items():
                # Check if the key starts with "IoU_weighted@"
                if key.startswith("IoU_weighted@"):
                    # Extract the number from the key
                    number = key.split("@")[1]
                    # Create the new key for wandb
                    new_key = f"validation/Interactive_metrics/IoU_weighted_{number}"
                    # Add to the dictionary
                    metrics_to_log_iou_weighted[new_key] = value

            # Log the dictionary to wandb
            # print(metrics_to_log_iou_weighted)
            self.log_dict(metrics_to_log_iou_weighted)

        else:
            self.log_dict(
                {
                    "validation/epoch": self.current_epoch,
                    "validation/loss_epoch": stats["loss"],
                    "validation/loss_bce_epoch": stats["loss_bce"],
                    "validation/loss_dice_epoch": stats["loss_dice"],
                    "validation/mIoU_quantized_epoch": stats["mIoU_quantized"],
                    "validation/Interactive_metrics/NoC_50_scene": stats["NoC@50"],
                    "validation/Interactive_metrics/NoC_obj_50_scene": stats["NoC_obj@50"],
                    "validation/Interactive_metrics/NoC_65_scene": stats["NoC@65"],
                    "validation/Interactive_metrics/NoC_obj_65_scene": stats["NoC_obj@65"],
                    "validation/Interactive_metrics/NoC_80_scene": stats["NoC@80"],
                    "validation/Interactive_metrics/NoC_obj_80_scene": stats["NoC_obj@80"],
                    "validation/Interactive_metrics/NoC_85_scene": stats["NoC@85"],
                    "validation/Interactive_metrics/NoC_obj_85_scene": stats["NoC_obj@85"],
                    "validation/Interactive_metrics/NoC_90_Scene": stats["NoC@90"],
                    "validation/Interactive_metrics/NoC_obj_90_Scene": stats["NoC_obj@90"],
                    "validation/Interactive_metrics/IoU_1": stats["IoU@1"],
                    "validation/Interactive_metrics/IoU_2": stats["IoU@2"],
                    "validation/Interactive_metrics/IoU_3": stats["IoU@3"],
                    "validation/Interactive_metrics/IoU_4": stats["IoU@4"],
                    "validation/Interactive_metrics/IoU_5": stats["IoU@5"],
                }
            )
        # self.validation_step_outputs.clear()  # free memory
        self.log_dict(class_noc)

        # evaluate iou tracking
        # print("IoU Tracking Results:")
        for click_of_interest in self.clicks_of_interest:
            # print(f"Tracking IoU at Number of Clicks: {click_of_interest}")
            total_iou = 0
            total_count = 0
            for panoptic_label, iou in self.iou_tracking[click_of_interest].items():
                # print(f"{panoptic_label}: {iou}")
                total_iou += iou
                total_count += 1
            tmp = total_iou / total_count
            click_of_interest_int = int(click_of_interest)
            self.log_dict({f"validation/Interactive_metrics/Tracking_IoU_{click_of_interest_int}": tmp})
        self.validation_metric_logger = utils.MetricLogger(delimiter="  ")  # reset metric

    def pre_interactive_sampling(
        self,
        pcd_features,
        aux,
        coordinates,
        raw_coords,
        batch_indicators,
        pos_encodings_pcd,
        labels,
        click_idx,
        click_time_idx,
        scan_numbers=None,
        rank_strategy="boundary_dependent",
        initial_strategy="centroid",
        refinement_strategy="random",
    ):
        batch_size = batch_indicators.max() + 1
        current_num_iter = 0
        error_cluster_dict = {}
        num_forward_iters = random.randint(0, self.config.general.max_num_clicks - 1)

        with torch.no_grad():
            self.interactive4d.eval()
            eval_model = self.interactive4d
            while current_num_iter <= num_forward_iters:
                if current_num_iter == 0:
                    pred = [torch.zeros(l.shape).to(raw_coords) for l in labels]
                else:
                    outputs = eval_model.forward_mask(
                        pcd_features,
                        aux,
                        coordinates,
                        pos_encodings_pcd,
                        click_idx=click_idx,
                        click_time_idx=click_time_idx,
                        scan_numbers=scan_numbers,
                    )
                    pred_logits = outputs["pred_masks"]
                    pred = [p.argmax(-1) for p in pred_logits]

                for idx in range(batch_size):
                    sample_mask = batch_indicators == idx
                    sample_pred = pred[idx]

                    if current_num_iter != 0:
                        # update prediction with sparse gt
                        for obj_id, cids in click_idx[idx].items():
                            sample_pred[cids] = int(obj_id)

                    sample_labels = labels[idx]
                    sample_raw_coords = raw_coords[sample_mask]

                    objects_info = get_objects_iou(pred, labels)

                    new_clicks, new_clicks_num, new_click_pos, new_click_time, error_cluster_dict = get_simulated_clicks(
                        sample_pred,
                        sample_labels,
                        sample_raw_coords,
                        current_num_iter,
                        objects_info=objects_info[idx],
                        current_click_idx=click_idx[idx],
                        training=True,
                        error_cluster_dict=error_cluster_dict,
                        max_clicks_per_obj=self.config.general.max_clicks_per_obj,
                        rank_strategy=rank_strategy,
                        initial_strategy=initial_strategy,
                        refinement_strategy=refinement_strategy,
                    )

                    ### add new clicks ###
                    if new_clicks is not None:
                        click_idx[idx], click_time_idx[idx] = extend_clicks(click_idx[idx], click_time_idx[idx], new_clicks, new_click_time)

                current_num_iter += 1

            return click_idx, click_time_idx

    def verify_labels_post_quantization(self, labels, click_idx, obj2label, batch_size):
        """Remove objects which are not in the scene (due to quantization) and update the labels accordingly
        This is done during training to ignore mistakes not caused by the model itself"""

        obj_to_remove = []
        for i in range(batch_size):
            unique_labels_after_qunatization = labels[i].unique()
            unique_labels_after_qunatization = {int(label) for label in unique_labels_after_qunatization}
            obj_to_remove.extend((i, key) for key in click_idx[i] if int(key) not in unique_labels_after_qunatization)

        if obj_to_remove:
            for i, key in obj_to_remove:
                if key != "0":
                    del click_idx[i][key]
                    del obj2label[i][key]

            for i in range(batch_size):
                mapping = {old_key: new_key for new_key, old_key in enumerate(sorted(click_idx[i].keys(), key=int))}
                click_idx[i] = {str(j): click_idx[i][old_key] for j, old_key in enumerate(sorted(click_idx[i].keys(), key=int))}
                obj2label[i] = {str(j): obj2label[i][old_key] for j, old_key in enumerate(sorted(obj2label[i].keys(), key=int), start=1)}

                # Update the keys in labels[i] using the mapping dictionary
                for old_id, new_id in mapping.items():
                    labels[i][labels[i] == int(old_id)] = new_id

        return click_idx, obj2label, labels


class CustomAdamW(AdamW):
    # A wrapper class to ensure that the state steps are on the CPU (used for multi-GPU resume training)
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                # Ensure state steps are on the CPU
                if "step" in state and state["step"].is_cuda:
                    state["step"] = state["step"].cpu()
        super().step(closure)
