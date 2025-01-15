import torch
import os
import json
import sys
import wandb
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans

if sys.version_info[:2] >= (3, 8):
    from collections.abc import MutableMapping
else:
    from collections import MutableMapping


def flatten_dict(d, parent_key="", sep="_"):
    """
    https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def save_predictions(scan_sample_pred_full, obj2label, scene_names, sweep_number, average_clicks_per_obj, prediction_dir, dataset_type):
    scan_sample_pred_full_cpu = scan_sample_pred_full.cpu().numpy()
    mapped_predictions = np.array([obj2label[str(point_pred)] if str(point_pred) in obj2label else point_pred for point_pred in scan_sample_pred_full_cpu])
    mapped_predictions = mapped_predictions.astype(np.uint32)
    if dataset_type == "semantickitti":
        learning_map_inv = {0: 0, 1: 10, 2: 11, 3: 15, 4: 18, 5: 20, 6: 30, 7: 31, 8: 32, 9: 40, 10: 44, 11: 48, 12: 49, 13: 50, 14: 51, 15: 70, 16: 71, 17: 72, 18: 80, 19: 81}
        semantic_preds = mapped_predictions & 0xFFFF
        semantic_preds = np.vectorize(learning_map_inv.__getitem__)(semantic_preds)
        instance_preds = mapped_predictions >> 16
        mapped_predictions = (instance_preds << 16) + semantic_preds
    else:
        raise ValueError(f"Saving prediction for dataset type is supported only for 'semantickitti' dataset. Got {dataset_type}.")
    current_scan_id = scene_names[sweep_number].split("/")[-1].split(".")[0]
    average_clicks_folder = f"average_{average_clicks_per_obj}_clicks"
    base_path, remaining_path = prediction_dir.rsplit("sequences", 1)
    updated_prediction_dir = os.path.join(base_path, average_clicks_folder, "sequences" + remaining_path)
    updated_prediction_dir = Path(updated_prediction_dir)
    output_filepath = updated_prediction_dir / f"{current_scan_id}.label"
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    if not os.path.exists(output_filepath):
        with open(output_filepath, "wb") as f:
            f.write(mapped_predictions.astype(np.uint32).tobytes())


def save_clicks_to_json(average_clicks_per_obj, click_idx, click_time_idx, unique_maps, current_scan_id, sweep, prediction_dir):
    # save the clicks into a file (after mapping them to the full point cloud)
    current_scan_id_int = int(current_scan_id)
    if sweep == 1:
        end_scan_id_int = current_scan_id_int
    else:
        end_scan_id_int = current_scan_id_int + sweep - 1
    end_scan_id = str(end_scan_id_int).zfill(len(current_scan_id))
    filename = f"clicks_sweep_{current_scan_id}_{end_scan_id}.json"
    average_clicks_folder = f"average_{average_clicks_per_obj}_clicks"
    base_path, _ = prediction_dir.rsplit("sequences", 1)

    updated_prediction_dir = os.path.join(base_path, "used_clicks", average_clicks_folder)
    clicks_filepath = os.path.join(updated_prediction_dir, f"average_{average_clicks_per_obj}_clicks", filename)
    original_clicks = {}
    for key, voxel_indices in click_idx.items():
        original_clicks[key] = [unique_maps[voxel_idx].item() for voxel_idx in voxel_indices]
    click_data = {"original_clicks": original_clicks, "click_time_idx": click_time_idx}
    clicks_filepath = os.path.join(updated_prediction_dir, f"clicks_sweep_{current_scan_id}_{end_scan_id}.json")
    os.makedirs(os.path.dirname(clicks_filepath), exist_ok=True)
    # Save the combined data as a JSON file
    with open(clicks_filepath, "w") as f:
        json.dump(click_data, f, indent=4)


######################################################
###########  wandb visualization functions ###########
######################################################
def calculate_bounding_box_corners(min_x, max_x, min_y, max_y, min_z, max_z):
    corners = []
    for x in (min_x, max_x):
        for y in (min_y, max_y):
            for z in (min_z, max_z):
                corners.append([x, y, z])
    return corners


def generate_wandb_objects3d(raw_coords, raw_coords_full, labels, labels_full, pred, sample_pred_full, click_idx, objects_info):
    # Ensure inputs are PyTorch tensors
    if not isinstance(raw_coords_full, torch.Tensor):
        raw_coords_full = torch.tensor(raw_coords_full, device=raw_coords.device)
    if not isinstance(labels_full, torch.Tensor):
        labels_full = torch.tensor(labels_full, device=raw_coords.device)
    if not isinstance(sample_pred_full, torch.Tensor):
        sample_pred_full = torch.tensor(sample_pred_full, device=raw_coords.device)

    # Create a mapping from label to color
    unique_labels = torch.unique(labels_full)
    # Check if 0 is in unique_labels
    if not (unique_labels == 0).any():
        # If 0 is not in unique_labels, concatenate 0 to it
        zero_tensor = torch.tensor([0], device=unique_labels.device)
        unique_labels = torch.cat((unique_labels, zero_tensor))
    num_labels = unique_labels.size(0)
    distinct_colors = generate_distinct_colors_kmeans(num_labels)
    label_to_color = {label.item(): distinct_colors[i] for i, label in enumerate(unique_labels)}

    # Add a new dimension to labels and pred
    labels = torch.unsqueeze(labels, dim=1)
    pred = torch.unsqueeze(pred, dim=1)
    labels_full = torch.unsqueeze(labels_full, dim=1)
    sample_pred_full = torch.unsqueeze(sample_pred_full, dim=1)

    # Prepare arrays to hold coordinates and corresponding colors
    pcd_gt = torch.cat((raw_coords, labels), dim=1).cpu().numpy()
    pcd_gt_full = torch.cat((raw_coords_full, labels_full), dim=1).cpu().numpy()
    pcd_pred = torch.cat((raw_coords, pred), dim=1).cpu().numpy()
    pcd_pred_full = torch.cat((raw_coords_full, sample_pred_full), dim=1).cpu().numpy()

    # Initialize arrays for points with RGB values
    pcd_gt_rgb = np.zeros((pcd_gt.shape[0], 6))
    pcd_pred_rgb = np.zeros((pcd_pred.shape[0], 6))
    pcd_pred_full_rgb = np.zeros((pcd_pred_full.shape[0], 6))
    pcd_gt_full_rgb = np.zeros((pcd_gt_full.shape[0], 6))
    # Fill the arrays with coordinates and RGB values
    pcd_gt_rgb[:, :3] = pcd_gt[:, :3]
    pcd_pred_rgb[:, :3] = pcd_pred[:, :3]
    pcd_pred_full_rgb[:, :3] = pcd_pred_full[:, :3]
    pcd_gt_full_rgb[:, :3] = pcd_gt_full[:, :3]
    for i in range(pcd_gt.shape[0]):
        label = int(pcd_gt[i, 3])
        color = label_to_color[label]
        pcd_gt_rgb[i, 3:] = color
    for i in range(pcd_pred.shape[0]):
        label = int(pcd_pred[i, 3])
        color = label_to_color[label]
        pcd_pred_rgb[i, 3:] = color
    for i in range(pcd_gt_full.shape[0]):
        label = int(pcd_gt_full[i, 3])
        color = label_to_color[label]
        pcd_gt_full_rgb[i, 3:] = color
    for i in range(pcd_pred_full.shape[0]):
        label = int(pcd_pred_full[i, 3])
        color = label_to_color[label]
        pcd_pred_full_rgb[i, 3:] = color

    ####################################################################################
    ############### Get the Predicted and clicks as small Bounding Boxes ###############
    ####################################################################################
    boxes_array = []
    # Iterate over each object in sample_click_idx
    for obj, click_indices in click_idx.items():
        if obj == "0":
            obj_class = "background/outlier"
            obj_iou = 0.0
        else:
            obj_class = objects_info[obj]["class"]
            obj_iou = objects_info[obj]["miou"]
        # Extract click points from numpy_array
        click_points = pcd_pred[click_indices]
        max_clicks_for_obj = len(click_points)
        # Calculate bounding box coordinates
        for i, click in enumerate(click_points):
            min_x, max_x = click[0] - 0.1, click[0] + 0.1
            min_y, max_y = click[1] - 0.1, click[1] + 0.1
            min_z, max_z = click[2] - 0.1, click[2] + 0.1

            current_box_click = {
                "corners": calculate_bounding_box_corners(min_x, max_x, min_y, max_y, min_z, max_z),
                "label": f"{obj}-{i+1}/{max_clicks_for_obj}-{obj_class}-{obj_iou:.02f}",
                "color": [255, 0, 255],
            }
            boxes_array.append(current_box_click)

    gt_scene = wandb.Object3D({"type": "lidar/beta", "points": pcd_gt_rgb})
    pred_scene = wandb.Object3D({"type": "lidar/beta", "points": pcd_pred_rgb, "boxes": np.array(boxes_array)})
    gt_full_scene = wandb.Object3D({"type": "lidar/beta", "points": pcd_gt_full_rgb})
    pred_full_scene = wandb.Object3D({"type": "lidar/beta", "points": pcd_pred_full_rgb})

    return gt_scene, gt_full_scene, pred_scene, pred_full_scene


def generate_distinct_colors_kmeans(n):
    # Sample a large number of colors in RGB space
    np.random.seed(0)
    large_sample = np.random.randint(0, 256, (10000, 3))

    # Apply k-means clustering to find n clusters
    kmeans = KMeans(n_clusters=n, n_init=1).fit(large_sample)
    colors = kmeans.cluster_centers_.astype(int)

    return [tuple(color) for color in colors]


def labels_to_colors(labels):
    import numpy as np

    unique_labels = np.unique(labels.cpu())
    num_colors = len(unique_labels)
    # Generate distinct colors
    colors_list = generate_distinct_colors_kmeans(num_colors)
    # Create a mapping from labels to colors
    label_to_color = {label: colors_list[i] for i, label in enumerate(unique_labels)}
    # Apply the color map
    colors = np.array([label_to_color[int(label.item())] for label in labels])
    return colors
