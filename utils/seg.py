# ------------------------------------------------------------------------
# Ilya Fradlin
# RWTH Aachen
# ------------------------------------------------------------------------
import torch
import numpy as np

import datasets.datasets_info as datasets_info
from datasets.datasets_info import get_things_stuff_split_kitti360


def mean_iou_single(pred, labels):
    """Calculate the mean IoU for a single object"""
    truepositive = pred * labels
    intersection = torch.sum(truepositive == 1)
    uni = torch.sum(pred == 1) + torch.sum(labels == 1) - intersection

    iou = intersection / uni
    return iou


def mean_iou(pred, labels, obj2label, dataset_type="semantickitti"):
    """Calculate the mean IoU for a batch"""

    assert len(pred) == len(labels)
    bs = len(pred)
    iou_batch = 0.0
    label_mapping = get_label_mapping(dataset_type)
    iou_per_label = {}  # Initialize IoU for the entire batch
    for label_name in label_mapping.values():
        iou_per_label[label_name] = []
    for b in range(bs):
        pred_sample = pred[b]
        labels_sample = labels[b]
        obj_ids = torch.unique(labels_sample)
        obj_ids = obj_ids[obj_ids != 0]
        obj_num = len(obj_ids)
        iou_sample = 0.0
        for obj_id in obj_ids:
            if dataset_type == "semantickitti":
                original_label = label_mapping[obj2label[b][str(int(obj_id.item()))] & 0xFFFF]
            elif "nuScenes" in dataset_type:
                original_label = label_mapping[obj2label[b][str(int(obj_id.item()))] // 1000]
            obj_iou = mean_iou_single(pred_sample == obj_id, labels_sample == obj_id)
            iou_sample += obj_iou
            # Accumulate IoU for each original label
            iou_per_label[original_label].append(obj_iou)

        iou_sample /= obj_num
        iou_batch += iou_sample

    iou_batch /= bs

    # Aggregate iou_per_label across batches
    average_iou_per_label = {label_name: sum(iou_list) / len(iou_list) if iou_list else None for label_name, iou_list in iou_per_label.items()}
    average_iou_per_label = {"miou_class/" + k: v for k, v in average_iou_per_label.items() if v is not None}

    return iou_batch, average_iou_per_label


def mean_iou_validation(pred, labels, obj2label, label_mapping=None, dataset_type="semantickitti"):
    """Calculate the mean IoU for a batch"""
    assert len(pred) == len(labels)
    bs = len(pred)
    iou_batch = 0.0
    # label_mapping = get_label_mapping(dataset_type)[0]
    iou_per_label = {}  # Initialize IoU for the entire batch
    objects_info = {}  # Initialize IoU for the entire batch
    for label_name in label_mapping.values():
        iou_per_label[label_name] = []
    for obj_id, panoptic_label in obj2label[0].items():
        objects_info[obj_id] = {}
        if dataset_type == "semantickitti":
            objects_info[obj_id]["class"] = label_mapping[obj2label[0][obj_id] & 0xFFFF]
        elif "nuScenes" in dataset_type:
            objects_info[obj_id]["class"] = label_mapping[obj2label[0][obj_id] // 1000]
        elif dataset_type == "kitti360":
            objects_info[obj_id]["class"] = label_mapping[obj2label[0][obj_id] // 1000]

    for b in range(bs):
        pred_sample = pred[b]
        labels_sample = labels[b]
        obj_ids = torch.unique(labels_sample)
        obj_ids = obj_ids[obj_ids != 0]
        obj_num = len(obj_ids)
        iou_sample = 0.0
        for obj_id in obj_ids:
            if dataset_type == "semantickitti":
                original_label = label_mapping[obj2label[b][str(int(obj_id.item()))] & 0xFFFF]
            elif "nuScenes" in dataset_type:
                original_label = label_mapping[obj2label[b][str(int(obj_id.item()))] // 1000]
            elif dataset_type == "kitti360":
                original_label = label_mapping[obj2label[b][str(int(obj_id.item()))] // 1000]
            obj_iou = mean_iou_single(pred_sample == obj_id, labels_sample == obj_id)
            objects_info[str(int(obj_id.item()))]["miou"] = obj_iou.item()
            iou_sample += obj_iou
            # Accumulate IoU for each original label
            iou_per_label[original_label].append(obj_iou)

        iou_sample /= obj_num
        iou_batch += iou_sample

    iou_batch /= bs

    # Aggregate iou_per_label across batches
    average_iou_per_label = {label_name: sum(iou_list) / len(iou_list) if iou_list else None for label_name, iou_list in iou_per_label.items()}
    average_iou_per_label = {"miou_class/" + k: v for k, v in average_iou_per_label.items() if v is not None}

    return iou_batch, average_iou_per_label, objects_info


def get_objects_iou(pred, labels):
    """Calculate the mean IoU for a batch"""
    assert len(pred) == len(labels)
    bs = len(pred)
    objects_info = []  # Initialize IoU for the entire batch

    for b in range(bs):
        objects_info.append({})
        pred_sample = pred[b]
        labels_sample = labels[b]
        obj_ids = torch.unique(labels_sample)
        iou_sample = 0.0
        for obj_id in obj_ids:
            obj_iou = mean_iou_single(pred_sample == obj_id, labels_sample == obj_id)
            objects_info[b][int(obj_id)] = obj_iou.item()
            iou_sample += obj_iou

    return objects_info


def mean_iou_scene(pred, labels):
    """Calculate the mean IoU for all target objects in the scene"""
    obj_ids = torch.unique(labels)
    obj_ids = obj_ids[obj_ids != 0]
    obj_num = len(obj_ids)
    iou_sample = 0.0
    iou_dict = {}
    for obj_id in obj_ids:
        obj_iou = mean_iou_single(pred == obj_id, labels == obj_id)
        iou_dict[int(obj_id)] = float(obj_iou)
        iou_sample += obj_iou

    iou_sample /= obj_num

    return iou_sample, iou_dict


def loss_weights(points, clicks, w_min, w_max, delta):
    """Points closer to clicks have bigger weights. Vice versa."""
    pairwise_distances = torch.cdist(points, clicks)
    pairwise_distances, _ = torch.min(pairwise_distances, dim=1)

    weights = w_min + (w_max - w_min) * (1 - torch.clamp(pairwise_distances, max=delta) / delta)

    return weights


def cal_click_loss_weights(batch_idx, raw_coords, labels, click_idx, w_min, w_max, delta):
    """Calculate the loss weights for each point in the point cloud."""
    weights = []

    bs = batch_idx.max() + 1
    for i in range(bs):

        click_idx_sample = click_idx[i]
        sample_mask = batch_idx == i
        raw_coords_sample = raw_coords[sample_mask]
        all_click_idx = [np.array(v) for k, v in click_idx_sample.items()]
        all_click_idx = np.hstack(all_click_idx).astype(np.int64).tolist()
        click_points_sample = raw_coords_sample[all_click_idx]
        weights_sample = loss_weights(raw_coords_sample, click_points_sample, w_min, w_max, delta)
        weights.append(weights_sample)

    return weights


def get_label_mapping(dataset_type):
    if dataset_type == "semantickitti":
        label_mapping = datasets_info.semantickitti_label_mapping
    elif dataset_type == "nuScenes_challenge":
        label_mapping = datasets_info.nuScenes_challenge_label_mapping
    elif dataset_type == "nuScenes_general":
        label_mapping = datasets_info.datnuScenes_general_label_mapping
    return label_mapping


def get_class_name(dataset_type, obj2label, b, obj_id, label_mapping):
    if dataset_type == "semantickitti":
        original_label = label_mapping[obj2label[b][obj_id] & 0xFFFF]
    elif "nuScenes" in dataset_type:
        original_label = label_mapping[obj2label[b][obj_id] // 1000]
    if dataset_type == "kitti360":
        original_label = label_mapping[obj2label[b][obj_id] // 1000]
    return original_label


def get_obj_ids_per_scan(labels_full, num_points_split):
    obj_ids_per_scan = {}
    for idx, sample_labels_full in enumerate(labels_full):
        obj_ids_per_scan[idx] = {}
        start_index = 0
        for scan_index, split_size in enumerate(num_points_split[idx]):
            end_index = start_index + split_size
            scan_sample_labels_full = sample_labels_full[start_index:end_index]
            obj_ids = torch.unique(scan_sample_labels_full)
            obj_ids = obj_ids[obj_ids != 0]
            obj_ids_per_scan[idx][scan_index] = np.asarray(obj_ids.cpu())
            start_index = end_index
    return obj_ids_per_scan


def get_things_stuff_miou(dataset_type, class_IoU_weighted_results, label_mapping, clicks_of_interest):
    if dataset_type == "semantickitti":
        ignore_labels = datasets_info.semantickitti_ignore_labels
        thing_labels = datasets_info.semantickitti_thing_labels
        stuff_labels = datasets_info.semantickitti_stuff_labels
    elif dataset_type == "nuScenes_challenge":
        ignore_labels = datasets_info.nuScenes_challenge_ignore_labels
        thing_labels = datasets_info.nuScenes_challenge_thing_labels
        stuff_labels = datasets_info.nuScenes_challenge_stuff_labels
    elif dataset_type == "nuScenes_general":
        ignore_labels = datasets_info.nuScenes_general_ignore_labels
        thing_labels = datasets_info.nuScenes_general_thing_labels
        stuff_labels = datasets_info.nuScenes_general_stuff_labels
    elif dataset_type == "kitti360":
        ignore_labels = {"unlabeled", "unknown construction", "unknown vehicle", "unknown object"}
        thing_labels, stuff_labels = get_things_stuff_split_kitti360()
    classwise_mIoU_score = {}
    for click_of_interest in clicks_of_interest:
        classwise_mIoU_score[f"classwise_miou_things@{click_of_interest}"] = []
        classwise_mIoU_score[f"classwise_miou_stuff@{click_of_interest}"] = []

    for class_type in label_mapping.values():
        if class_type in ignore_labels:
            continue
        elif class_type in thing_labels:
            is_thing = True
        elif class_type in stuff_labels:
            is_thing = False
        else:
            print("Issue in getting the mIoU for the class")
        for click_of_interest in clicks_of_interest:
            current_miou = class_IoU_weighted_results[class_type][f"IoU@{click_of_interest}"]
            if current_miou is None:
                continue
            if is_thing:
                classwise_mIoU_score[f"classwise_miou_things@{click_of_interest}"].append(current_miou)
            else:
                classwise_mIoU_score[f"classwise_miou_stuff@{click_of_interest}"].append(current_miou)

    classwise_mIoU_score_summed = {}
    for click_of_interest in clicks_of_interest:
        classwise_mIoU_score_summed[f"validation/classwise_metric/miou_things_{click_of_interest}"] = np.mean(classwise_mIoU_score[f"classwise_miou_things@{click_of_interest}"])
        classwise_mIoU_score_summed[f"validation/classwise_metric/miou_stuff_{click_of_interest}"] = np.mean(classwise_mIoU_score[f"classwise_miou_stuff@{click_of_interest}"])

    return classwise_mIoU_score_summed
