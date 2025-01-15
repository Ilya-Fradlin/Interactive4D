# ------------------------------------------------------------------------
# Ilya Fradlin
# RWTH Aachen
# ------------------------------------------------------------------------
import torch
import random
import numpy as np

from sklearn.cluster import DBSCAN


def get_simulated_clicks(pred_qv, labels_qv, coords_qv, current_num_clicks=None, current_click_idx=None, training=True, objects_info={}, error_cluster_dict={}, max_clicks_per_obj=20, rank_strategy="boundary_dependent", initial_strategy="centroid", refinement_strategy="random"):
    if rank_strategy == "BD":
        return get_simulated_clicks_BD(pred_qv, labels_qv, coords_qv, current_num_clicks, current_click_idx, training, max_clicks_per_obj)
    elif rank_strategy == "SI":
        return get_simulated_clicks_SI(pred_qv, labels_qv, coords_qv, current_num_clicks, current_click_idx, training, objects_info, error_cluster_dict, max_clicks_per_obj, initial_strategy, refinement_strategy)
    else:
        raise NotImplementedError


def get_simulated_clicks_BD(pred_qv, labels_qv, coords_qv, current_num_clicks, current_click_idx, training, max_clicks_per_obj):
    """Sample simulated clicks .
    The simulation samples next clicks from the top biggest error regions in the current iteration.
    """
    labels_qv = labels_qv.float()
    pred_label = pred_qv.float()

    # Do not generate new clicks for obj that has been clicked more than the threshold
    if current_click_idx is not None:
        for obj_id, click_ids in current_click_idx.items():
            # if obj_id != "0":  # background can receive as many clicks as needed
            if len(click_ids) >= max_clicks_per_obj:
                # Artificially set the pred_label to labels_qv for this object (as it received the threshold number of clicks)
                pred_label[labels_qv == int(obj_id)] = int(obj_id)

    error_mask = torch.abs(pred_label - labels_qv) > 0

    if error_mask.sum() == 0:
        return None, None, None, None, None

    cluster_ids = labels_qv * 9973 + pred_label * 11

    num_obj = (torch.unique(labels_qv) != 0).sum()

    error_clusters = cluster_ids[error_mask]
    error_cluster_ids = torch.unique(error_clusters)
    num_error_cluster = len(error_cluster_ids)

    error_cluster_ids_mask = torch.ones(coords_qv.shape[0], device=error_mask.device) * -1
    error_cluster_ids_mask[error_mask] = error_clusters

    ### measure the size of each error cluster and store the distance
    error_sizes, error_distances = {}, {}

    for cluster_id in error_cluster_ids:
        error = error_cluster_ids_mask == cluster_id
        pairwise_distances = measure_bd_error_size(coords_qv, error)
        error_distances[int(cluster_id)] = pairwise_distances
        error_sizes[int(cluster_id)] = torch.max(pairwise_distances).tolist()

    error_cluster_ids_sorted = sorted(error_sizes, key=error_sizes.get, reverse=True)

    if training:
        if num_error_cluster >= num_obj:
            selected_error_cluster_ids = error_cluster_ids_sorted[:num_obj]
        else:
            selected_error_cluster_ids = error_cluster_ids_sorted
    else:
        if current_num_clicks == 0:
            selected_error_cluster_ids = error_cluster_ids_sorted
        else:
            selected_error_cluster_ids = error_cluster_ids_sorted[:1]

    new_clicks, new_click_num, new_click_pos, new_click_time = get_next_simulated_click_BD(selected_error_cluster_ids, error_cluster_ids_mask, pred_qv, labels_qv, coords_qv, error_distances)

    error_cluster_dict = {}  # in the BD strategy, we do not need to maintain the error_cluster_dict
    return new_clicks, new_click_num, new_click_pos, new_click_time, error_cluster_dict


def get_next_simulated_click_BD(error_cluster_ids, error_cluster_ids_mask, pred_qv, labels_qv, coords_qv, error_distances):
    """Sample the next clicks for each error region"""

    click_dict = {}
    new_click_pos = {}
    click_time_dict = {}
    click_order = 0

    random.shuffle(error_cluster_ids)

    for cluster_id in error_cluster_ids:

        error = error_cluster_ids_mask == cluster_id

        pair_distances = error_distances[cluster_id]

        # get next click candidate
        center_id, center_coo, center_gt, max_dist, candidates = get_next_click_bd_multi_errors(coords_qv, error, labels_qv, pred_qv, pair_distances)

        if click_dict.get(str(int(center_gt))) == None:
            click_dict[str(int(center_gt))] = [int(center_id)]
            new_click_pos[str(int(center_gt))] = [center_coo]
            click_time_dict[str(int(center_gt))] = [click_order]
        else:
            click_dict[str(int(center_gt))].append(int(center_id))
            new_click_pos[str(int(center_gt))].append(center_coo)
            click_time_dict[str(int(center_gt))].append(click_order)

        click_order += 1

    click_num = len(error_cluster_ids)

    return click_dict, click_num, new_click_pos, click_time_dict


def measure_bd_error_size(discrete_coords, unique_labels):
    """Measure error size in 3D space"""
    torch.cuda.empty_cache()

    zero_indices = unique_labels == 0  # background
    one_indices = unique_labels == 1  # foreground
    if zero_indices.sum() == 0 or one_indices.sum() == 0:
        return None, None, None, -1, None, None

    # All distances from foreground points to background points
    pairwise_distances = torch.cdist(discrete_coords[zero_indices, :], discrete_coords[one_indices, :])
    # Bg points on the border
    pairwise_distances, _ = torch.min(pairwise_distances, dim=0)

    return pairwise_distances


def get_simulated_clicks_SI(pred_qv, labels_qv, coords_qv, current_num_clicks, current_click_idx, training, objects_info, error_cluster_dict, max_clicks_per_obj, initial_strategy, refinement_strategy):
    labels_qv = labels_qv.float()
    pred_label = pred_qv.float()

    # Do not generate new clicks for obj that has been clicked more than the threshold
    if current_click_idx is not None:
        for obj_id, click_ids in current_click_idx.items():
            # if obj_id != "0":  # background can receive as many clicks as needed
            if len(click_ids) >= max_clicks_per_obj:
                # Artificially set the pred_label to labels_qv for this object (as it received the threshold number of clicks)
                pred_label[labels_qv == int(obj_id)] = int(obj_id)

    error_mask = torch.abs(pred_label - labels_qv) > 0

    if error_mask.sum() == 0:
        return None, None, None, None, None

    cluster_ids = labels_qv * 9973 + pred_label * 11

    num_obj = (torch.unique(labels_qv) != 0).sum()

    error_clusters = cluster_ids[error_mask]
    error_cluster_ids = torch.unique(error_clusters)
    num_error_cluster = len(error_cluster_ids)

    error_cluster_ids_mask = torch.ones(coords_qv.shape[0], device=error_mask.device) * -1
    error_cluster_ids_mask[error_mask] = error_clusters

    ### measure the size of each error cluster and store the distance
    error_sizes, center_ids, center_coos, center_gts = {}, {}, {}, {}

    for cluster_id in error_cluster_ids:
        error = error_cluster_ids_mask == cluster_id
        if int(cluster_id) not in error_cluster_dict.keys():
            error_cluster_dict[int(cluster_id)] = 1
        else:
            error_cluster_dict[int(cluster_id)] += 1

        furthest_point, furthest_point_index = get_next_simulated_click_SI(error_cluster_dict[int(cluster_id)], coords_qv, error, initial_strategy, refinement_strategy)

        original_indices = torch.nonzero(error).squeeze()
        if original_indices.dim() == 0:  # There is only one point in the error region
            furthest_point_original_index = original_indices.item()
        else:
            furthest_point_original_index = original_indices[furthest_point_index]
        center_ids[int(cluster_id)], center_coos[int(cluster_id)], center_gts[int(cluster_id)] = furthest_point_original_index, furthest_point, labels_qv[furthest_point_original_index]

        correct_label, _ = decode_cluster_ids(cluster_id)
        error_region_percentage = torch.sum(error) / (labels_qv == correct_label).count_nonzero()

        if objects_info[int(correct_label)] == 0:
            error_sizes[int(cluster_id)] = np.inf
        else:
            error_sizes[int(cluster_id)] = error_region_percentage / (objects_info[int(correct_label)])

    error_cluster_ids_sorted = sorted(error_sizes, key=error_sizes.get, reverse=True)

    if training:
        if num_error_cluster >= num_obj:
            selected_error_cluster_ids = error_cluster_ids_sorted[:num_obj]
        else:
            selected_error_cluster_ids = error_cluster_ids_sorted
    else:
        if current_num_clicks == 0:
            selected_error_cluster_ids = error_cluster_ids_sorted
        else:
            selected_error_cluster_ids = error_cluster_ids_sorted[:1]

    # new_clicks, new_click_num, new_click_pos, new_click_time = get_next_simulated_click_multi(selected_error_cluster_ids, error_cluster_ids_mask, pred_qv, labels_qv, coords_qv, error_distances)
    new_clicks, new_click_num, new_click_pos, new_click_time = get_next_simulated_clicks(selected_error_cluster_ids, error_cluster_ids_mask, center_ids, center_coos, center_gts)
    return new_clicks, new_click_num, new_click_pos, new_click_time, error_cluster_dict


def get_next_simulated_click_SI(refinement_counter, coords_qv, error, initial_strategy, refinement_strategy):
    selected_strategy = initial_strategy if refinement_counter < 2 else refinement_strategy
    if selected_strategy == "random":
        closest_point, closest_point_index = get_next_click_random(coords_qv, error)
    elif selected_strategy == "boundary_dependent":
        closest_point, closest_point_index = get_next_click_bd_single_error(coords_qv, error)
    elif selected_strategy == "centroid":
        closest_point, closest_point_index = get_next_click_centroid(coords_qv, error)
    elif selected_strategy == "dbscan":
        closest_point, closest_point_index = get_next_click_dbscan(coords_qv, error)
    else:
        raise NotImplementedError
    return closest_point, closest_point_index


def get_next_click_bd_single_error(coords, error):
    """Select the point furthest from the border"""
    # Extract the coordinates of the wrongly classified points
    wrong_points = coords[error]
    # Compute the Euclidean distance of each point from the border
    distances = torch.cdist(wrong_points, wrong_points)
    # Find the point furthest from the border
    max_distances = torch.max(distances, dim=0)[0]
    furthest_point_index = torch.argmax(max_distances)
    furthest_point = wrong_points[furthest_point_index]
    return furthest_point, furthest_point_index


def get_next_click_bd_multi_errors(discrete_coords, unique_labels, gt, pred, pairwise_distances):
    """Sample the next click from the center of the error region"""
    zero_indices = unique_labels == 0
    one_indices = unique_labels == 1
    if zero_indices.sum() == 0 or one_indices.sum() == 0:
        return None, None, None, -1, None, None

    # point furthest from border
    center_id = torch.where(pairwise_distances == torch.max(pairwise_distances, dim=0)[0])
    center_coo = discrete_coords[one_indices, :][center_id[0][0]]
    center_label = gt[one_indices][center_id[0][0]]
    center_pred = pred[one_indices][center_id[0][0]]

    local_mask = torch.zeros(pairwise_distances.shape[0], device=discrete_coords.device)
    global_id_mask = torch.zeros(discrete_coords.shape[0], device=discrete_coords.device)
    local_mask[center_id] = 1
    global_id_mask[one_indices] = local_mask
    center_global_id = torch.argwhere(global_id_mask)[0][0]

    candidates = discrete_coords[one_indices, :]

    max_dist = torch.max(pairwise_distances)

    return center_global_id, center_coo, center_label, max_dist, candidates


def get_next_click_dbscan(coords, error, eps=1, min_samples=2):
    # Extract the coordinates of the wrongly classified points
    wrong_points = coords[error]
    wrong_points_np = wrong_points.cpu().numpy()
    # Apply DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(wrong_points_np)

    # Find the largest cluster
    unique_labels, counts = torch.unique(torch.tensor(labels), return_counts=True)
    if torch.equal(unique_labels, torch.tensor([-1])):
        # DBSCAN did not find any clusters, all points are considered noise, In such case where choose randomly
        largest_cluster_points = wrong_points
        selected_point_index = random.randint(0, wrong_points.shape[0] - 1)
        selected_point = wrong_points[selected_point_index]
        centroid_biggest_cluster = get_next_click_random
        return selected_point, selected_point_index

    largest_cluster_label = unique_labels[torch.argmax(counts[unique_labels != -1])]  # Ignore noise (-1)
    # Get the points belonging to the largest cluster
    largest_cluster_points = wrong_points[torch.tensor(labels) == largest_cluster_label]
    # Compute the centroid of the largest cluster points
    centroid_biggest_cluster = torch.mean(largest_cluster_points, dim=0)
    # Compute the Euclidean distance of each point in the largest cluster from the centroid
    distances = torch.norm(wrong_points - centroid_biggest_cluster, dim=1)

    # Find the point closest to the centroid
    closest_point_index = torch.argmin(distances)
    closest_point = wrong_points_np[closest_point_index]

    return closest_point, closest_point_index


def get_next_click_random(coords, error):
    """Select random point from the error region"""
    # Extract the coordinates of the wrongly classified point
    wrong_points = coords[error]
    selected_point_index = random.randint(0, wrong_points.shape[0] - 1)
    selected_point = wrong_points[selected_point_index]
    return selected_point, selected_point_index


def get_next_click_centroid(coords, error):
    # Extract the coordinates of the wrongly classified points
    wrong_points = coords[error]
    # Compute the centroid of the wrongly classified points
    centroid = torch.mean(wrong_points, dim=0)
    # Compute the Euclidean distance of each point from the centroid
    distances = torch.norm(wrong_points - centroid, dim=1)
    # Find the point closest to the centroid
    closest_point_index = torch.argmin(distances)
    closest_point = wrong_points[closest_point_index]

    return closest_point, closest_point_index


def get_next_simulated_clicks(error_cluster_ids, error_cluster_ids_mask, center_ids, center_coos, center_gts):
    """Sample the next clicks for each error region"""

    click_dict = {}
    new_click_pos = {}
    click_time_dict = {}
    click_order = 0

    random.shuffle(error_cluster_ids)

    for cluster_id in error_cluster_ids:
        center_id, center_coo, center_gt = center_ids[int(cluster_id)], center_coos[int(cluster_id)], center_gts[int(cluster_id)]
        if click_dict.get(str(int(center_gt))) == None:
            click_dict[str(int(center_gt))] = [int(center_id)]
            new_click_pos[str(int(center_gt))] = [center_coo]
            click_time_dict[str(int(center_gt))] = [click_order]
        else:
            click_dict[str(int(center_gt))].append(int(center_id))
            new_click_pos[str(int(center_gt))].append(center_coo)
            click_time_dict[str(int(center_gt))].append(click_order)

        click_order += 1

    click_num = len(error_cluster_ids)

    return click_dict, click_num, new_click_pos, click_time_dict


def decode_cluster_ids(cluster_ids, prime1=9973, prime2=11):
    # Recover pred_label using modular arithmetic
    # pred_label = (cluster_ids % prime1) // prime2
    pred_label = torch.div(cluster_ids % prime1, prime2, rounding_mode="trunc")

    # Recover labels_qv by isolating it
    # labels_qv = (cluster_ids - pred_label * prime2) // prime1
    labels_qv = torch.div(cluster_ids - pred_label * prime2, prime1, rounding_mode="trunc")

    return labels_qv, pred_label


def extend_clicks(current_clicks, current_clicks_time, new_clicks, new_click_time):
    """Append new click to existing clicks"""

    current_click_num = sum([len(c) for c in current_clicks_time.values()])

    for obj_id, click_ids in new_clicks.items():
        current_clicks[obj_id].extend(click_ids)
        current_clicks_time[obj_id].extend([t + current_click_num for t in new_click_time[obj_id]])

    return current_clicks, current_clicks_time
