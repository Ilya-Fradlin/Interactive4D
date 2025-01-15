import numpy as np
import volumentations as V
import yaml
import json
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional
from random import random, shuffle


class SemanticKittiDataset(Dataset):
    def __init__(
        self,
        data_dir: Optional[str] = "datasets/jsons/",
        mode: Optional[str] = "train",
        volume_augmentations_path: Optional[str] = None,
        sweep: Optional[int] = 1,
        center_coordinates=True,
        window_overlap=0,
    ):
        super(SemanticKittiDataset, self).__init__()

        self.mode = mode
        self.data_dir = data_dir
        self.sweep = sweep
        self.center_coordinates = center_coordinates
        self.config = self._load_yaml("conf/semantic-kitti.yaml")

        if sweep > 1:
            self.drop_outliers = True
        else:
            self.drop_outliers = False

        # loading database file
        database_path = Path(self.data_dir)
        database_file = database_path.joinpath(f"semantickitti_{mode}_list.json")
        if not database_file.exists():
            raise FileNotFoundError(f"Database file not found at {database_file}")
        with open(database_file) as json_file:
            self.data = json.load(json_file)

        # reformulating in sweeps
        data = [[]]
        scene_names = list(self.data.keys())
        last_scene = self.data[scene_names[0]]["scene"]
        for scene_name in scene_names:
            x = self.data[scene_name]  # get the actual sample from the dictionary
            if x["scene"] == last_scene:
                data[-1].append(x)
            else:
                last_scene = x["scene"]
                data.append([x])
        for i in range(len(data)):
            data[i] = list(self.chunks(data[i], sweep, overlap=window_overlap))
        self.data = [val for sublist in data for val in sublist]

        # augmentations
        self.volume_augmentations = V.NoOp()
        if volume_augmentations_path is not None:
            self.volume_augmentations = V.load(volume_augmentations_path, data_format="yaml")

    def chunks(self, lst, n, overlap=0):
        if "train" in self.mode or n == 1:
            for i in range(len(lst) - n + 1):
                yield lst[i : i + n]
        else:
            if overlap == 0:
                # Non-overlapping chunks
                for i in range(0, len(lst) - n, n):
                    yield lst[i : i + n]
                # Ensure the last chunk is also of size n, taking the last n elements
                yield lst[-n:]

            # overlapping chunks
            else:
                step = n - overlap  # Determine the step size based on the overlap
                for i in range(0, len(lst) - n, step):
                    yield lst[i : i + n]
                # Ensure the last chunk is also of size n, taking the last n elements
                yield lst[-n:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        coordinates_list = []
        features_list = []
        labels_list = []
        acc_num_points = []
        num_obj_in_scene = []
        obj2label_maps_list = []
        file_paths = []

        # for debugging can specify idx = 1397 (for scene 1397)
        label2obj_map, obj2label_map, click_idx = {}, {}, {}
        for time, scan in enumerate(self.data[idx]):
            file_paths.append(scan["filepath"])
            points = np.fromfile(scan["filepath"], dtype=np.float32).reshape(-1, 4)
            coordinates = points[:, :3]
            # rotate and translate
            pose = np.array(scan["pose"]).T
            coordinates = coordinates @ pose[:3, :3] + pose[3, :3]

            # features
            features = points[:, 3:4]  # intensity
            time_array = np.ones((features.shape[0], 1)) * time
            features = np.hstack((time_array, features))  # (time, intensity)

            # labels
            if "test" in self.mode:
                labels = np.zeros_like(features).astype(np.int64)
                obj2label_maps_list.append({})
            else:
                # Convert the panoptic labels into object labels
                labels, obj2label_map, click_idx, label2obj_map = self.generate_object_labels(scan, label2obj_map, obj2label_map, click_idx)
                obj2label_maps_list.append(obj2label_map)
                unique_labels = np.unique(labels)
                unique_labels = unique_labels[unique_labels != 0]
                num_obj_in_scene.append(len(unique_labels))

            if self.drop_outliers:
                # Create a boolean mask where labels are not 0
                mask = labels != 0
                # Apply the mask to each array
                labels = labels[mask]
                coordinates = coordinates[mask]
                features = features[mask]

            acc_num_points.append(len(labels))
            labels_list.append(labels)
            coordinates_list.append(coordinates)
            features_list.append(features)

        coordinates = np.vstack(coordinates_list)
        if self.center_coordinates:
            coordinates -= coordinates.mean(0)
        features = np.vstack(features_list)
        labels = np.hstack(labels_list)

        # Enrich the features with the distance to the center
        center_coordinate = coordinates.mean(0)
        features = np.hstack((features, np.linalg.norm(coordinates - center_coordinate, axis=1)[:, np.newaxis]))

        # volume and image augmentations for train
        if "train" in self.mode:
            coordinates -= coordinates.mean(0)
            if 0.5 > random():
                coordinates += np.random.uniform(coordinates.min(0), coordinates.max(0)) / 2
            aug = self.volume_augmentations(points=coordinates)
            coordinates = aug["points"]

        # stack coordinates and features: ((coordinates),(time, intensity, distance))
        features = np.hstack((coordinates, features))

        return {
            "sequence": file_paths,
            "num_points": acc_num_points,
            "num_obj": num_obj_in_scene,
            "coordinates": coordinates,
            "features": features,
            "labels": labels,
            "click_idx": click_idx,
            "obj2label": obj2label_maps_list,
        }

    def generate_object_labels(self, scan, label2obj_map, obj2label_map, click_idx):
        # Extract semantic labels
        panoptic_labels = np.fromfile(scan["label_filepath"], dtype=np.uint32)
        semantic_labels = panoptic_labels & 0xFFFF
        updated_semantic_labels = np.vectorize(self.config["learning_map"].__getitem__)(semantic_labels)
        # Update semantic labels according to the learning map
        panoptic_labels &= np.array(~0xFFFF).astype(np.uint32)  # Clear lower 16 bits
        panoptic_labels |= updated_semantic_labels.astype(np.uint32)  # Set lower 16 bits with updated semantic labels

        unique_panoptic_labels = np.unique(panoptic_labels)
        unique_panoptic_labels = unique_panoptic_labels[unique_panoptic_labels != 0].tolist()  # Drop 0 (outlier class)
        shuffle(unique_panoptic_labels)

        obj_labels = np.zeros(panoptic_labels.shape)
        if label2obj_map == {}:
            # This is the first scene we are generating object labels
            for obj_idx, label in enumerate(unique_panoptic_labels):
                obj_idx += 1  # 0 is background
                obj_labels[panoptic_labels == label] = int(obj_idx)
                obj2label_map[str(int(obj_idx))] = int(label)
                label2obj_map[label] = int(obj_idx)
                click_idx[str(obj_idx)] = []
            # Background
            click_idx["0"] = []
        else:
            # We have already generated object labels for previous scene in the sweep, now need to update for new object
            current_obj_idx = max(label2obj_map.values()) + 1  # In case there are new objects in the scene, add them as the following index
            for label in unique_panoptic_labels:
                if label in label2obj_map.keys():
                    defined_obj_id = label2obj_map[label]
                    obj_labels[panoptic_labels == label] = int(defined_obj_id)
                else:
                    # a new obj is introduced
                    obj2label_map[str(int(current_obj_idx))] = int(label)
                    label2obj_map[label] = int(current_obj_idx)
                    obj_labels[panoptic_labels == label] = int(current_obj_idx)
                    click_idx[str(current_obj_idx)] = []
                    current_obj_idx += 1

        return obj_labels, obj2label_map, click_idx, label2obj_map

    def augment(self, point_cloud):
        if np.random.random() > 0.5:
            # Flipping along the YZ plane
            point_cloud[:, 0] = -1 * point_cloud[:, 0]

        if np.random.random() > 0.5:
            # Flipping along the XZ plane
            point_cloud[:, 1] = -1 * point_cloud[:, 1]

        # Rotation along up-axis/Z-axis
        rot_angle_pre = np.random.choice([0, np.pi / 2, np.pi, np.pi / 2 * 3])
        rot_mat_pre = self.rotz(rot_angle_pre)
        point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat_pre))

        rot_angle = (np.random.random() * 2 * np.pi) - np.pi  # -180 ~ +180 degree
        rot_mat = self.rotz(rot_angle)
        point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))

        return point_cloud

    def rotz(self, t):
        """Rotation about the z-axis."""
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    @staticmethod
    def _load_yaml(filepath):
        with open(filepath) as f:
            file = yaml.safe_load(f)
        return file
