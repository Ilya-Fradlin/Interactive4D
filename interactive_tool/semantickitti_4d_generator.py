import numpy as np
import yaml
import os
import json
from pathlib import Path
from fire import Fire

from utils import write_ply


class GenerateScene:
    def __init__(
        self,
        scans_data_file: str = "/datasets/processed_jsons/semantickitti_validation_list.json",
        save_dir: str = "/datasets/interactive_scenes/",
        save_with_gt: bool = True,
        sweep: int = 4,
        start_scan: int = 100,
        radius: float = 30,
    ):
        self.save_dir = save_dir
        self.save_with_gt = save_with_gt
        self.radius = radius

        obj2label_map = {}

        with open(scans_data_file, "r") as file:
            scans_data = json.load(file)
            scans_to_superimpose = list(range(start_scan, start_scan + sweep, 1))

        accumulated_pcd = []
        accumulated_labels = []
        accumulated_features = []

        # Iterate over the scans to superimpose
        for scan_idx, scan in enumerate(scans_to_superimpose):
            # Fetch the scan data
            scan_name = str(scan).zfill(6)
            val_key = f"scene_08_{scan_name}"
            current_scan_data = scans_data[val_key]

            # Load the pcd, labels and features
            coordinates, features, labels = GenerateScene.get_scan_data(current_scan_data, scan_idx)

            # Accumulate the data for the sweep
            accumulated_labels.append(labels)
            accumulated_pcd.append(coordinates)
            accumulated_features.append(features)

        # Concatenate point clouds and labels
        self.combined_point_cloud = np.vstack(accumulated_pcd)
        self.combined_point_cloud -= np.mean(self.combined_point_cloud, axis=0)

        self.combined_panoptic_labels = np.hstack(accumulated_labels)

        combined_features = np.vstack(accumulated_features)
        self.combined_features = np.hstack((combined_features, np.linalg.norm(self.combined_point_cloud - self.combined_point_cloud.mean(0), axis=1)[:, np.newaxis]))  # now features include: (time, intensity, distance)

        # Crop the scene to the desired radius
        self.crop_desired_radius()

        # Update the challenge labels
        self.update_challenge_labels()

        # as in semantickitti there is no rgb information, give all points the same color (black)
        # it is also possible to use the color based on the semantic/instance label
        self.colors = np.zeros((self.combined_point_cloud.shape[0], 3), dtype=np.uint8)

        obj_id_labels, obj2label_map = self.convert_panoptic_to_obj_id()

        scene_idx = 1
        starting_scan = str(scans_to_superimpose[0])
        ending_scan = str(scans_to_superimpose[-1])

        # get the appropriate scene index
        # make sure f"scene_{scene_idx}" is with at least 2 digits
        while any(name.startswith(f"scene_{scene_idx:02d}") for name in os.listdir(self.save_dir)):
            scene_idx += 1

        dir_path = os.path.join(self.save_dir, f"scene_{scene_idx:02d}_{starting_scan}_{ending_scan}")
        os.makedirs(dir_path, exist_ok=True)

        # save obj2label_map
        with open(os.path.join(dir_path, "obj2label.yaml"), "w") as file:
            yaml.dump(obj2label_map, file)

        ply_path = os.path.join(dir_path, f"scan.ply")
        if self.save_with_gt == True:
            field_names = ["x", "y", "z", "red", "green", "blue", "time", "intensity", "distance", "label"]
            write_ply(ply_path, [self.combined_point_cloud, self.colors, self.combined_features, obj_id_labels], field_names)
        else:
            field_names = ["x", "y", "z", "red", "green", "blue", "time", "intensity", "distance"]
            write_ply(ply_path, [self.combined_point_cloud, self.colors, self.combined_features], field_names)

        print(f"Scene saved successfully at {ply_path}")

    def crop_desired_radius(self):
        mask = np.linalg.norm(self.combined_point_cloud, axis=1) < self.radius
        self.combined_point_cloud = self.combined_point_cloud[mask]
        self.combined_panoptic_labels = self.combined_panoptic_labels[mask]
        self.combined_features = self.combined_features[mask]

    def update_challenge_labels(self):
        # Load the learning map
        project_root = Path(__file__).resolve().parent.parent
        yaml_path = project_root / "conf" / "semantic-kitti.yaml"
        learning_map = yaml.safe_load(open(yaml_path))["learning_map"]

        # Extract the semantic labels
        semantic_labels = self.combined_panoptic_labels & 0xFFFF
        updated_semantic_labels = np.vectorize(learning_map.__getitem__)(semantic_labels)

        # Combine the new category_id and instance_id
        self.combined_panoptic_labels &= np.array(~0xFFFF).astype(np.uint32)  # Clear lower 16 bits
        self.combined_panoptic_labels |= updated_semantic_labels.astype(np.uint32)  # Set lower 16 bits with updated semantic labels

    def convert_panoptic_to_obj_id(self):
        obj_id_labels = np.zeros(self.combined_panoptic_labels.shape)
        unique_panoptic_labels = np.unique(self.combined_panoptic_labels)

        # create a map from object index to label
        obj2label_map = {}
        for obj_idx, label in enumerate(unique_panoptic_labels):
            obj_idx += 1
            obj_id_labels[self.combined_panoptic_labels == label] = int(obj_idx)
            obj2label_map[str(int(obj_idx))] = int(label)

        return obj_id_labels, obj2label_map

    @staticmethod
    def get_scan_data(scan_data, scan_idx):
        scan_pcd_path = scan_data["filepath"]
        scan_label_path = scan_data["label_filepath"]

        # Load the point cloud
        point_cloud, features = GenerateScene.load_point_cloud(scan_pcd_path)
        pose = np.array(scan_data["pose"]).T
        coordinates = point_cloud[:, :3]
        coordinates = coordinates @ pose[:3, :3] + pose[3, :3]

        # Load the labels
        labels = GenerateScene.load_labels(scan_label_path)

        # Generate the features
        time_array = np.ones((features.shape[0], 1)) * scan_idx
        features = np.hstack((time_array, features))  # (time, intensity)

        return coordinates, features, labels

    @staticmethod
    def load_point_cloud(scan_path):
        scan = np.fromfile(scan_path, dtype=np.float32)
        scan = scan.reshape((-1, 4))  # The point cloud data is stored in a Nx4 format (x, y, z, intensity)
        points = scan[:, :3]  # Extracting the (x, y, z) coordinates
        features = scan[:, 3].reshape(-1, 1)  # Extracting the intensity values
        return points, features

    @staticmethod
    def load_labels(label_path):
        labels = np.fromfile(label_path, dtype=np.uint32)  # Labels are stored as unsigned 32-bit integers
        return labels


if __name__ == "__main__":
    Fire(GenerateScene)
