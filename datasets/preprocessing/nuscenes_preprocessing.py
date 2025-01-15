import os
import json
import numpy as np
from fire import Fire
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion


class NuScenesPreprocessing:
    def __init__(
        self,
        data_dir: str = "/datasets/nuscenes",
        save_dir: str = "datasets/jsons/",
        modes: tuple = ["validation"],
    ):
        self.data_dir = Path(data_dir)
        self.save_dir = Path(save_dir)
        self.modes = modes
        self.databases = {}

        if not self.data_dir.exists():
            logger.error("Data folder doesn't exist")
            raise FileNotFoundError
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True, exist_ok=True)

        self.nusc_trainval = NuScenes(version="v1.0-trainval", dataroot=data_dir)
        scene_splits = create_splits_scenes()

        self.files = {}

        for mode in self.modes:
            self.files[mode] = []

        self.nusc = self.nusc_trainval
        for scene in self.nusc.scene:
            if scene["name"] in scene_splits["val"]:
                mode = "validation"
            else:
                continue
            next_sample = scene["first_sample_token"]
            scan = 0
            sample = self.nusc.get("sample", next_sample)
            token = sample["data"]["LIDAR_TOP"]
            while token != "":
                sample_data = self.nusc.get("sample_data", token)
                sensors = self.get_sensors(sample)
                label_filepath = self.nusc.get("panoptic", sample["data"]["LIDAR_TOP"])["filename"]
                sensors.update({"sequence": scene["name"], "scan": scan, "label_filepath": label_filepath, "is_key_frame": sample_data["is_key_frame"], "sample_token": sample_data["sample_token"]})

                self.files[mode].append(sensors)

                next_sample = sample["next"]
                scan = scan + 1
                token = sample_data["next"]

    def preprocess(self):
        for mode in self.modes:
            if mode == "test":
                self.nusc = self.nusc_test
            else:
                self.nusc = self.nusc_trainval
            Path(self.save_dir / mode).mkdir(parents=True, exist_ok=True)
            database = []
            for sensors in tqdm(self.files[mode], unit="file"):
                filebase = self.process_file(sensors, mode)
                database.append(filebase)
            self.databases[mode] = database

        logger.info("Preprocessing finished, saving databases as JSON")
        self.save_databases_as_json()

    def process_file(self, sensors, mode):
        sequence, scan, sample_token = sensors["sequence"], sensors["scan"], sensors["sample_token"]
        filepath = self.data_dir / sensors["lidar"]["filename"]
        lidar_pose = self.lidar_pose(sensors["lidar"])
        image_paths = []
        cam_calibs = []
        for cam_id in range(6):
            image_paths.append(str(Path(self.data_dir) / sensors[f"cam_{cam_id}"]["filename"]))
            l2c, c_intrinsic = self.lidar_camera_calibration(sensors["lidar"], sensors[f"cam_{cam_id}"])
            cam_calibs.append(
                {
                    "distorted_img_K": c_intrinsic,
                    "D": [0, 0, 0, 0, 0],
                    "upper2cam": l2c,
                }
            )
        filebase = {
            "sample_token": sample_token,
            "filepath": str(filepath),
            "scene": sequence,
            "scan": scan,
            "pose": lidar_pose,
            # "cameras": cam_calibs,
            # "image_paths": image_paths,
        }
        if mode in ["train", "validation"] and sensors["is_key_frame"]:
            filebase["label_filepath"] = str(self.data_dir / sensors["label_filepath"])
            filebase["is_key_frame"] = sensors["is_key_frame"]
        return filebase

    def lidar_camera_calibration(self, lidar_sensor, camera_sensor):
        lidar_calibration = self.nusc.get("calibrated_sensor", lidar_sensor["calibrated_sensor_token"])
        lidar2ego = self.calibration_to_transformation_matrix(lidar_calibration)

        lidar_ego_pose_calibration = self.nusc.get("ego_pose", lidar_sensor["ego_pose_token"])
        lidar_ego_pose = self.calibration_to_transformation_matrix(lidar_ego_pose_calibration)

        cam_ego_pose_calibration = self.nusc.get("ego_pose", camera_sensor["ego_pose_token"])
        cam_ego_pose_inv = self.calibration_to_transformation_matrix(cam_ego_pose_calibration, inverse=True)

        camera_calibration = self.nusc.get("calibrated_sensor", camera_sensor["calibrated_sensor_token"])
        camera2ego_inv = self.calibration_to_transformation_matrix(camera_calibration, inverse=True)

        lidar2camera = camera2ego_inv @ cam_ego_pose_inv @ lidar_ego_pose @ lidar2ego
        camera_intrinsic = np.array(camera_calibration["camera_intrinsic"])

        return lidar2camera.tolist(), camera_intrinsic.tolist()

    def lidar_pose(self, lidar_sensor):
        lidar_calibration = self.nusc.get("calibrated_sensor", lidar_sensor["calibrated_sensor_token"])
        lidar2ego = self.calibration_to_transformation_matrix(lidar_calibration)

        lidar_ego_pose_calibration = self.nusc.get("ego_pose", lidar_sensor["ego_pose_token"])
        lidar_ego_pose = self.calibration_to_transformation_matrix(lidar_ego_pose_calibration)

        lidar_pose = np.linalg.inv(lidar2ego) @ lidar_ego_pose @ lidar2ego

        return lidar_pose.tolist()

    def calibration_to_transformation_matrix(self, calibration, inverse=False):
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = Quaternion(calibration["rotation"]).rotation_matrix
        transformation_matrix[:3, 3] = calibration["translation"]
        if inverse:
            transformation_matrix = np.linalg.inv(transformation_matrix)
        return transformation_matrix

    def get_sensors(self, sample):
        lidar_sensor = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        cam_front_sensor = self.nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        cam_front_right_sensor = self.nusc.get("sample_data", sample["data"]["CAM_FRONT_RIGHT"])
        cam_back_right_sensor = self.nusc.get("sample_data", sample["data"]["CAM_BACK_RIGHT"])
        cam_back_sensor = self.nusc.get("sample_data", sample["data"]["CAM_BACK"])
        cam_back_left_sensor = self.nusc.get("sample_data", sample["data"]["CAM_BACK_LEFT"])
        cam_front_left_sensor = self.nusc.get("sample_data", sample["data"]["CAM_FRONT_LEFT"])

        return {
            "lidar": lidar_sensor,
            "cam_0": cam_front_sensor,
            "cam_1": cam_front_right_sensor,
            "cam_2": cam_back_right_sensor,
            "cam_3": cam_back_sensor,
            "cam_4": cam_back_left_sensor,
            "cam_5": cam_front_left_sensor,
        }

    def save_databases_as_json(self):
        val_json_file_name = "nuscenes_validation_list.json"

        # Save data as JSON for each mode
        for mode in self.modes:
            data = self.databases[mode]
            json_data = {}
            for item in data:
                # Construct the key based on the filepath
                scene_id = item["scene"]
                sample_token = item["sample_token"]
                key = f"{scene_id}_{sample_token}"
                json_data[key] = item

            with open(os.path.join(self.save_dir, val_json_file_name), "w") as file:
                json_data = json.dumps(json_data, indent=2)
                file.write(json_data)
                logger.info(f"Databases saved at {os.path.join(self.save_dir, val_json_file_name)}")


if __name__ == "__main__":
    Fire(NuScenesPreprocessing)
