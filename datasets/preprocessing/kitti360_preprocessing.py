import json
import os
import numpy as np
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from fire import Fire


class KITTI360Preprocessing:
    def __init__(
        self,
        data_dir: str = "/datasets/KITTI-360/",
        label_dir: str = "/datasets/KITTI360SingleScan/",
        save_dir: str = "datasets/jsons/",
        modes: tuple = ["validation"],
    ):
        self.data_dir = Path(data_dir)
        self.label_dir = Path(label_dir)
        self.save_dir = Path(save_dir)
        self.validation_split_file = os.path.join(self.data_dir, "data_3d_semantics/train/", "2013_05_28_drive_val.txt")
        self.modes = modes
        self.databases = {}

        if not self.data_dir.exists():
            logger.error("Data folder doesn't exist")
            raise FileNotFoundError("Data folder doesn't exist")
        if self.save_dir.exists() is False:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # Pose adjustments
        poses_path = os.path.join(self.data_dir, "data_poses")
        extrinsicName = os.path.join(self.data_dir, "calibration", "calib_cam_to_velo.txt")
        cam2poseName = os.path.join(self.data_dir, "calibration", "calib_cam_to_pose.txt")
        Tr_cam_velo, Tr_cam_pose = loadTransform(extrinsicName), loadCamPose(cam2poseName)
        self.Tr_velo_pose = Tr_cam_pose[0] @ np.linalg.inv(Tr_cam_velo)
        self.poses = {}
        self.timestamp = {}
        for scene in os.listdir(poses_path):
            self.poses[scene] = {}
            poses_file = os.path.join(poses_path, scene, "poses.txt")
            with open(poses_file, "r") as file:
                for line in file:
                    if line:
                        lineProcessed = line.rstrip("\n").split(" ")
                        transform = np.eye(4)
                        for i in range(12):
                            xi = i // 4
                            yi = i % 4
                            transform[xi, yi] = float(lineProcessed[i + 1])

                        velo_to_world = transform @ self.Tr_velo_pose
                        self.poses[scene][int(line.split()[0])] = velo_to_world.tolist()

            self.timestamp[scene] = {}
            timestamp_file = os.path.join(self.data_dir, f"data_3d_raw/{scene}/velodyne_points/timestamps.txt")
            scan_counter = 0
            if not os.path.exists(timestamp_file):
                print(f"Timestamp file does not exist: {timestamp_file}")
                continue

            with open(timestamp_file, "r") as file:
                for line in file:
                    line = line.strip()
                    if line:
                        self.timestamp[scene][scan_counter] = line
                        scan_counter += 1
                    else:
                        raise ValueError("Empty line in timestamp file")

        with open(self.validation_split_file, "r") as file:
            self.validation_chunks = file.readlines()
            self.validation_chunks = [scene.strip() for scene in self.validation_chunks]

        self.files = {}
        for data_type in self.modes:
            self.files.update({data_type: []})

        mode, counter = "validation", 0
        for chunk in sorted(self.validation_chunks):
            current_scene = chunk.split("/")[2]
            velodyne_dir = os.path.join(self.data_dir, f"data_3d_raw/{current_scene}/velodyne_points/data/")
            single_label_dir = os.path.join(label_dir, f"{current_scene}/labels/")
            chunk_ranges = chunk.split("/")[4].split(".")[0]
            start_str, end_str = chunk_ranges.split("_")
            start_num = int(start_str)
            end_num = int(end_str)
            for num in range(start_num, end_num + 1):
                scan = f"{num:010d}"  # Format as a 10-digit number with leading zeros
                single_label_path = os.path.join(single_label_dir, f"{scan}.bin")
                if os.path.exists(single_label_path):
                    raw_scan_path = os.path.join(velodyne_dir, f"{scan}.bin")
                    assert os.path.exists(raw_scan_path), f"File does not exist: {raw_scan_path}"
                    if raw_scan_path not in self.files[mode]:
                        self.files[mode].append(raw_scan_path)
                    else:
                        print(f"File already exists in database: {raw_scan_path}")
                else:
                    counter += 1
                    print(f"File does not exist: {single_label_path}")

        print(f"Number of missing files: {counter}")

    def preprocess(self):
        logger.info(f"starting preprocessing...")
        for mode in self.modes:
            logger.info(f"Initializing {mode} database...")
            self.databases[mode] = []
            database = []
            for filepath in tqdm(self.files[mode], unit="file"):
                if "extra_tile" in filepath:
                    # don't process extra tiles
                    continue
                filebase = self.process_file(filepath, mode)
                if filebase is None:
                    continue
                database.append(filebase)
            self.databases[mode] = database
        logger.info(f"Finished initializing")

        self.save_databases_as_json()

    def process_file(self, filepath, mode):
        scene, scan = filepath.split("/")[-4], (filepath.split("/")[-1]).split(".")[0]
        sample = {
            "filepath": filepath,
            "scene": scene,
            "scan": scan,
        }

        if mode in ["validation"]:
            # getting label info
            scene = filepath.split("/")[-4]
            scan = (filepath.split("/")[-1]).split(".")[0]
            label_filepath = os.path.join(self.label_dir, scene, "labels", f"{scan}.bin")
            assert os.path.exists(label_filepath), f"Label file does not exist: {label_filepath}"

            sample["label_filepath"] = label_filepath
            sample["pose"] = self.poses[scene][int(scan)]
            sample["timestamp"] = self.timestamp[scene][int(scan)]

        else:
            raise ValueError(f"Mode {mode} is not supported for kitti360")

        return sample

    def save_databases_as_json(self):
        # Save data as JSON for each mode
        val_json_file_name = os.path.join(self.save_dir, "kitti360_validation_list.json")

        for mode in self.modes:
            data = self.databases[mode]
            json_data = {}
            for item in data:
                scene_id = item["scene"]
                scan = item["scan"]
                key = f"scene_{scene_id}_{scan}"
                json_data[key] = item

            with open(val_json_file_name, "w") as file:
                json_data = json.dumps(json_data, indent=2)
                file.write(json_data)

            logger.info(f"Databases saved at: {val_json_file_name}")


###############################################################################
############################# Utility Functions ###############################
###############################################################################


def loadTransform(filename):
    transform = np.eye(4)
    try:
        infile = open(filename).readline().rstrip("\n").split(" ")
    except:
        print("Failed to open transforms " + filename)
        return transform, False

    for i in range(12):
        xi = i // 4
        yi = i % 4
        transform[xi, yi] = float(infile[i])

    return transform


def loadCamPose(filename):
    poses = [None for _ in range(4)]
    try:
        infile = open(filename)
    except:
        print("Failed to open camera poses " + filename)
        return poses, False

    for line in infile:
        lineProcessed = line.rstrip("\n").split(" ")
        if any("image_0" in x for x in lineProcessed):
            transform = np.eye(4)
            index = int(lineProcessed[0][7])
            for i in range(12):
                xi = i // 4
                yi = i % 4
                transform[xi, yi] = float(lineProcessed[i + 1])
            poses[index] = transform

    infile.close()
    return poses


if __name__ == "__main__":
    Fire(KITTI360Preprocessing)
