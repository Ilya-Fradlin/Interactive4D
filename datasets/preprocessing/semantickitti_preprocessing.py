import re
import json
import os
import numpy as np
import yaml
from pathlib import Path
from natsort import natsorted
from loguru import logger
from tqdm import tqdm
from fire import Fire


class SemanticKittiPreprocessing:
    def __init__(
        self,
        data_dir: str = "/datasets/SemanticKITTI/",
        save_dir: str = "datasets/jsons/",
        modes: tuple = ("train", "validation"),
    ):
        self.data_dir = Path(data_dir)
        self.save_dir = Path(save_dir)
        self.modes = modes
        self.databases = {}

        if not self.data_dir.exists():
            logger.error("Data folder doesn't exist")
            raise FileNotFoundError
        if self.save_dir.exists() is False:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        self.files = {}
        for data_type in self.modes:
            self.files.update({data_type: []})

        config_path = self.data_dir / "semantic-kitti.yaml"
        self.config = self._load_yaml(config_path)
        self.pose = dict()

        for mode in self.modes:
            scene_mode = "valid" if mode == "validation" else mode
            self.pose[mode] = dict()
            for scene in sorted(self.config["split"][scene_mode]):
                filepaths = list(self.data_dir.glob(f"*/{scene:02}/velodyne/*bin"))
                filepaths = [str(file) for file in filepaths]
                self.files[mode].extend(natsorted(filepaths))
                calibration = parse_calibration(Path(filepaths[0]).parent.parent / "calib.txt")
                self.pose[mode].update({scene: parse_poses(Path(filepaths[0]).parent.parent / "poses.txt", calibration)})

    def preprocess(self):
        logger.info(f"starting preprocessing...")
        for mode in self.modes:
            logger.info(f"Initializing {mode} database...")
            self.databases[mode] = []
            database = []
            for filepath in tqdm(self.files[mode], unit="file"):
                filebase = self.process_file(filepath, mode)
                if filebase is None:
                    continue
                database.append(filebase)
            self.databases[mode] = database
        logger.info(f"Finished preprocessing")

        self.save_databases_as_json()

    def save_databases_as_json(self):
        """
        Save the databases as JSON files.

        This method saves the data in the databases attribute as JSON files for each mode.
        The JSON files are named based on the mode and whether the data is subsampled or not.

        Returns:
            None
        """
        train_json_file_name = "semantickitti_train_list.json"
        val_json_file_name = "semantickitti_validation_list.json"

        # Save data as JSON for each mode
        for mode in self.modes:
            data = self.databases[mode]
            json_data = {}
            for item in data:
                # Construct the key based on the filepath
                filepath = item["filepath"]
                scene_id = os.path.basename(os.path.dirname(os.path.dirname(filepath)))
                filename = os.path.splitext(os.path.basename(filepath))[0]
                key = f"scene_{scene_id}_{filename}"
                json_data[key] = item

            # Determine file name based on mode
            json_file_name = train_json_file_name if "train" in mode else val_json_file_name
            with open(os.path.join(self.save_dir, json_file_name), "w") as file:
                json_data = json.dumps(json_data, indent=2)
                file.write(json_data)

    @classmethod
    def _save_yaml(cls, path, file):
        with open(path, "w") as f:
            yaml.safe_dump(file, f, default_style=None, default_flow_style=False)

    @classmethod
    def _dict_to_yaml(cls, dictionary):
        if not isinstance(dictionary, dict):
            return
        for k, v in dictionary.items():
            if isinstance(v, dict):
                cls._dict_to_yaml(v)
            if isinstance(v, np.ndarray):
                dictionary[k] = v.tolist()
            if isinstance(v, Path):
                dictionary[k] = str(v)

    @classmethod
    def _load_yaml(cls, filepath):
        with open(filepath) as f:
            file = yaml.safe_load(f)
        return file

    def process_file(self, filepath, mode):
        scene, sub_scene = re.search(r"(\d{2}).*(\d{6})", filepath).group(1, 2)
        sample = {
            "filepath": filepath,
            "scene": int(scene),
            "pose": self.pose[mode][int(scene)][int(sub_scene)].tolist(),
        }

        if mode in ["train", "validation"]:
            # getting label info
            label_filepath = filepath.replace("velodyne", "labels").replace("bin", "label")
            sample["label_filepath"] = label_filepath

        return sample


###############################################################################
############################# Utility Functions ###############################
###############################################################################
def parse_calibration(filename):
    calib = {}

    with open(filename) as calib_file:
        for line in calib_file:
            key, content = line.strip().split(":")
            values = [float(v) for v in content.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            calib[key] = pose
    return calib


def parse_poses(filename, calibration):
    poses = []

    Tr = calibration["Tr"]
    Tr_inv = np.linalg.inv(Tr)

    with open(filename) as file:
        for line in file:
            values = [float(v) for v in line.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

    return poses


if __name__ == "__main__":
    Fire(SemanticKittiPreprocessing)
