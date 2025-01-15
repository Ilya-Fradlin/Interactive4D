import torch
import psutil
from torchmetrics import Metric
from pytorch_lightning.callbacks import Callback

import datasets.datasets_info as datasets_info


class IoU_at_numClicks(Metric):
    higher_is_better = True
    full_state_update = True

    def __init__(self, num_clicks=[1, 2, 3, 4, 5]):
        super().__init__()
        self.num_clicks = num_clicks

        for noc in num_clicks:
            self.add_state(f"iou_at_{noc}", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state(f"count_for_{noc}", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, iou, noc):
        if noc in self.num_clicks:
            noc = int(noc)
            self.__dict__[f"iou_at_{noc}"] += iou
            self.__dict__[f"count_for_{noc}"] += 1
        else:
            print("iou not found")
            raise ValueError

    def compute(self):
        metrics_dictionary = {}

        for noc in self.num_clicks:
            metrics_dictionary[noc] = {}
            metrics_dictionary[noc]["iou"] = self.__dict__[f"iou_at_{noc}"]
            metrics_dictionary[noc]["count"] = self.__dict__[f"count_for_{noc}"]

        return metrics_dictionary, self.num_clicks


class NumClicks_for_IoU(Metric):
    full_state_update = True

    def __init__(self, iou_thresholds=[0.50, 0.65, 0.80, 0.85, 0.90]):
        super().__init__()

        self.iou_thresholds = [int(100 * iou) for iou in iou_thresholds if iou <= 1]
        for iou in self.iou_thresholds:
            self.add_state(f"noc_for_{iou}", default=torch.tensor(0.0), dist_reduce_fx="mean")
            self.add_state(f"count_for_{iou}", default=torch.tensor(0.0), dist_reduce_fx="mean")

    def update(self, iou, noc):
        iou = int(100 * iou)
        if iou in self.iou_thresholds:
            self.__dict__[f"noc_for_{iou}"] += noc
            self.__dict__[f"count_for_{iou}"] += 1
        else:
            print("iou not found")
            raise ValueError

    def compute(self):
        metrics_dictionary = {}

        for iou in self.iou_thresholds:
            metrics_dictionary[iou] = {}
            metrics_dictionary[iou]["noc"] = self.__dict__[f"noc_for_{iou}"]
            metrics_dictionary[iou]["count"] = self.__dict__[f"count_for_{iou}"]

        return metrics_dictionary, self.iou_thresholds


class NumClicks_for_IoU_class(Metric):
    full_state_update = True

    def __init__(self, label_mapping, iou_thresholds=[0.50, 0.65, 0.80, 0.85, 0.90]):
        super().__init__()
        self.classes = label_mapping.values()
        self.iou_thresholds = [int(100 * iou) for iou in iou_thresholds if iou <= 1]
        for iou in self.iou_thresholds:
            for class_type in self.classes:
                self.add_state(f"{class_type}_noc_for_{iou}", default=torch.tensor(0.0), dist_reduce_fx="mean")
                self.add_state(f"{class_type}_count_for_{iou}", default=torch.tensor(0.0), dist_reduce_fx="mean")

    def update(self, iou, noc, class_type):
        iou = int(100 * iou)
        if iou in self.iou_thresholds:
            self.__dict__[f"{class_type}_noc_for_{iou}"] += noc
            self.__dict__[f"{class_type}_count_for_{iou}"] += 1
        else:
            print("iou not found")
            raise ValueError

    def compute(self):
        metrics_dictionary = {}

        for class_type in self.classes:
            metrics_dictionary[class_type] = {}
            for iou in self.iou_thresholds:
                metrics_dictionary[class_type][iou] = {}
                metrics_dictionary[class_type][iou]["noc"] = self.__dict__[f"{class_type}_noc_for_{iou}"]
                metrics_dictionary[class_type][iou]["count"] = self.__dict__[f"{class_type}_count_for_{iou}"]

        return metrics_dictionary, self.iou_thresholds


class mIoU_metric(Metric):
    higher_is_better = True
    full_state_update = True

    def __init__(self):
        super().__init__()
        self.add_state(f"mIoU", default=torch.tensor(0.0), dist_reduce_fx="mean")
        self.add_state(f"count", default=torch.tensor(0.0), dist_reduce_fx="mean")

    def update(self, mIoU):
        self.__dict__[f"mIoU"] += mIoU
        self.__dict__[f"count"] += 1

    def compute(self):
        return self.__dict__[f"mIoU"] / self.__dict__[f"count"]


class losses_metric(Metric):
    higher_is_better = False
    full_state_update = True

    def __init__(self):
        super().__init__()
        self.add_state(f"loss", default=torch.tensor(0.0), dist_reduce_fx="mean")
        self.add_state(f"count", default=torch.tensor(0.0), dist_reduce_fx="mean")

    def update(self, loss):
        self.__dict__[f"loss"] += loss
        self.__dict__[f"count"] += 1

    def compute(self):
        return self.__dict__[f"loss"] / self.__dict__[f"count"]


class mIoU_per_class_metric(Metric):
    higher_is_better = True
    full_state_update = True

    def __init__(self, training=True):
        super().__init__()
        self.training = training
        self.label_mapping = datasets_info.semantickitti_label_mapping
        for _, label in self.label_mapping.items():
            self.add_state(f"miou_for_{label}", default=torch.tensor(0.0), dist_reduce_fx="mean")
            self.add_state(f"count_for_{label}", default=torch.tensor(0.0), dist_reduce_fx="mean")

    def update(self, label_miou_dict):
        for label, mIoU in label_miou_dict.items():
            label = label.split("/")[-1]
            if label in self.label_mapping.values():
                self.__dict__[f"miou_for_{label}"] += mIoU
                self.__dict__[f"count_for_{label}"] += 1
            else:
                raise ValueError("unknown label found")

    def compute(self):
        metrics_dictionary = {}
        for _, label in self.label_mapping.items():
            if self.training:
                metrics_dictionary[f"training_miou_class/{label}"] = self.__dict__[f"miou_for_{label}"] / self.__dict__[f"count_for_{label}"]
            else:
                metrics_dictionary[f"validation_miou_class/{label}"] = self.__dict__[f"miou_for_{label}"] / self.__dict__[f"count_for_{label}"]
        return metrics_dictionary


class MemoryUsageLogger(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log_memory_usage(pl_module, "train_batch_end")

    def on_train_epoch_end(self, trainer, pl_module):
        self.log_memory_usage(pl_module, "train_epoch_end")

    def on_validation_epoch_end(self, trainer, pl_module):
        self.log_memory_usage(pl_module, "validation_epoch_end")

    def on_test_epoch_end(self, trainer, pl_module):
        self.log_memory_usage(pl_module, "test_epoch_end")

    def log_memory_usage(self, pl_module, log_point):
        memory_info = psutil.virtual_memory()
        cpu_memory_used = memory_info.used / (1024**3)  # Convert to GB
        cpu_memory_available = memory_info.available / (1024**3)  # Convert to GB
        # Use self.log to log metrics
        pl_module.logger.experiment.log({f"System/CPU Memory Used (GB) [{log_point}]": cpu_memory_used, f"System/CPU Memory Available (GB) [{log_point}]": cpu_memory_available})
