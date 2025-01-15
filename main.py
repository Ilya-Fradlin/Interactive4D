import logging
import os
import argparse
from datetime import datetime

import torch
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from trainer.trainer import ObjectSegmentation
from utils.utils import flatten_dict
from models.metrics.utils import MemoryUsageLogger


def configure_environment(cfg: DictConfig):
    """
    Configure the runtime environment and initialize seed for reproducibility.
    """
    load_dotenv(".env")
    seed_everything(cfg.general.seed)
    cfg.general.gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {cfg.general.gpus}")


def setup_loggers(cfg: DictConfig):
    """
    Configure loggers based on the experiment mode.
    """
    loggers = []
    cfg.logging.save_dir = os.path.join("saved", cfg.general.experiment_name)

    os.makedirs(cfg.logging.save_dir, exist_ok=True)
    loggers.append(
        WandbLogger(
            project=cfg.logging.project_name,
            name=cfg.general.experiment_name,
            save_dir=cfg.logging.save_dir,
            id=cfg.general.experiment_name,
            entity=cfg.logging.entity,
        )
    )
    loggers[-1].log_hyperparams(flatten_dict(OmegaConf.to_container(cfg, resolve=True)))

    return loggers


def get_parameters(cfg: DictConfig):
    """
    Parse and prepare model parameters, loggers, and configurations.
    """
    if cfg.general.debug:
        os.environ["WANDB_MODE"] = "dryrun"
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
        os.environ["GLOO_LOG_LEVEL"] = "DEBUG"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"

        cfg.data.dataloader.voxel_size = 0.1
        cfg.trainer.num_sanity_val_steps = 0
        cfg.trainer.num_devices = 1
        cfg.trainer.num_nodes = 1
        cfg.trainer.detect_anomaly = True
        cfg.trainer.log_every_n_steps = 1
        cfg.trainer.max_epochs = 5
        cfg.trainer.check_val_every_n_epoch = 5
        cfg.trainer.limit_train_batches = 2
        cfg.trainer.limit_val_batches = 2

        os.environ.update({"WANDB_MODE": "dryrun", "TORCH_DISTRIBUTED_DEBUG": "DETAIL", "TORCH_CPP_LOG_LEVEL": "INFO", "GLOO_LOG_LEVEL": "DEBUG", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})

    logger = logging.getLogger(__name__)
    configure_environment(cfg)
    loggers = setup_loggers(cfg)
    model = ObjectSegmentation(cfg)
    logger.info(flatten_dict(OmegaConf.to_container(cfg, resolve=True)))
    return cfg, model, loggers


def configure_callbacks(cfg: DictConfig):
    """
    Set up training callbacks for model checkpointing, learning rate monitoring, etc.
    """
    return [
        ModelCheckpoint(
            verbose=True,
            save_top_k=1,
            save_last=True,
            monitor="mIoU_epoch",
            mode="max",
            dirpath=cfg.logging.save_dir,
            every_n_epochs=1,
            filename="{epoch:02d}-{mIoU_epoch:.3f}",
            save_on_train_epoch_end=True,
        ),
        LearningRateMonitor(),
        MemoryUsageLogger(),
    ]


def train(cfg: DictConfig):
    """
    Train the model based on the provided configuration.
    """
    cfg, model, loggers = get_parameters(cfg)
    callbacks = configure_callbacks(cfg)
    runner = Trainer(
        callbacks=callbacks,
        logger=loggers,
        default_root_dir=cfg.logging.save_dir,
        devices=cfg.trainer.num_devices,
        num_nodes=cfg.trainer.num_nodes,
        accelerator=cfg.trainer.accelerator,
        max_epochs=cfg.trainer.max_epochs,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        limit_train_batches=cfg.trainer.limit_train_batches,
        limit_val_batches=cfg.trainer.limit_val_batches,
        detect_anomaly=cfg.trainer.detect_anomaly,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        strategy="ddp_find_unused_parameters_false",
    )
    runner.fit(model, ckpt_path=cfg.general.ckpt_path)


def validate(cfg: DictConfig):
    """
    Validate the model using the validation dataset.
    """
    cfg, model, loggers = get_parameters(cfg)
    runner = Trainer(
        logger=loggers,
        default_root_dir=cfg.logging.save_dir,
        devices=1,
        num_nodes=1,
        accelerator="gpu",
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        limit_val_batches=cfg.trainer.limit_train_batches,
    )
    runner.validate(model, ckpt_path=cfg.general.ckpt_path)


def main():
    """
    Entry point for the script. Load the configuration and determine the mode of operation.
    """

    # Parse command-line arguments for overrides
    parser = argparse.ArgumentParser(description="Override config parameters.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--validate", action="store_true", help="Run Evaluation mode")
    parser.add_argument("--voxel_size", type=float, help="Specify the voxel size")
    parser.add_argument("--num_sweeps", type=int, help="Specify the number of sweeps")
    args = parser.parse_args()

    # Load configuration
    base_cfg = OmegaConf.load(os.path.join("conf", "config.yaml"))
    if args.validate:
        base_cfg.general.mode = "validate"
    mode = base_cfg.general.mode
    mode_cfg = base_cfg.modes.get(mode, {})
    cfg = OmegaConf.merge(base_cfg, mode_cfg)
    cfg.general.experiment_name = cfg.general.experiment_name.replace("now", datetime.now().strftime("%Y-%m-%d_%H%M%S"))
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    if args.debug:
        cfg.general.debug = True
    if args.voxel_size is not None:
        cfg.data.dataloader.voxel_size = args.voxel_size
        print(f"Voxel size changed to {cfg.data.dataloader.voxel_size}")
    if args.num_sweeps is not None:
        cfg.data.datasets.sweep = args.num_sweeps
        print(f"Number of sweeps changed to {cfg.data.datasets.sweep}")

    # Execute based on mode
    if mode == "train":
        train(cfg)
    elif mode == "validate":
        validate(cfg)
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
