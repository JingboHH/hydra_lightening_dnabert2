from typing import Any, Dict, List, Tuple
import hydra
from omegaconf import DictConfig
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import Logger
from lightning.pytorch.callbacks import Callback
from lightning.pytorch import seed_everything
from src.utils import (
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)
from lightning import LightningDataModule, LightningModule
from lightning.pytorch.loggers import MLFlowLogger
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    # Instantiate model
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    print(f"Model type: {type(model)}")  # Debug log
    print(f"Model details: {model}")  # Debug log

    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))
    loggers: List[Logger] = instantiate_loggers(cfg.get("logger"))

    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=[],
        logger=loggers,
        accelerator="cpu",  # Ensure this matches your setup
    )

    log_hyperparameters(
        {"cfg": cfg, "model": model, "datamodule": datamodule, "trainer": trainer}
    )

    # Debugging the type of model
    print(f"Model type: {type(model)}")

    # Validate that the model is a LightningModule
    if not isinstance(model, LightningModule):
        raise TypeError(f"`model` must be a LightningModule, got {type(model)}")

    if cfg.get("train", True):
        trainer.fit(model=model, datamodule=datamodule)

    if cfg.get("test", False):
        ckpt_path = trainer.checkpoint_callback.best_model_path or None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    train_metrics = trainer.callback_metrics
    return train_metrics, {
        "cfg": cfg,
        "model": model,
        "datamodule": datamodule,
        "trainer": trainer,
    }


@hydra.main(config_path="../configs", config_name="train.yaml", version_base="1.3")
def main(cfg: DictConfig):
    datamodule = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)

    # Trainer with dynamic logger
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)

    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)


if __name__ == "__main__":
    main()
