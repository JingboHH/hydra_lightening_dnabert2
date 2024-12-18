from typing import Any, Dict, Optional
import hydra
from omegaconf import DictConfig
from lightning.pytorch import Trainer
from lightning.pytorch import seed_everything
from src.utils import task_wrapper


@task_wrapper
def evaluate(cfg: DictConfig) -> Dict[str, Any]:
    """
    Evaluates the DNABERT2 model using the test dataset.
    :param cfg: Configuration composed by Hydra.
    :return: Evaluation metrics.
    """
    # Set random seed for reproducibility
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    # Instantiate datamodule
    datamodule = hydra.utils.instantiate(cfg.data)

    # Instantiate model
    model = hydra.utils.instantiate(cfg.model)

    # Instantiate trainer
    trainer = hydra.utils.instantiate(cfg.trainer)

    # Run evaluation
    ckpt_path = cfg.get("ckpt_path", None)
    if not ckpt_path:
        print("No checkpoint path provided. Using current model weights.")
        ckpt_path = None

    metrics = trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    print(f"Evaluation Metrics: {metrics}")
    return metrics


@hydra.main(config_path="../configs", config_name="eval.yaml", version_base="1.3")
def main(cfg: DictConfig) -> Optional[float]:
    """
    Main entry point for DNABERT2 evaluation.
    :param cfg: Configuration composed by Hydra.
    :return: Optional optimized metric for hyperparameter tuning.
    """
    metrics = evaluate(cfg)

    # Retrieve optimized metric
    optimized_metric = cfg.get("optimized_metric")
    if optimized_metric and optimized_metric in metrics[0]:
        return metrics[0][optimized_metric]

    return None


if __name__ == "__main__":
    main()
