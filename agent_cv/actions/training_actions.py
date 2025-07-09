"""Training-related action functions for agents."""

import time
from pathlib import Path
import yaml

from omegaconf import OmegaConf
from lightning import Trainer

from yolo.config.config import Config
from yolo.tools.solver import TrainModel
from yolo.utils.logging_utils import setup

from ..models import TrainingConfig, TrainingResult, TrainingMetrics, DatasetInfo


def train_model(
    dataset_info: DatasetInfo,
    training_config: TrainingConfig,
    output_dir: Path,
) -> TrainingResult:
    """Train a YOLO model on the provided dataset using MultimediaTechLab YOLO.
    
    This function trains a YOLO model using the specified dataset and configuration.
    Training progress, metrics, and results are printed to stdout for agent visibility.
    The function handles both successful training and error cases gracefully.

    Args:
        dataset_info: Dataset information from analyze_dataset containing:
            - Dataset path and class information
            - Class distribution and counts
            - Train/val/test split details
        training_config: Training configuration parameters including:
            - Model type (e.g., 'yolov9c', 'yolov9e')
            - Training hyperparameters (epochs, batch_size, learning_rate)
            - Hardware settings (device, workers)
        output_dir: Directory to save model checkpoints and training logs

    Returns:
        TrainingResult: Comprehensive training results including:
            - Training status (completed/failed)
            - Best and final epoch metrics
            - Model checkpoint path
            - Training logs path
            - Total training time
            - Error information (if failed)
            
    Prints:
        Training progress, metrics, and final results to stdout for agent consumption
    """

    print(f"🚀 Starting model training")
    print(f"📊 Dataset Information:")
    print(dataset_info)
    print(f"⚙️  Training Configuration:")
    print(training_config)
    print(f"📁 Output Directory: {output_dir}")
    
    start_time = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Create YOLO configuration
        cfg = _create_yolo_config(dataset_info, training_config, output_dir)

        # Setup logging and callbacks
        callbacks, loggers, _ = setup(cfg)

        # Create Lightning trainer
        trainer = Trainer(
            accelerator="auto",
            max_epochs=training_config.epochs,
            precision="16-mixed",
            callbacks=callbacks,
            logger=loggers,
            log_every_n_steps=1,
            gradient_clip_val=10,
            gradient_clip_algorithm="value",
            deterministic=True,
            enable_progress_bar=True,
            default_root_dir=str(output_dir),
        )

        # Create training model
        model = TrainModel(cfg)

        # Start training
        trainer.fit(model)

        # Extract metrics from training logs
        best_metrics, final_metrics = _extract_lightning_metrics(
            trainer, training_config
        )

        # Find model checkpoint
        model_path = _find_best_checkpoint(output_dir)
        logs_path = output_dir

        total_time = time.time() - start_time

        result = TrainingResult(
            config=training_config,
            status="completed",
            best_epoch=best_metrics.epoch,
            best_metrics=best_metrics,
            final_metrics=final_metrics,
            total_epochs=training_config.epochs,
            total_time_seconds=total_time,
            model_path=model_path,
            logs_path=logs_path,
        )

        print(f"✅ Training completed successfully!")
        print(f"📈 Training Results:")
        print(result)
        return result

    except Exception as e:
        total_time = time.time() - start_time

        # Create dummy metrics for failed training
        dummy_metrics = TrainingMetrics(
            epoch=0,
            train_loss=float("inf"),
            learning_rate=training_config.learning_rate,
            time_seconds=total_time,
        )

        result = TrainingResult(
            config=training_config,
            status="failed",
            best_epoch=0,
            best_metrics=dummy_metrics,
            final_metrics=dummy_metrics,
            total_epochs=0,
            total_time_seconds=total_time,
            error_message=str(e),
        )
        
        print(f"❌ Training failed!")
        print(f"📈 Training Results:")
        print(result)
        return result


def _create_dataset_yaml(dataset_info: DatasetInfo, output_dir: Path) -> Path:
    """Create YOLO dataset configuration file."""

    # Extract class names from dataset info
    class_names = list(dataset_info.class_distribution.keys())

    dataset_config = {
        "path": str(dataset_info.path),
        "train": "train",  # Relative to path
        "val": "val",  # Relative to path
        "test": "test",  # Relative to path (optional)
        "nc": len(class_names),
        "names": class_names,
    }

    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_config, f, default_flow_style=False)

    return yaml_path


def _create_yolo_config(
    dataset_info: DatasetInfo, training_config: TrainingConfig, output_dir: Path
) -> Config:
    """Create YOLO configuration using OmegaConf for MultimediaTechLab YOLO."""

    # Create dataset YAML configuration
    dataset_yaml = _create_dataset_yaml(dataset_info, output_dir)

    # Create YOLO config dict structure matching their Config class
    config_dict = {
        "model": {
            "name": training_config.model_type,
        },
        "dataset": {
            "name": dataset_info.name,
            "path": str(dataset_yaml),
            "class_num": len(dataset_info.class_distribution),
            "class_list": list(dataset_info.class_distribution.keys()),
        },
        "task": {
            "task": "train",
            "epoch": training_config.epochs,
            "data": {
                "shuffle": True,
                "batch_size": training_config.batch_size,
                "pin_memory": True,
                "cpu_num": training_config.workers,
                "image_size": [training_config.image_size, training_config.image_size],
                "data_augment": {},
                "dataset": str(dataset_yaml),
            },
            "optimizer": {
                "type": "AdamW",
                "args": {
                    "lr": training_config.learning_rate,
                    "weight_decay": training_config.weight_decay,
                    "momentum": training_config.momentum,
                },
            },
            "loss": {
                "objective": {"box": 7.5, "cls": 0.5, "dfl": 1.5},
                "aux": 0.25,
                "matcher": {
                    "iou": "CIoU",
                    "topk": 10,
                    "factor": {"positive": 5.0, "negative": 7.0},
                },
            },
            "scheduler": {
                "type": "LinearWarmupCosineAnnealingLR",
                "warmup": {"epoch": 3, "multiplier": 0.1},
                "args": {"T_max": training_config.epochs},
            },
            "ema": {"enable": True, "decay": 0.9999},
            "validation": {
                "task": "validation",
                "data": {
                    "shuffle": False,
                    "batch_size": training_config.batch_size,
                    "pin_memory": True,
                    "cpu_num": training_config.workers,
                    "image_size": [
                        training_config.image_size,
                        training_config.image_size,
                    ],
                    "data_augment": {},
                    "dataset": str(dataset_yaml),
                },
                "nms": {
                    "min_confidence": 0.25,
                    "min_iou": 0.45,
                    "max_bbox": 300,
                },
            },
        },
        "name": f"{training_config.model_type}_{dataset_info.name}",
        "device": training_config.device,
        "cpu_num": training_config.workers,
        "weight": True,  # Use pretrained weights
        "quite": False,  # Show progress bar
        "image_size": [training_config.image_size, training_config.image_size],
    }

    # Add any additional hyperparameters
    if training_config.hyperparameters:
        for key, value in training_config.hyperparameters.items():
            # Navigate nested config structure
            if "." in key:
                parts = key.split(".")
                current = config_dict
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            else:
                config_dict[key] = value

    return OmegaConf.create(config_dict)  # type: ignore


def _extract_lightning_metrics(
    trainer, training_config: TrainingConfig
) -> tuple[TrainingMetrics, TrainingMetrics]:
    """Extract training metrics from Lightning trainer."""
    # Extract metrics from trainer logs
    if hasattr(trainer, "logged_metrics"):
        logged_metrics = trainer.logged_metrics
    else:
        logged_metrics = {}

    # Get current epoch metrics
    current_epoch = trainer.current_epoch if trainer.current_epoch is not None else 0

    # Create metrics for best and final epochs
    best_metrics = TrainingMetrics(
        epoch=current_epoch,
        train_loss=float(logged_metrics.get("train_loss", 0.0)),
        val_loss=(
            float(logged_metrics.get("val_loss", 0.0))
            if "val_loss" in logged_metrics
            else None
        ),
        precision=(
            float(logged_metrics.get("PyCOCO/AP @ .5", 0.0))
            if "PyCOCO/AP @ .5" in logged_metrics
            else None
        ),
        recall=None,  # Not directly available in logged metrics
        map50=(
            float(logged_metrics.get("PyCOCO/AP @ .5", 0.0))
            if "PyCOCO/AP @ .5" in logged_metrics
            else None
        ),
        map95=(
            float(logged_metrics.get("PyCOCO/AP @ .5:.95", 0.0))
            if "PyCOCO/AP @ .5:.95" in logged_metrics
            else None
        ),
        learning_rate=training_config.learning_rate,
        time_seconds=0.0,  # Would need to track this separately
    )

    # For simplicity, final metrics are same as best metrics
    final_metrics = best_metrics

    return best_metrics, final_metrics


def _find_best_checkpoint(output_dir: Path) -> Path | None:
    """Find the best model checkpoint in output directory."""
    # Look for Lightning checkpoint files
    checkpoint_patterns = ["**/*best*.ckpt", "**/best.pt", "**/last.ckpt", "**/*.ckpt"]

    for pattern in checkpoint_patterns:
        checkpoints = list(output_dir.glob(pattern))
        if checkpoints:
            return checkpoints[0]  # Return first match

    return None
