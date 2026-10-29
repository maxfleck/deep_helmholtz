"""
Ray Tune hyperparameter optimization for HEOSNet.

Usage:
    python -m helmholtz_eos_net.ray_wrapper --config config_new.json --num-samples 50

This module wraps train_orchestrated() / GoalEvaluator with Ray Tune
for automated hyperparameter search using ASHA scheduler.
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Any

import torch
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.air import RunConfig

from helmholtz_eos_net.training_orchestrator import GoalEvaluator
from helmholtz_eos_net.train_orchestrated import train_orchestrated
from helmholtz_eos_net.model import HEOSNet
from helmholtz_eos_net.substance_parameters import SubstanceParameters


# =============================================================================
# SEARCH SPACE DEFINITION
# =============================================================================

def get_search_space() -> Dict[str, Any]:
    """x Local hyperparameter search space around the current best config."""
    return {
        # Bias towards 2–3 layers, but keep 1 and 4 as occasional outliers.
        "depth": tune.choice([2, 3, 3, 3]),

        # Bias towards 512–1024, but still occasionally test 128, 256, 2048.
        "width": tune.choice([512, 512, 768, 1024, 1024, 1024, 2048]),

        # Favor tanh/gelu, but keep sigmoid/softplus to maintain diversity.
        "activation": tune.choice(
            ["gelu", "gelu", "gelu"]
        ),

        # Mild variation in dropout.
        "dropout_rate": tune.uniform(0.08, 0.12),

        # Optimizer: narrow band around incumbent.
        # Center ~1.8e-4 with about one order of magnitude span.
        "lr": tune.loguniform(6e-5, 6e-4),

        # Center ~1.8e-8, allow ~1 order of magnitude.
        "weight_decay": tune.loguniform(3e-9, 3e-8),

        # Batch size: small local variation to change noise profile.
        "batch_size": tune.choice([64, 128, 256, 512]),
    }

def xx_get_search_space() -> Dict[str, Any]:
    """ini Define the hyperparameter search space."""
    return {
        # Architecture hyperparameters
        "depth": tune.choice([1, 2, 3, 4]),
        "width": tune.choice([128, 256, 512, 1024, 2048]),
        "activation": tune.choice(["tanh", "gelu", "sigmoid", "softplus"]),
        "dropout_rate": tune.uniform(0.0, 0.2),

        # Optimizer hyperparameters
        "lr": tune.loguniform(1e-5, 1e-2),
        "weight_decay": tune.loguniform(1e-8, 1e-4),

        # Training hyperparameters
        "batch_size": tune.choice([32, 64, 128, 256]),
    }


# =============================================================================
# TRAINABLE FUNCTION
# =============================================================================

def train_heos_tune(config: Dict[str, Any]):
    """
    Ray Tune trainable function that wraps train_orchestrated().

    Parameters
    ----------
    config : dict
        Trial hyperparameters from Ray Tune search space, plus:
        - base_config: Base configuration with data paths and fixed settings
        - substance_params_path: Path to substance parameters CSV
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Extract base config and substance params info
    base_config = config.pop("base_config")
    substance_params_path = config.pop("substance_params_path")
    id_column = config.pop("id_column")
    parameter_columns = config.pop("parameter_columns")
    fixed_columns = config.pop("fixed_columns")
    results_dir = config.pop("results_dir", "")
    experiment_name = config.pop("experiment_name", "")

    # Merge trial config with base config
    trial_config = {**base_config, **config}

    # Load substance parameters (can't be serialized, so load per trial)
    substance_params = SubstanceParameters(
        csv_path=substance_params_path,
        id_column=id_column,
        parameter_columns=parameter_columns,
        fixed_columns=fixed_columns,
        embedding_dim=len(parameter_columns),
    ).to(device)

    # Create model with trial architecture hyperparameters
    model = HEOSNet(
        substance_params=substance_params,
        depth=config["depth"],
        width=config["width"],
        activation=config["activation"],
        dropout_rate=config["dropout_rate"],
        apply_bias=trial_config.get('apply_bias', True),
        use_output_scale=False,
        training_mode=trial_config.get("training_mode", "gradients"),
    ).to(device)

    # GoalEvaluator handles goal routing, data loading (residual + VLE), and loss weighting
    evaluator = GoalEvaluator(model=model, config=trial_config)

    # A_RES PRETRAINING — invisible to ASHA (epoch_callback=None)
    if (trial_config.get("pretrain", False)
            and trial_config.get("training_preset", "all_residuals") != "a_res_only"):
        pretrain_lr = config["lr"] * trial_config.get("pretrain_lr_factor", 10.0)
        pretrain_config = {
            **trial_config,
            "training_preset": "a_res_only",
            "lr": pretrain_lr,
            "patience": trial_config.get("patience_pretrain", 10),
            "max_epochs": trial_config.get("max_epochs_pretrain", 10),
        }
        pretrain_evaluator = GoalEvaluator(model=model, config=pretrain_config)
        model, _, _, _ = train_orchestrated(
            model=model,
            evaluator=pretrain_evaluator,
            config=pretrain_config,
            optimizer=None,
            device=device,
            verbose=False,
            epoch_callback=None,
        )

    # Create main optimizer after pretraining so its momentum is clean
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    # Per-epoch callback for ASHA intermediate reporting
    def epoch_callback(epoch, val_loss, component_losses=None, train_loss=None):
        metrics = {"val_loss": val_loss, "training_iteration": epoch}
        if train_loss is not None:
            metrics["train_loss"] = train_loss
        if component_losses:
            for k, v in component_losses.items():
                metrics[f"val_{k}"] = v
        tune.report(metrics)

    # Point model_path at the Ray trial directory so train_orchestrated writes
    # checkpoint files there directly — each trial gets its own isolated path.
    # If trial_dir falls under /tmp/ (some Ray versions), redirect to a persistent
    # path under RESULTS_DIR so checkpoints survive a reboot.
    from ray.air import session as ray_session
    trial_dir = ray_session.get_trial_dir()
    trial_id = ray_session.get_trial_id() if trial_dir else ""
    print(f"[Ray] trial_dir: {trial_dir}")

    if trial_dir and "/tmp/" in trial_dir and results_dir and experiment_name and trial_id:
        persistent_dir = os.path.join(results_dir, experiment_name, trial_id)
        os.makedirs(persistent_dir, exist_ok=True)
        trial_config["model_path"] = persistent_dir
    elif trial_dir:
        trial_config["model_path"] = trial_dir

    # Run training — wrap in try/finally so files are saved even when ASHA
    # prunes the trial mid-run by raising inside tune.report()
    train_report = {}
    try:
        model, optimizer, _, train_report = train_orchestrated(
            model=model,
            evaluator=evaluator,
            config=trial_config,
            optimizer=optimizer,
            device=device,
            verbose=False,
            epoch_callback=epoch_callback,
        )
    finally:
        # best_checkpoint.pt and best_model.json are already written inside the
        # training loop — they survive pruning. Only training_report.json needs
        # to be saved here since it requires the full completed run.
        report_target = trial_config.get("model_path", trial_dir) if trial_dir else ""
        if report_target and train_report:
            with open(os.path.join(report_target, "training_report.json"), "w") as f:
                json.dump(train_report, f, indent=2)
        # If trial_dir was under /tmp/, copy its contents to the persistent dir
        # (Ray may write additional metadata files we want to keep).
        if (trial_dir and "/tmp/" in trial_dir
                and results_dir and experiment_name and trial_id):
            persistent_dir = trial_config.get("model_path", "")
            if persistent_dir and os.path.exists(trial_dir):
                for fname in os.listdir(trial_dir):
                    src = os.path.join(trial_dir, fname)
                    dst = os.path.join(persistent_dir, fname)
                    if os.path.isfile(src) and not os.path.exists(dst):
                        shutil.copy2(src, dst)


# =============================================================================
# MAIN TUNING FUNCTION
# =============================================================================

def run_hyperparameter_tuning(
    config_path: str,
    num_samples: int = 50,
    max_epochs: int = 100,
    gpus_per_trial: float = 1.0,
    cpus_per_trial: int = 8,
    results_dir: str = "ray_results",
    experiment_name: str = "heos_tune",
    grace_period: int = 10,
    num_cpus: int = 16,
):
    """
    Run hyperparameter tuning with Ray Tune.

    Parameters
    ----------
    config_path : str
        Path to base configuration JSON file
    num_samples : int
        Number of hyperparameter configurations to try
    max_epochs : int
        Maximum epochs per trial
    gpus_per_trial : float
        GPU resources per trial (0.5 for 2 trials per GPU)
    cpus_per_trial : int
        CPU resources per trial
    results_dir : str
        Directory to store Ray Tune results
    experiment_name : str
        Name for this tuning experiment
    grace_period : int
        Minimum epochs before ASHA can stop a trial
    """
    # Resolve results_dir to absolute so it stays valid inside Ray workers
    results_dir = str(Path(results_dir).resolve())

    # Load base configuration
    with open(config_path, 'r') as f:
        base_config = json.load(f)

    # Resolve all path fields to absolute — Ray workers change CWD so relative
    # paths break. Paths in the config are relative to the config file's directory.
    config_dir = Path(config_path).resolve().parent
    _path_keys = [
        'data_csv_train', 'data_csv_val',
        'substance_params_csv',
        'vle_data_csv_train', 'vle_data_csv_val',
        'model_path', 'pretrained_model_path',
    ]
    for key in _path_keys:
        if base_config.get(key):
            p = Path(base_config[key])
            if not p.is_absolute():
                base_config[key] = str(config_dir / p)

    # Override max_epochs and disable auto-resume
    base_config['max_epochs'] = max_epochs
    base_config['resume_from_checkpoint'] = False

    # Initialize Ray
    ray.init(ignore_reinit_error=True, num_cpus=num_cpus)

    # Configure ASHA scheduler for early stopping
    # metric/mode go on the scheduler only — putting them in both scheduler
    # and TuneConfig causes a ValueError in Ray 2.53+ ("already instantiated")
    scheduler = ASHAScheduler(
        metric="val_loss",
        mode="min",
        max_t=max_epochs,
        grace_period=grace_period,
        reduction_factor=3,
    )

    # Build search space with base config info
    search_space = get_search_space()
    search_space["base_config"] = base_config
    search_space["substance_params_path"] = base_config['substance_params_csv']
    search_space["id_column"] = base_config['id_column']
    search_space["parameter_columns"] = base_config['parameter_columns']
    search_space["fixed_columns"] = base_config.get('fixed_columns', [])
    search_space["results_dir"] = os.path.abspath(results_dir)
    search_space["experiment_name"] = experiment_name

    # Configure tuner — metric/mode NOT repeated here (already on the scheduler)
    tuner = tune.Tuner(
        tune.with_resources(
            train_heos_tune,
            resources={"cpu": cpus_per_trial, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=search_space,
        run_config=RunConfig(
            name=experiment_name,
            storage_path=os.path.abspath(results_dir),
        ),
    )

    # Run tuning
    print(f"Starting hyperparameter tuning with {num_samples} trials...")
    print(f"Results will be saved to: {results_dir}/{experiment_name}")
    results = tuner.fit()

    # Get best result
    best_result = results.get_best_result(metric="val_loss", mode="min")
    best_config = best_result.config
    best_metrics = best_result.metrics

    print("\n" + "=" * 80)
    print("HYPERPARAMETER TUNING COMPLETE")
    print("=" * 80)
    print(f"\nBest validation loss: {best_metrics['val_loss']:.6f}")
    print(f"\nBest hyperparameters:")
    tuned_params = ["depth", "width", "activation", "dropout_rate", "lr", "weight_decay", "batch_size"]
    for key in tuned_params:
        if key in best_config:
            print(f"  {key}: {best_config[key]}")

    # Save best config
    output_config = {**base_config}
    for key in tuned_params:
        if key in best_config:
            output_config[key] = best_config[key]

    experiment_dir = Path(results_dir) / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Save best config
    output_path = experiment_dir / "best_config.json"
    with open(output_path, 'w') as f:
        json.dump(output_config, f, indent=2)
    print(f"\nBest configuration saved to: {output_path}")

    # Copy best trial's files to best_result/.
    # If best_result.path is under /tmp/, we need two sources:
    #   1. best_result.path  — Ray metadata (params.json, result.json, …)
    #   2. RESULTS_DIR/experiment_name/<trial_id>/  — training artifacts
    #      (best_checkpoint.pt, best_model.json, training_report.json)
    # If best_result.path is already under RESULTS_DIR, one copytree suffices.
    best_trial_path = best_result.path
    best_result_dir = experiment_dir / "best_result"
    if best_result_dir.exists():
        shutil.rmtree(best_result_dir)
    best_result_dir.mkdir(parents=True, exist_ok=True)

    sources = []
    if best_trial_path and os.path.exists(best_trial_path):
        sources.append(best_trial_path)
        if "/tmp/" in best_trial_path:
            # Also pull training artifacts from the persistent per-trial dir
            trial_id = best_result.metrics.get("trial_id", "") if best_result.metrics else ""
            if trial_id:
                persistent_trial_dir = str(experiment_dir / trial_id)
                if os.path.exists(persistent_trial_dir):
                    sources.append(persistent_trial_dir)

    for src_dir in sources:
        for fname in os.listdir(src_dir):
            src = os.path.join(src_dir, fname)
            dst = str(best_result_dir / fname)
            if os.path.isfile(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
            elif os.path.isdir(src) and not os.path.exists(dst):
                shutil.copytree(src, dst)
    if sources:
        print(f"Best trial folder assembled at: {best_result_dir}")

    # Save results DataFrame
    import pandas as pd
    df = results.get_dataframe(filter_metric="val_loss", filter_mode="min")
    df = df.rename(columns={col: col.split("/", 1)[1] for col in df.columns if col.startswith("config/")})
    csv_path = experiment_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results DataFrame saved to: {csv_path}")

    # Save ranked summary JSON (build from the already-computed DataFrame)
    _bm = best_metrics or {}
    _bc = best_config or {}
    summary = {
        "best_val_loss": float(_bm.get("val_loss", float("inf"))),
        "best_trial_id": _bm.get("trial_id", ""),
        "best_hyperparameters": {k: _bc[k] for k in tuned_params if k in _bc},
        "trials": [],
    }
    sort_col = "val_loss" if "val_loss" in df.columns else df.columns[0]
    for _, row in df.sort_values(sort_col).iterrows():
        entry = {
            "trial_id": row.get("trial_id", ""),
            "val_loss": row.get("val_loss"),
            "total_epochs": row.get("training_iteration"),
            "training_time_s": row.get("time_total_s"),
        }
        for k in tuned_params:
            if k in row:
                entry[k] = row[k]
        summary["trials"].append(entry)
    summary_path = experiment_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    return results


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run hyperparameter tuning for HEOSNet with Ray Tune"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config_new.json",
        help="Path to base configuration file"
    )
    parser.add_argument(
        "--num-samples", "-n",
        type=int,
        default=50,
        help="Number of hyperparameter configurations to try"
    )
    parser.add_argument(
        "--max-epochs", "-e",
        type=int,
        default=100,
        help="Maximum epochs per trial"
    )
    parser.add_argument(
        "--gpus-per-trial",
        type=float,
        default=1.0,
        help="GPU resources per trial (e.g., 0.5 for 2 trials per GPU)"
    )
    parser.add_argument(
        "--cpus-per-trial",
        type=int,
        default=2,
        help="CPU resources per trial"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="ray_results",
        help="Directory for Ray Tune results"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="heos_tune",
        help="Name for this experiment"
    )
    parser.add_argument(
        "--grace-period",
        type=int,
        default=10,
        help="Minimum epochs before ASHA scheduler can stop a trial"
    )

    args = parser.parse_args()

    run_hyperparameter_tuning(
        config_path=args.config,
        num_samples=args.num_samples,
        max_epochs=args.max_epochs,
        gpus_per_trial=args.gpus_per_trial,
        cpus_per_trial=args.cpus_per_trial,
        results_dir=args.results_dir,
        experiment_name=args.experiment_name,
        grace_period=args.grace_period,
    )


if __name__ == "__main__":
    main()
