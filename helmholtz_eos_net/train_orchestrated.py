"""
Orchestrated Training module for HEOSNet with goal-based training.

Uses GoalEvaluator from training_orchestrator.py for dynamic, goal-based training.
Supports arbitrary combinations of training goals via presets.

Provides:
- train_orchestrated(): Single-phase training with configurable goals
- train_manager_orchestrated(): High-level manager that loads config and runs training
"""

import numpy as np
import copy
import os
import json
import time
from collections import defaultdict
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from helmholtz_eos_net.model import HEOSNet
from helmholtz_eos_net.substance_parameters import SubstanceParameters, TrainableSubstanceParameters
from helmholtz_eos_net.training_orchestrator import GoalEvaluator


# =============================================================================
# OPTIMIZER HELPER
# =============================================================================

def _make_optimizer(model, lr, weight_decay, transport_lr=None):
    """Adam with separate param groups: base LR for Helmholtz, transport_lr for transport_params."""
    transport_lr = transport_lr if transport_lr is not None else lr
    tp_params = (list(model.transport_params.parameters())
                 if model.transport_params is not None else [])
    tp_ids = {id(p) for p in tp_params}
    base_params = [p for p in model.parameters() if id(p) not in tp_ids]

    groups = [{'params': base_params, 'lr': lr}]
    if tp_params:
        groups.append({'params': tp_params, 'lr': transport_lr, 'is_transport': True})
    return torch.optim.Adam(groups, weight_decay=weight_decay)


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train_orchestrated(
    model: HEOSNet,
    evaluator: GoalEvaluator,
    config: Dict,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
    epoch_callback=None,
) -> Tuple[HEOSNet, torch.optim.Optimizer, Dict, Dict]:
    """
    Orchestrated training function for goal-based training.

    Uses GoalEvaluator to handle forward passes, target normalization,
    and loss computation for arbitrary goal combinations.

    Parameters:
    -----------
    model : HEOSNet
        Model instance (already initialized)
    evaluator : GoalEvaluator
        Goal evaluator configured with desired training preset
    config : Dict
        Training configuration with keys:
        - lr: learning rate
        - weight_decay: L2 regularization (default: 1e-7)
        - max_epochs: maximum epochs
        - patience: early stopping patience
        - rampup_epochs: epochs before early stopping kicks in (optional)
        - gradient_clip_norm: max gradient norm (default: 10.0)
        - batch_size: batch size for dataloaders
    optimizer : torch.optim.Optimizer, optional
        If None, creates fresh Adam optimizer
    device : torch.device, optional
        Device to use (auto-detected if None)
    verbose : bool
        Print progress (default: True)

    Returns:
    --------
    model : HEOSNet
        Trained model with best weights restored
    optimizer : torch.optim.Optimizer
        Optimizer with state preserved
    best_model_dict : Dict
        Best model checkpoint with metadata
    train_report : Dict
        Training report with metrics and loss histories
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.train()

    # Extract config
    lr = config.get("lr", 1e-4)
    weight_decay = config.get("weight_decay", 1e-7)
    max_epochs = config.get("max_epochs", 100)
    patience = config.get("patience", 20)
    rampup_epochs = config.get("rampup_epochs", None)
    rampup_epochs = rampup_epochs if rampup_epochs and rampup_epochs > 0 else -1
    gradient_clip_norm = config.get("gradient_clip_norm", 10.0)
    batch_size = config.get("batch_size", 64)
    lipschitz_penalty = config.get("lipschitz_penalty", False)
    lipschitz_target  = config.get("lipschitz_target", 400.0)
    lipschitz_lambda  = config.get("lipschitz_lambda", 0.01)
    model_path        = config.get("model_path", "")
    if model_path:
        os.makedirs(model_path, exist_ok=True)

    # Create or reuse optimizer — transport params get their own (higher) LR
    transport_lr = config.get("transport_lr", lr * config.get("transport_lr_scale", 1000.0))
    if optimizer is None:
        optimizer = _make_optimizer(model, lr, weight_decay, transport_lr)
    else:
        for param_group in optimizer.param_groups:
            if param_group.get("is_transport"):
                param_group["lr"] = transport_lr
            else:
                param_group["lr"] = lr

    # Create dataloaders from evaluator
    train_loader, val_loader = evaluator.create_dataloaders(batch_size=batch_size)

    # Tracking
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_model_dict = None

    train_losses = defaultdict(list)
    val_losses = defaultdict(list)

    # Get target feature names for logging
    target_names = evaluator.config.get("target_features", [])

    start_time = time.time()

    for epoch in range(max_epochs):
        # =================================================================
        # TRAINING
        # =================================================================
        model.train()
        train_loss_batch = defaultdict(list)

        for batches in train_loader:
            optimizer.zero_grad()

            # Forward pass through evaluator
            result = evaluator.forward(batches, return_jacobian=lipschitz_penalty)
            losses, weights = result[0], result[1]

            # Compute weighted total loss
            total_loss = (losses * weights).sum()

            # Lipschitz gradient penalty (for fine-tuning stabilization)
            if lipschitz_penalty and len(result) > 2:
                jacobian_norms = result[2]
                lip_penalty = lipschitz_lambda * torch.clamp(
                    jacobian_norms - lipschitz_target, min=0
                ).pow(2).mean()
                total_loss = total_loss + lip_penalty

            # Skip batch if NaN/Inf or exploding loss
            if not torch.isfinite(total_loss) or total_loss > 1e6:
                optimizer.zero_grad()
                continue

            # Track individual losses
            losses_np = losses.detach().cpu().numpy()
            weights_np = weights.detach().cpu().numpy()
            for i, name in enumerate(target_names):
                if i < len(losses_np):
                    train_loss_batch[name].append(float(losses_np[i]))
            train_loss_batch["total"].append(total_loss.item())

            # Backward pass
            total_loss.backward()
            # Skip if any gradient is NaN/Inf — prevents weight corruption
            if any(p.grad is not None and not torch.isfinite(p.grad).all()
                   for p in model.parameters()):
                optimizer.zero_grad()
                continue
            tp_ids = {id(p) for p in (model.transport_params.parameters()
                      if model.transport_params is not None else [])}
            base_params = [p for p in model.parameters() if id(p) not in tp_ids]
            torch.nn.utils.clip_grad_norm_(base_params, max_norm=gradient_clip_norm)
            optimizer.step()

        # Average training losses for epoch
        for key in train_loss_batch:
            train_losses[key].append(np.mean(train_loss_batch[key]))

        # =================================================================
        # VALIDATION
        # =================================================================
        model.eval()
        val_loss_batch = defaultdict(list)
        n_val_batches = 0
        n_val_batches_finite = 0

        for batches in val_loader:
            n_val_batches += 1
            # Model needs gradient tracking for internal autograd.grad() calls
            losses, weights = evaluator.forward(batches, update_ema=False)

            # Replace NaN/Inf per-goal — don't discard the whole batch
            if verbose and (torch.any(torch.isnan(losses)) or torch.any(torch.isinf(losses))):
                nan_goals = [target_names[i] for i in range(len(losses)) if not torch.isfinite(losses[i])]
                print(f"  [val] NaN/Inf in: {nan_goals}")
            finite_mask = torch.isfinite(losses)
            losses_np = losses.detach().cpu().numpy()
            for i, name in enumerate(target_names):
                if i < len(losses_np) and finite_mask[i]:
                    val_loss_batch[name].append(float(losses_np[i]))
            # total: sum only finite goals — batch still contributes if any goal is finite
            if finite_mask.any():
                n_val_batches_finite += 1
                val_loss_batch["total"].append(losses[finite_mask].sum().item())
                val_loss_batch["total_weighted"].append((losses[finite_mask] * weights[finite_mask]).sum().item())

        # Average validation losses for epoch
        for key in val_loss_batch:
            val_losses[key].append(np.mean(val_loss_batch[key]))

        # Coverage-adjusted val loss: penalises models where most batches are NaN/masked.
        # If only 1/38 batches contributed, adjusted = raw × 38 — a bad model looks bad.
        val_coverage = n_val_batches_finite / max(n_val_batches, 1)
        raw_val_loss = val_losses["total"][-1] if val_losses["total"] else float("inf")
        current_val_loss = raw_val_loss / max(val_coverage, 1.0 / max(n_val_batches, 1))
        if val_losses["total"]:
            val_losses["total"][-1] = current_val_loss

        if epoch_callback is not None:
            component_losses = {k: v[-1] for k, v in val_losses.items() if v and k != "total"}
            train_loss = train_losses["total"][-1] if train_losses["total"] else None
            epoch_callback(epoch, current_val_loss, component_losses, train_loss)

        # =================================================================
        # EARLY STOPPING LOGIC with Ramp-up Support
        # =================================================================
        if rampup_epochs > 0:
            # Ramp-up period: disable early stopping for first rampup_epochs
            if epoch < rampup_epochs:
                # Just track best, no patience accumulation
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    best_epoch = epoch
                    best_model_dict = {
                        "model_state_dict": copy.deepcopy(model.state_dict()),
                        "epoch": epoch,
                        "val_loss": best_val_loss,
                        "config": {k: v for k, v in config.items()
                                   if isinstance(v, (str, int, float, bool, type(None)))},
                    }
            elif epoch == rampup_epochs:
                # End of ramp-up: reset patience counter
                best_val_loss = float("inf")
                patience_counter = 0
                best_model_dict = None
            else:
                # Normal early stopping (after ramp-up)
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    best_epoch = epoch
                    patience_counter = 0
                    best_model_dict = {
                        "model_state_dict": copy.deepcopy(model.state_dict()),
                        "epoch": epoch,
                        "val_loss": best_val_loss,
                        "config": {k: v for k, v in config.items()
                                   if isinstance(v, (str, int, float, bool, type(None)))},
                    }
                else:
                    patience_counter += 1
        else:
            # No ramp-up: early stopping from epoch 0
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_epoch = epoch
                patience_counter = 0
                best_model_dict = {
                    "model_state_dict": copy.deepcopy(model.state_dict()),
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                }
            else:
                patience_counter += 1

        # Persist best model immediately so it survives pruning/crashes
        if model_path and best_model_dict is not None and best_model_dict.get("epoch") == epoch:
            torch.save(best_model_dict, os.path.join(model_path, "best_checkpoint.pt"))
            torch.save(best_model_dict["model_state_dict"], os.path.join(model_path, "best_model.pt"))
            with open(os.path.join(model_path, "best_model.json"), "w") as f:
                json.dump({
                    "best_val_loss": float(best_val_loss),
                    "best_epoch": int(best_epoch),
                    "epoch_reported": int(epoch),
                    "config": {k: v for k, v in config.items() if isinstance(v, (str, int, float, bool, type(None)))},
                    "val_losses": {k: [float(x) for x in v] for k, v in val_losses.items()},
                    "train_losses": {k: [float(x) for x in v] for k, v in train_losses.items()},
                }, f, indent=2)

        # Print progress
        if verbose and ((epoch + 1) % 10 == 0 or epoch < 10):
            elapsed = time.time() - start_time
            train_total = train_losses['total'][-1] if train_losses['total'] else 0
            val_total = val_losses['total'][-1] if val_losses['total'] else 0

            print(
                f"Epoch {epoch+1:4d} | "
                f"Train: {train_total:.6f} | "
                f"Val: {val_total:.6f} | "
                f"Cov: {val_coverage:.2f} | "
                f"Best: {best_val_loss:.6f} @ {best_epoch+1} | "
                f"Patience: {patience_counter}/{patience} | "
                f"Time: {elapsed:.1f}s"
            )

            # Print per-goal losses
            if target_names and verbose:
                loss_parts = []
                for name in target_names:
                    if name in val_losses and val_losses[name]:
                        loss_parts.append(f"{name}={val_losses[name][-1]:.6f}")
                if loss_parts:
                    print(f"  Components: {' | '.join(loss_parts)}")

            # Print weights
            if 'weights_np' in dir() and weights_np is not None:
                weight_parts = []
                for i, name in enumerate(target_names):
                    if i < len(weights_np):
                        weight_parts.append(f"{name}={weights_np[i]:.3f}")
                if weight_parts:
                    print(f"  Weights: {' | '.join(weight_parts)}")

        # Early stopping condition
        should_stop = (
            (rampup_epochs <= 0 or epoch >= rampup_epochs) and
            patience_counter >= patience
        )
        if should_stop:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}\n")
            break

    # =========================================================================
    # RESTORE BEST MODEL
    # =========================================================================
    if best_model_dict is not None:
        model.load_state_dict(best_model_dict["model_state_dict"])

    elapsed = time.time() - start_time
    if verbose:
        print(f"Training complete in {elapsed:.1f}s\n")

    # =========================================================================
    # TRAINING REPORT
    # =========================================================================
    train_report = {
        "best_val_loss": float(best_val_loss),
        "best_epoch": best_epoch,
        "total_epochs": len(train_losses["total"]),
        "train_losses": {k: [float(x) for x in v] for k, v in train_losses.items()},
        "val_losses": {k: [float(x) for x in v] for k, v in val_losses.items()},
        "training_preset": evaluator.config.get("training_preset", "unknown"),
        "target_features": target_names,
    }

    return model, optimizer, best_model_dict, train_report


# =============================================================================
# MANAGER FUNCTION
# =============================================================================

def train_manager_orchestrated(config: Dict, device=None, config_path=None) -> Tuple[HEOSNet, SubstanceParameters, Dict]:
    """
    High-level training manager using orchestrated goal-based training.

    Parameters:
    -----------
    config : Dict
        Full configuration with data paths, model architecture, and training params:
        - data_csv_train, data_csv_val: paths to training/validation CSVs
        - vle_data_csv_train, vle_data_csv_val: paths to VLE data (optional)
        - substance_params_csv: path to substance parameters
        - id_column, parameter_columns: column specifications
        - training_preset: one of the preset names (e.g., "p_res_and_p_sat")
        - model_path: path to save model
        - pretrained_model_path: path to pretrained model (optional)
        - resume_from_checkpoint: whether to resume from checkpoint
        - depth, width, activation, dropout_rate, apply_bias: network architecture
        - lr, weight_decay, batch_size, max_epochs, patience: training params

    Returns:
    --------
    model : HEOSNet
        Best trained model
    substance_params : SubstanceParameters
        Substance parameters
    combined_report : Dict
        Training report
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Copy config file + .py source files to model_path at the very start (survives crashes)
    if config_path and os.path.isfile(config_path):
        import shutil, glob as _glob
        _dst = config["model_path"]
        os.makedirs(_dst, exist_ok=True)
        shutil.copy2(config_path, os.path.join(_dst, os.path.basename(config_path)))
        with open(os.path.join(_dst, "config.json"), "w") as _f:
            json.dump(config, _f, indent=2)
        _src_dir = os.path.dirname(os.path.abspath(__file__))
        _code_dst = os.path.join(_dst, "helmholtz_eos_net")
        os.makedirs(_code_dst, exist_ok=True)
        for _py in _glob.glob(os.path.join(_src_dir, "*.py")):
            shutil.copy2(_py, _code_dst)
        print(f"Saved config + source snapshot to: {_dst}")

    # =========================================================================
    # SUBSTANCE PARAMETERS
    # =========================================================================
    print("Initializing substance parameters...")
    substance_params = SubstanceParameters(
        csv_path=config["substance_params_csv"],
        id_column=config.get("id_column", "ID"),
        parameter_columns=config.get("parameter_columns", ["p0_norm", "p1_norm", "p2_norm"]),
        fixed_columns=config.get("fixed_columns", []),
        embedding_dim=len(config.get("parameter_columns", ["p0_norm", "p1_norm", "p2_norm"])),
    )
    substance_params = substance_params.to(device)

    # =========================================================================
    # MODEL
    # =========================================================================
    print("Initializing HEOSNet model...")
    model = HEOSNet(
        substance_params=substance_params,
        depth=config.get("depth", 2),
        width=config.get("width", 1024),
        activation=config.get("activation", "tanh"),
        dropout_rate=config.get("dropout_rate", 0.0),
        transport_net_hidden=config.get("transport_net_hidden", 0),
        transport_net_layers=config.get("transport_net_layers", 2),
        transport_net_p=config.get("transport_net_p", 8),
    )
    model = model.to(device)
    model.transport_rho_n_steps = config.get("transport_rho_n_steps", 20)

    # =========================================================================
    # LOAD PRETRAINED MODEL (if specified)
    # =========================================================================
    pretrained_path = config.get("pretrained_model_path")
    pretrained_loaded = False
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pretrained model from: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        # Drop keys whose shape doesn't match the current model (e.g. substance_params.embeddings
        # when fine-tuning on a smaller substance subset).
        model_state = model.state_dict()
        filtered = {k: v for k, v in state_dict.items()
                    if k in model_state and v.shape == model_state[k].shape}
        skipped = [k for k in state_dict if k not in filtered]
        if skipped:
            print(f"  Pretrained load — skipped (shape mismatch): {skipped}")
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        if missing:
            print(f"  Pretrained load — missing keys: {missing}")
        pretrained_loaded = True

    # =========================================================================
    # FREEZE FLAGS
    # =========================================================================
    if config.get("freeze_a_res_net", False):
        model.simple_net.requires_grad_(False)
        if model.substance_params is not None:
            model.substance_params.requires_grad_(False)
        print("  Frozen: a_res network + substance_params")

    if config.get("freeze_a_res_net_first_layer", False):
        model.freeze_helmholtz_layers(not_frozen_layers=[0])
        if model.substance_params is not None:
            model.substance_params.requires_grad_(False)
        print("  Frozen: a_res network + substance_params")        

    if config.get("freeze_transport_net", False):
        if model.transport_params is not None:
            model.transport_params.requires_grad_(False)
        print("  Frozen: transport_params")

    # print requires_grad_ overview
    print("\n CHECK -- requires_grad_ overview:")
    for n, p in model.named_parameters():
        print("    ",n,p.requires_grad_)
    print("\n")

    # =========================================================================
    # GOAL EVALUATOR
    # =========================================================================
    print(f"Creating GoalEvaluator with preset: {config.get('training_preset', 'unknown')}")
    evaluator = GoalEvaluator(model=model, config=config)

    # =========================================================================
    # RESUME FROM CHECKPOINT (if enabled)
    # =========================================================================
    start_epoch = 0
    optimizer = None
    resume_loaded = False

    if config.get("resume_from_checkpoint", False):
        model_path = config.get("model_path", "")
        checkpoint_path = os.path.join(model_path, "checkpoint.pt")
        if os.path.exists(checkpoint_path):
            print(f"Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr", 1e-4))
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint.get("epoch", 0) + 1
            print(f"Resuming from epoch {start_epoch}")
            resume_loaded = True

    # =========================================================================
    # A_RES PRETRAINING (skipped if any checkpoint was loaded -- or if not speciffically wanted)
    # =========================================================================
    pretrain_report = None
    checkpoint_was_loaded = pretrained_loaded or resume_loaded

    if (config.get("force_pretrain", False) or (config.get("pretrain", False)
            and not checkpoint_was_loaded
            and config.get("training_preset", "all_residuals") != "a_res_only")):

        pretrain_lr = config["lr"] * config.get("pretrain_lr_factor", 10.0)
        pretrain_patience = config.get("patience_pretrain", 10)
        pretrain_max_epochs = config.get("max_epochs_pretrain", 10)

        pretrain_config = {
            **config,
            "training_preset": "a_res_only",
            "lr": pretrain_lr,
            "patience": pretrain_patience,
            "max_epochs": pretrain_max_epochs,
        }
        pretrain_evaluator = GoalEvaluator(model=model, config=pretrain_config)

        print("\n" + "="*70)
        print(f"PRETRAINING PHASE  (a_res_only, lr={pretrain_lr:.2e}, max_epochs={pretrain_max_epochs})")
        print("="*70 + "\n")

        model, _, _, pretrain_report = train_orchestrated(
            model=model,
            evaluator=pretrain_evaluator,
            config=pretrain_config,
            optimizer=None,       # fresh optimizer — discarded after pretraining
            device=device,
            verbose=True,
            epoch_callback=None,  # not reported to Ray, avoids ASHA interference
        )
        # Model weights carry over; main phase starts with a fresh Adam optimizer.

    if config.get("force_pretrain",False):
        model.unfreeze_helmholtz()
        print("Helmholtz NN unfrozen after pretraining because of force_pretrain flag")

    # =========================================================================
    # TRANSPORT PRETRAINING (skipped if any checkpoint was loaded)
    # =========================================================================
    if (config.get("pretrain_transport", False)
            and not resume_loaded
            and model.transport_params is not None):

        weight_decay = config.get("weight_decay", 1e-7)
        tp_preset   = config.get("transport_pretrain_preset", "transport_both")
        tp_lr       = config["lr"] * config.get("transport_pretrain_lr_factor", 10000.0)
        tp_patience = config.get("patience_transport_pretrain", config.get("patience", 10))
        tp_epochs   = config.get("max_epochs_transport_pretrain", 20)

        tp_config = {
            **config,
            "training_preset": tp_preset,
            "lr": tp_lr,
            "patience": tp_patience,
            "max_epochs": tp_epochs,
        }
        tp_evaluator = GoalEvaluator(model=model, config=tp_config)
        tp_optimizer = torch.optim.Adam(
            model.transport_params.parameters(), lr=tp_lr,
            weight_decay=weight_decay,
        )

        model.freeze_helmholtz()
        n_tp = sum(p.numel() for p in model.transport_params.parameters())
        print("\n" + "="*70)
        print("TRANSPORT PRETRAIN")
        print(f"  Preset:     {tp_preset}")
        print(f"  Goals:      {list(tp_evaluator.config.get('target_features', []))}")
        print(f"  LR:         {tp_lr:.2e}  (base × {config.get('transport_pretrain_lr_factor', 10000.0):.0f})")
        print(f"  Max epochs: {tp_epochs}  |  Patience: {tp_patience}")
        print(f"  Params:     {n_tp:,}  (Helmholtz frozen)")
        print("="*70 + "\n")

        model, _, _, _ = train_orchestrated(
            model=model,
            evaluator=tp_evaluator,
            config=tp_config,
            optimizer=tp_optimizer,
            device=device,
            verbose=True,
            epoch_callback=None,
        )
        if not config.get("freeze_a_res_net", False):
            model.unfreeze_helmholtz()

    # =========================================================================
    # TRAINING
    # =========================================================================
    print("\n" + "="*70)
    print("ORCHESTRATED TRAINING")
    print(f"  Preset: {config.get('training_preset', 'unknown')}")
    print(f"  Goals: {evaluator.config.get('target_features', [])}")
    print("="*70 + "\n")

    model, optimizer, best_model_dict, train_report = train_orchestrated(
        model=model,
        evaluator=evaluator,
        config=config,
        optimizer=optimizer,
        device=device,
        verbose=True,
    )

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    model_path = config.get("model_path", "trained_models/orchestrated/")
    os.makedirs(model_path, exist_ok=True)

    # Save checkpoint
    checkpoint_path = os.path.join(model_path, "checkpoint.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "best_val_loss": train_report["best_val_loss"],
        "epoch": train_report["best_epoch"],
        "config": config,
    }, checkpoint_path)
    print(f"Saved checkpoint to: {checkpoint_path}")

    # Save training history
    history_path = os.path.join(model_path, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(train_report, f, indent=2)

    print(f"Saved training history to: {history_path}")

    # =========================================================================
    # COMBINED REPORT
    # =========================================================================
    combined_report = {
        "best_val_loss": train_report["best_val_loss"],
        "training_report": train_report,
        "pretrain_report": pretrain_report,
    }

    return model, substance_params, combined_report


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    """Command-line interface for orchestrated training."""
    import argparse

    parser = argparse.ArgumentParser(description="Orchestrated HEOSNet Training")
    parser.add_argument("config", type=str, help="Path to config JSON file")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = json.load(f)

    # Optional thread limiting — set before PyTorch initializes its thread pool
    num_threads = config.get("num_threads")
    if num_threads:
        import os
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        os.environ["MKL_NUM_THREADS"] = str(num_threads)
        torch.set_num_threads(num_threads)
        print(f"Thread limit: {num_threads}")

    # Run training
    model, substance_params, report = train_manager_orchestrated(config, config_path=args.config)

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print(f"  Best validation loss: {report['best_val_loss']:.6f}")
    print("="*70)


if __name__ == "__main__":
    main()
