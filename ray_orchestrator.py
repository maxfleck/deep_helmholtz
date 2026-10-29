import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from helmholtz_eos_net.ray_wrapper import run_hyperparameter_tuning

print("Imports successful!")

# =============================================================================
# TUNING CONFIGURATION — edit these
# =============================================================================

CONFIG_PATH      = "config_ray_full00.json"   # base config with data paths
NUM_SAMPLES      = 4                  # number of trials
MAX_EPOCHS       = 1000                # max epochs per trial
GPUS_PER_TRIAL   = 0.0                 # 0.0 = CPU only; 1.0 = one GPU per trial
CPUS_PER_TRIAL   = 16                   # CPU cores per trial
RESULTS_DIR      = "ray_results"       # output directory for Ray results
EXPERIMENT_NAME  = "heos_finxxx"        # subfolder name for this run
GRACE_PERIOD     = 50                 # min epochs before ASHA can prune a trial

# --- preview base config ---
with open(CONFIG_PATH) as f:
    base_config = json.load(f)

print(f"Base config: {CONFIG_PATH}")
print(f"  Data train : {base_config.get('data_csv_train')}")
print(f"  Data val   : {base_config.get('data_csv_val')}")
print(f"  Substances : {base_config.get('substance_params_csv')}")
print(f"  Preset     : {base_config.get('training_preset')}")
print()
print(f"Tuning settings:")
print(f"  Trials     : {NUM_SAMPLES}")
print(f"  Max epochs : {MAX_EPOCHS}  (grace period: {GRACE_PERIOD})")
print(f"  Resources  : {GPUS_PER_TRIAL} GPU, {CPUS_PER_TRIAL} CPU per trial")
print(f"  Results    : {RESULTS_DIR}/{EXPERIMENT_NAME}")

results = run_hyperparameter_tuning(
    config_path=CONFIG_PATH,
    num_samples=NUM_SAMPLES,
    max_epochs=MAX_EPOCHS,
    gpus_per_trial=GPUS_PER_TRIAL,
    cpus_per_trial=CPUS_PER_TRIAL,
    results_dir=RESULTS_DIR,
    experiment_name=EXPERIMENT_NAME,
    grace_period=GRACE_PERIOD,
)

# Best trial summary
best = results.get_best_result(metric="val_loss", mode="min")

print("=" * 60)
print("BEST TRIAL")
print("=" * 60)
print(f"  val_loss : {best.metrics['val_loss']:.6f}")
print()
print("  Hyperparameters:")
tuned_params = ["depth", "width", "activation", "dropout_rate", "lr", "weight_decay", "batch_size"]
for key in tuned_params:
    val = best.config.get(key)
    if val is not None:
        print(f"    {key:20s}: {val}")

best_config_path = Path(RESULTS_DIR) / EXPERIMENT_NAME / "best_config.json"
print()
print(f"  Saved to: {best_config_path}")


# Build results DataFrame
df = results.get_dataframe(filter_metric="val_loss", filter_mode="min")
df = df.replace([np.inf, -np.inf], np.nan, inplace=True)

df = df.rename(columns={col: col.split("/", 1)[1] for col in df.columns if col.startswith("config/")})

# Keep only completed/useful rows
df = df.dropna(subset=["val_loss"])
best_idx = df["val_loss"].idxmin()

print(f"Completed trials: {len(df)}")
print(df[["val_loss"] + tuned_params].sort_values("val_loss").head(10).to_string(index=False))

# Visualize results
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("Hyperparameter Tuning Results", fontsize=14, fontweight="bold")

plot_params = ["lr", "weight_decay", "width", "depth", "dropout_rate", "batch_size"]
log_scale   = {"lr", "weight_decay"}

for ax, param in zip(axes.flatten(), plot_params):
    if param not in df.columns:
        ax.set_visible(False)
        continue

    x = df[param]
    y = df["val_loss"]

    ax.scatter(x, y, alpha=0.6, s=40, label="trials")
    # highlight best
    ax.scatter(df.loc[best_idx, param], df.loc[best_idx, "val_loss"],
               color="red", s=120, zorder=5, label="best")

    if param in log_scale:
        ax.set_xscale("log")

    ax.set_xlabel(param)
    ax.set_ylabel("val_loss")
    ax.set_title(param)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(Path(RESULTS_DIR) / EXPERIMENT_NAME / "tuning_overview.png", dpi=150, bbox_inches="tight")
plt.show()

# Val-loss distribution
fig2, ax2 = plt.subplots(figsize=(7, 4))
ax2.hist(df["val_loss"].clip(upper=df["val_loss"].quantile(0.95)), bins=20, edgecolor="black")
ax2.axvline(df.loc[best_idx, "val_loss"], color="red", linestyle="--", label=f'best = {df.loc[best_idx, "val_loss"]:.4f}')
ax2.set_xlabel("val_loss")
ax2.set_ylabel("count")
ax2.set_title("Validation Loss Distribution Across Trials")
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

best_config_path = Path(RESULTS_DIR) / EXPERIMENT_NAME / "best_config.json"

with open(best_config_path) as f:
    best_config = json.load(f)

print(f"Best configuration ({best_config_path}):")
print(json.dumps({k: v for k, v in best_config.items() if not k.startswith("_")}, indent=2))

print()
print("To train with these hyperparameters, open train_model_new.ipynb and set:")
print(f'  config_path = "{best_config_path}"')
