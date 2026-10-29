import torch
import numpy as np
import json
import matplotlib.pyplot as plt

from helmholtz_eos_net.model import HEOSNet
from helmholtz_eos_net.substance_parameters import SubstanceParameters
from helmholtz_eos_net.training_orchestrator import GoalEvaluator, PRESETS, GOAL_REGISTRY
from helmholtz_eos_net.train_orchestrated import train_orchestrated, train_manager_orchestrated

print("Imports successful!")
print(f"Available presets: {list(PRESETS.keys())}")

cfg_file = "config_transport_joint_water.json"
cfg_file = "config_transport_joint_water_res.json"

# Load config_fine.json
with open(cfg_file, "r") as f:
    config = json.load(f)

print("Config loaded:")
for key, value in config.items():
    if not key.startswith("_"):
        print(f"  {key}: {value}")

# Uncomment to run full training
model, substance_params, report2 = train_manager_orchestrated(config, config_path=cfg_file)
print(f"Best val loss: {report2['best_val_loss']:.6f}")

train_report = report2["training_report"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Total loss
ax = axes[0]
if 'total' in train_report['train_losses']:
    ax.plot(train_report['train_losses']['total'], label='Train')
if 'total' in train_report['val_losses']:
    ax.plot(train_report['val_losses']['total'], label='Val')
ax.set_xlabel('Epoch')
ax.set_ylabel('Total Loss')
ax.set_title('Total Loss')
ax.legend()
ax.set_yscale('log')

# Per-goal losses
ax = axes[1]
for key in train_report['val_losses']:
    if key != 'total':
        ax.plot(train_report['val_losses'][key], label=key)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Per-Goal Val Losses')
ax.legend()
ax.set_yscale('log')

plt.tight_layout()
plt.show()
