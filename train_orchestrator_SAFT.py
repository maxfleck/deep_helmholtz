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

cfg_file = "config_SAFT2.json"

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
