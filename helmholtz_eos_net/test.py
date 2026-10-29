import os
import json
import torch
import torch._utils  # force-load before torch.load (PyTorch 2.10+ lazy-loading regression)
from typing import Tuple, Dict

from helmholtz_eos_net.model import HEOSNet
from helmholtz_eos_net.substance_parameters import SubstanceParameters
from helmholtz_eos_net.utils import fix_paths


def load_model_from_folder(
    model_folder: str,
    device: torch.device = None
) -> Tuple[HEOSNet, Dict]:
    """
    Load trained model and config from model folder.

    Args:
        model_folder: Path to folder containing checkpoint.pt and config.json
        device: torch device (default: auto-detect GPU if available)

    Returns:
        model: Trained HEOSNet instance
        config: Configuration dict
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint_path = os.path.join(model_folder, "checkpoint.pt")
    config_path = os.path.join(model_folder, "config.json")

    if not os.path.isfile(checkpoint_path):
        checkpoint_path = os.path.join(model_folder, "best_checkpoint.pt")
    if not os.path.isfile(config_path):
        config_path = os.path.join(model_folder, "training_config.json")

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    


    if not os.path.isfile(config_path):
        print(f"Config not found: {config_path} try params.json")
        config_path = os.path.join(model_folder, "params.json")
        if os.path.isfile(config_path):
            # Load config
            with open(config_path, 'r') as f:
                dummy = json.load(f)
            config = dummy["base_config"]
            for key in dummy.keys():
                if key != "base_config":
                    config[key] = dummy[key]                
        else:
            raise FileNotFoundError(f"Config not found: {config_path}")
    else:
        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)

    # Fix any stale absolute paths and save canonical config.json
    config = fix_paths(config, os.getcwd())
    save_path = os.path.join(model_folder, "config.json")
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved fixed config to: {save_path}")

    print(f"Loaded config from: {config_path}")
    print(f"  Scenario: {config.get('training_scenario', 'unknown')}")

    # Load checkpoint early to extract embedded architecture config
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    # Architecture keys must match the saved weights — override from checkpoint config.
    # Handles stale config.json that predates transport_net_hidden being added.
    # try/except keeps old checkpoints (no "config" key) working unchanged.
    _arch_keys = [
        "depth", "width", "activation", "dropout_rate",
        "transport_net_hidden", "transport_net_layers", "transport_net_p",
    ]
    try:
        ckpt_config = checkpoint.get("config", {})
        for k in _arch_keys:
            if k in ckpt_config:
                config[k] = ckpt_config[k]
        if ckpt_config:
            print("  Architecture config loaded from checkpoint.")
    except Exception:
        pass  # old checkpoint without embedded config — use config.json as-is

    # Create substance parameters
    substance_params = SubstanceParameters(
        csv_path=config["substance_params_csv"],
        id_column=config["id_column"],
        parameter_columns=config["parameter_columns"],
        # fixed_columns=config["fixed_columns"],
    )

    # Create model with architecture parameters from config
    model = HEOSNet(
        substance_params=substance_params,
        depth=config["depth"],
        width=config["width"],
        activation=config["activation"],
        dropout_rate=config["dropout_rate"],
        transport_net_hidden=int(config.get("transport_net_hidden", 0)),  # type: ignore[union-attr]
        transport_net_layers=int(config.get("transport_net_layers", 2)),  # type: ignore[union-attr]
        transport_net_p=int(config.get("transport_net_p", 8)),            # type: ignore[union-attr]
    )

    # Load state dict (checkpoint was already loaded above for config extraction)
    state = checkpoint["model_state_dict"]
    m_state = model.state_dict()
    filtered = {k: v for k, v in state.items() if k in m_state and v.shape == m_state[k].shape}
    model.load_state_dict(filtered, strict=False)
    model = model.to(device)
    model.eval()

    print(f"Loaded checkpoint from: {checkpoint_path}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, config
