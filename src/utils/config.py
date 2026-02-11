import yaml
from pathlib import Path


def load_config(config_path: str = "configs/base_config.yaml") -> dict:
    """Load YAML configuration file and resolve paths relative to project root."""
    config_path = Path(config_path)
    if not config_path.is_absolute():
        # Resolve relative to project root (two levels up from this file)
        project_root = Path(__file__).resolve().parent.parent.parent
        config_path = project_root / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    # Resolve data/output dirs relative to project root
    project_root = config_path.parent.parent
    config["data_dir"] = str(project_root / config["data_dir"])
    config["output_dir"] = str(project_root / config["output_dir"])
    return config
