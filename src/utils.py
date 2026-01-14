import yaml
from pathlib import Path


def load_config(config_path: str = "config.yaml") -> dict:
    # This function loads configuration values from config.yaml
    # Using a config file avoids hardcoding paths and column names in code
    # Get project root directory
    project_root = Path(__file__).resolve().parent.parent

    # Build correct config path
    config_path = project_root / "config" / "config.yaml"
    config_file = Path(config_path)

    # If config file does not exist, stop execution early
    # This prevents silent failures later in the pipeline
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    # Open and safely read YAML configuration into a Python dictionary
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Return config so it can be reused across scripts
    return config
