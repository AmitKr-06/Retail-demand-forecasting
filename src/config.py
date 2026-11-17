# paths, constants, load params.yaml

import yaml
from src.constant import PARAMS_FILE

def load_params(path=PARAMS_FILE) -> dict:
    """Reads params.yaml and returns it as a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

PARAMS = load_params()