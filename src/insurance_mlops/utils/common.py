"""Common helper utilities for the Insurance Prediction MLOps project."""

from pathlib import Path
from typing import Any, Dict, List, Union
import yaml

from insurance_mlops.utils.logger import get_logger

logger = get_logger(__name__)


def get_project_root() -> Path:
    """Returns the project root directory by searching for markers like pyproject.toml."""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists() or (parent / "config").exists():
            return parent
    return Path.cwd()


def load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Safely loads a YAML configuration file.

    Args:
        file_path: Relative or absolute path to the YAML file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty or corrupted.
    """
    path = Path(file_path)
    if not path.is_absolute():
        path = get_project_root() / path

    if not path.exists():
        msg = f"Configuration file not found at: {path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            if config is None:
                raise ValueError(f"YAML file is empty: {path}")
            logger.info(f"Loaded configuration successfully from {path.name}")
            return config
    except yaml.YAMLError as exc:
        msg = f"Error parsing YAML file {path}: {exc}"
        logger.error(msg)
        raise ValueError(msg) from exc


def save_yaml(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """Safely writes a dictionary to a YAML file.

    Args:
        data: Dictionary to serialize.
        file_path: Path where the YAML file should be written.
    """
    path = Path(file_path)
    if not path.is_absolute():
        path = get_project_root() / path

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Saved configuration successfully to {path}")


def create_directories(paths: List[Union[str, Path]]) -> None:
    """Creates a list of directories if they do not already exist.

    Args:
        paths: List of directory paths to create.
    """
    for p in paths:
        path = Path(p)
        if not path.is_absolute():
            path = get_project_root() / path
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {path}")
