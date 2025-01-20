import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import os
import subprocess
from pathlib import Path


class _SingletonConfig:
    _instance = None
    _config = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            # Create the one-and-only instance
            cls._instance = super().__new__(cls)
            # Initialize the config just once
            cls._instance._init_config()
        return cls._instance

    def _init_config(self):
        """Loads the config and prints exactly once."""
        if "JETPOINTNET_CONFIG_FILE" in os.environ:
            config_path = Path(os.environ["JETPOINTNET_CONFIG_FILE"])
            print(f"Detected manual config file location, using {config_path}")
        else:
            user = Path.home().name
            config_path = (
                Path(__file__).parent.parent / "configs" / f"{user}_config.toml"
            )

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")

        print(f"Loading config from {config_path}")
        with open(config_path, "rb") as file:
            config = tomllib.load(file)
        print(f'Welcome {config["global_params"]["user_name"]}')

        self._config = config

    @property
    def config(self):
        """Access the loaded config."""
        return self._config


def load_config():
    loader = _SingletonConfig()
    return loader.config


def get_git_status():
    """
    Returns the git hash of the last commit and the repository's dirty state.

    Returns:
        tuple: (commit_hash, is_dirty)
               commit_hash (str): Hash of the last commit.
               is_dirty (bool): True if the repository has uncommitted changes, False otherwise.
    """
    try:
        commit_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .strip()
            .decode("utf-8")
        )

        dirty_state = (
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
            )
            .strip()
            .decode("utf-8")
        )

        is_dirty = bool(dirty_state)

        return commit_hash, is_dirty

    except subprocess.CalledProcessError:
        raise RuntimeError(
            "Unable to retrieve Git information. Ensure this is a Git repository."
        )
