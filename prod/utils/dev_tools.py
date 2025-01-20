import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import os
import subprocess
from pathlib import Path


def load_config():
    if "JETPOINTNET_CONFIG_FILE" in os.environ:
        config_path = Path(os.environ["JETPOINTNET_CONFIG_FILE"])
        print(f"Detected manual config file location, using {config_path}")
    else:
        user = Path.home().name
        config_path = Path(__file__).parent.parent / "configs" / f"{user}_config.toml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, "rb") as file:
        config = tomllib.load(file)

    (git_hash, is_dirty) = get_git_status()
    config["run_info"] = {"git_hash": git_hash, "is_dirty": is_dirty}
    return config


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
