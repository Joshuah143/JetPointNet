import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from pathlib import Path
import os
import subprocess


def load_config():
    # allow a config file to be loaded from an env var
    if "CONFIG_FILE" in os.environ:
        config_path = Path(os.environ["CONFIG_FILE"])
        print(f"Detected manual config file location, using {config_path}")
    else:

        user = Path.home().name
        # repo/prod/configs/config.toml
        config_path = Path(__file__).parent.parent / "configs" / f"{user}_config.toml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, "rb") as file:
        print(f"Loading config from {config_path}")
        config = tomllib.load(file)
        print(f'Welcome {config["global_params"]["user_name"]}')

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
