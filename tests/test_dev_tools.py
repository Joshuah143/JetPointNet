import pytest
import prod.utils.dev_tools as dev_tools


def test_lead_config():
    config = dev_tools.load_config()
    assert config["global_params"]["user_name"]


def test_git_status():
    (git_hash, is_dirty) = dev_tools.get_git_status()
    assert git_hash
    assert type(is_dirty) == bool
