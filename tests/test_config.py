from pathlib import Path

from pitwall.config import load_settings

CONF_DIR = Path(__file__).resolve().parents[1] / "conf"


def test_base_config_loads_and_env_overrides(monkeypatch):
    monkeypatch.setenv("PITWALL_LAKEHOUSE_ROOT", "s3a://bucket/pitwall")
    settings = load_settings(env="does-not-exist", conf_dir=CONF_DIR)
    assert settings.lakehouse_root == "s3a://bucket/pitwall"
    assert settings.seasons.as_list()[0] == 2018
