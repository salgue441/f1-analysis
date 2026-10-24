from pathlib import Path

from pitwall.config import load_settings

CONF_DIR = Path(__file__).resolve().parents[1] / "conf"


def test_base_config_loads_and_env_overrides(monkeypatch):
    monkeypatch.setenv("PITWALL_LAKEHOUSE_ROOT", "s3a://bucket/pitwall")
    settings = load_settings(env="does-not-exist", conf_dir=CONF_DIR)
    assert settings.lakehouse_root == "s3a://bucket/pitwall"
    assert settings.seasons.as_list()[0] == 2018


def test_cloud_profiles_only_change_config():
    """SAD §8.2: moving to object storage or a managed platform is a config change."""
    local = load_settings(env="local", conf_dir=CONF_DIR)
    s3 = load_settings(env="s3local", conf_dir=CONF_DIR)
    assert s3.lakehouse_root.startswith("s3a://")
    assert any(p.startswith("org.apache.hadoop:hadoop-aws:") for p in s3.spark.packages)
    assert s3.spark.master == local.spark.master and s3.spark.delta_pip

    for env in ("aws", "databricks"):
        managed = load_settings(env=env, conf_dir=CONF_DIR)
        assert managed.lakehouse_root.startswith("s3://")
        assert managed.spark.master == ""  # the platform's own master
        assert not managed.spark.delta_pip  # the platform's own Delta
        assert managed.seasons == local.seasons and managed.openf1 == local.openf1
