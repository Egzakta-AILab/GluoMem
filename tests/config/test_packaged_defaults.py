"""An installed wheel has no ``conf/config.yaml``; its field defaults must match."""

from __future__ import annotations

from pathlib import Path

from gluomem.config.app_config import DEFAULT_CONFIG_PATH, load_config


def test_field_defaults_equal_conf_config_yaml(tmp_path: Path) -> None:
    assert DEFAULT_CONFIG_PATH.is_file()
    from_yaml = load_config().model_dump()
    without_yaml = load_config(path=tmp_path / "absent.yaml").model_dump()
    assert from_yaml == without_yaml
