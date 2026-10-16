"""Personal memory must not enter an unscoped persistent model cache by default."""

from gluomem.config.app_config import DspyCacheConfig


def test_model_disk_cache_is_opt_in() -> None:
    assert DspyCacheConfig().disk is False


def test_disabled_model_cache_does_not_retain_memory_results(tmp_path, monkeypatch) -> None:
    import dspy

    from gluomem.config.app_config import AppConfig
    from gluomem.llm import router

    cfg = AppConfig().with_overrides({"dspy.cache.dir": str(tmp_path / "model-cache")})
    monkeypatch.setattr(router, "_cache_configured", False)
    router.LMRouter(cfg)
    assert dspy.cache.enable_disk_cache is False
    assert dspy.cache.enable_memory_cache is False
