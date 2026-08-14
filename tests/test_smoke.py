"""Smoke test — proves the gate runs and the gluomem package imports."""


def test_imports() -> None:
    import gluomem

    assert isinstance(gluomem.__version__, str)
