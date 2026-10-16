"""The directory installer must run without a Hermes runtime dependency."""

import os
import subprocess
import sys
from pathlib import Path


def test_native_plugin_installs_without_importing_hermes(tmp_path):
    plugin = Path(__file__).resolve().parents[1] / "integrations" / "hermes"
    script = """
import importlib.abc
import runpy
import sys

class NoHostImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = {'agent', 'hermes_constants', 'hermes_cli', 'plugins', 'utils'}
        if fullname.split('.')[0] in blocked:
            raise ModuleNotFoundError('Installer must not require Hermes: ' + fullname)

sys.meta_path.insert(0, NoHostImports())
sys.argv = ['gluomem-hermes-install', '--hermes-home', sys.argv[1]]
runpy.run_module('gluomem_hermes.install', run_name='__main__')
assert 'gluomem_hermes.provider' not in sys.modules
assert 'dspy' not in sys.modules
"""
    run = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(plugin)},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    target = tmp_path / "plugins" / "gluomem"
    for name in ("__init__.py", "provider.py", "cli.py", "plugin.yaml", "README.md"):
        assert (target / name).is_file(), name
