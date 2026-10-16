"""Regressions against the installed Hermes CLI and execution classifier."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PLUGIN = Path(__file__).resolve().parents[1] / "integrations" / "hermes"
sys.path.insert(0, str(PLUGIN))


@unittest.skipUnless(importlib.util.find_spec("agent"), "requires actual Hermes host")
class HostCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"])
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        env = patch.dict(os.environ, {"HERMES_HOME": str(self.home)})
        env.start()
        self.addCleanup(env.stop)
        from gluomem_hermes.provider import GluoMemProvider

        self.provider = GluoMemProvider()
        self.provider.save_config(
            {
                "endpoint": "http://127.0.0.1:9",
                "automatic_capture": False,
                "outbox_max_events": 17,
                "outbox_max_bytes": 2048,
            },
            self.home,
        )

    def test_generic_memory_status_dispatch_uses_canonical_nonsecret_config(self):
        from hermes_cli.config import save_config

        shutil.copytree(PLUGIN / "gluomem_hermes", self.home / "plugins" / "gluomem")
        # The host normally passes {}, but stale host-section values must not
        # override the file used by initialize(), capture toggles and admission.
        for section in (
            {},
            {
                "endpoint": "https://stale.invalid",
                "automatic_capture": True,
                "api_key": "test-secret-must-not-print",
            },
        ):
            with self.subTest(section=bool(section)):
                save_config({"memory": {"provider": "gluomem", "gluomem": section}})
                result = subprocess.run(
                    [sys.executable, "-m", "hermes_cli.main", "memory", "status"],
                    env=dict(
                        os.environ,
                        PYTHONPATH=str(PLUGIN),
                        GLUOMEM_SERVICE_KEY="isolated-test-key",
                        PYTHONIOENCODING="utf-8",
                    ),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    timeout=60,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertNotIn("status_config_error", result.stdout)
                self.assertIn("endpoint: http://127.0.0.1:9", result.stdout)
                self.assertIn("automatic_capture: False", result.stdout)
                self.assertIn("outbox_max_events: 17", result.stdout)
                self.assertIn("outbox_max_bytes: 2048", result.stdout)
                self.assertNotIn("stale.invalid", result.stdout)
                self.assertNotIn("test-secret-must-not-print", result.stdout)
                self.assertNotIn("isolated-test-key", result.stdout)
                self.assertNotIn("NOT installed", result.stdout)
        self.assertEqual(self.provider.get_status_config()["outbox_max_events"], 17)

    def test_status_tool_is_success_with_truthful_persisted_delivery_data(self):
        # This is the exact classifier imported by agent.tool_executor, not the
        # unrelated file-mutation helpers in tool_result_classification.py.
        from agent.memory_manager import MemoryManager
        from agent.tool_executor import _detect_tool_failure

        p = self.provider
        p.initialize("status-test", hermes_home=self.home, platform="cli")
        p.shutdown()  # Stop delivery before deterministic offline DB fixtures.
        manager = MemoryManager()
        manager.add_provider(p)
        for has_failures in (False, True):
            with self.subTest(has_failures=has_failures):
                if has_failures:
                    with p._db() as db:
                        db.execute(
                            "INSERT INTO outbox(event_id,envelope,last_error,attempts) "
                            "VALUES (?,?,?,?)",
                            (
                                "pending-test",
                                '{"content":"synthetic observation"}',
                                "service HTTP 503",
                                2,
                            ),
                        )
                        db.execute(
                            "INSERT INTO dead_letters VALUES (?,?,?,?)",
                            ("rejected-test", 422, 3, 1.0),
                        )
                p.save_config(
                    {
                        "endpoint": "http://127.0.0.1:9",
                        "automatic_capture": has_failures,
                        "outbox_max_events": 17,
                        "outbox_max_bytes": 2048,
                    },
                    self.home,
                )
                original = p.status()
                raw = manager.handle_tool_call("gluomem_status", {})
                self.assertIsInstance(raw, str)
                self.assertFalse(_detect_tool_failure("gluomem_status", raw)[0], raw)
                data = json.loads(raw)
                self.assertIs(data.pop("success"), True)
                # Tool-safe names avoid the host's raw-string error-key heuristic;
                # the count and every receipt remain present and truthful.
                data["failed"] = data.pop("rejected_events")
                for issue in data["delivery_errors"]:
                    issue["error"] = issue.pop("reason")
                self.assertEqual(data, original)
                self.assertEqual(data["failed"], int(has_failures))
                self.assertEqual(data["pending"], int(has_failures))
                self.assertIs(data["automatic_capture"], has_failures)
                self.assertTrue(data["primary_context"])
                self.assertNotIn("synthetic observation", raw)
                self.assertNotIn("endpoint", data)

    def test_actual_manager_failures_still_classify_as_errors(self):
        from agent.memory_manager import MemoryManager
        from agent.tool_executor import _detect_tool_failure

        manager = MemoryManager()
        manager.add_provider(self.provider)
        with patch.object(self.provider, "status", side_effect=OSError("test storage unavailable")):
            results = [manager.handle_tool_call("gluomem_status", {})]
        results.append(manager.handle_tool_call("gluomem_status", {"unexpected": True}))
        for raw in results:
            with self.subTest(raw=raw):
                data = json.loads(raw)
                self.assertFalse(data.get("success", False))
                self.assertIn("error", data)
                self.assertTrue(_detect_tool_failure("gluomem_status", raw)[0])


if __name__ == "__main__":
    unittest.main()
