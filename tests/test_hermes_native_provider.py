"""Native plugin tests; run with Hermes' Python (no GluoMem ML dependencies)."""

import importlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_hermes_outbox_backpressure import job_ack

PLUGIN = Path(__file__).resolve().parents[1] / "integrations" / "hermes"
sys.path.insert(0, str(PLUGIN))


@unittest.skipUnless(
    importlib.util.find_spec("agent"),
    "requires an actual Hermes host interpreter; not a GluoMem core test",
)
class NativeProviderTests(unittest.TestCase):
    def provider(self):
        return importlib.import_module("gluomem_hermes").GluoMemProvider()

    def test_durable_offline_observation_replays_after_restart_with_same_scope(self):
        import sqlite3
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        received = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                received.append(
                    (
                        dict(self.headers),
                        json.loads(self.rfile.read(int(self.headers["Content-Length"]))),
                    )
                )
                scope = {
                    field: self.headers[header]
                    for field, header in (
                        ("tenant_id", "X-GluoMem-Tenant"),
                        ("principal_id", "X-GluoMem-Principal"),
                        ("namespace", "X-GluoMem-Namespace"),
                    )
                }
                body = json.dumps(job_ack(scope, received[-1][1])).encode()
                self.send_response(202)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except OSError:
                    pass  # Expected: offline request timed out before reconnect.

            def log_message(self, format: str, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        endpoint = "http://127.0.0.1:" + str(server.server_port)
        with (
            tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as home,
            patch("agent.secret_scope.get_secret", return_value="test-key"),
        ):
            p = self.provider()
            p.save_config({"endpoint": endpoint}, home)
            p.initialize("session-a", hermes_home=home, user_id="alice", cwd="C:/work/project")
            result = json.loads(
                p.handle_tool_call("gluomem_remember", {"content": "inferred preference"})
            )
            self.assertEqual(result["status"], "queued_local")
            from contextlib import closing

            with closing(sqlite3.connect(Path(home) / "gluomem" / "outbox.sqlite3")) as db:
                row = db.execute("SELECT envelope FROM outbox").fetchone()
            envelope = json.loads(row[0])
            self.assertEqual(envelope["schema_version"], 1)
            self.assertNotIn("test-key", row[0])
            self.assertEqual(envelope["payload"]["revision"], 1)
            self.assertNotEqual(envelope["payload"]["source_type"], "explicit_memory")
            self.assertTrue(envelope["payload"]["require_review"])
            p.shutdown()
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            q = self.provider()
            q.initialize("session-b", hermes_home=home, user_id="bob", cwd="C:/work/other")
            self.assertTrue(q.flush(timeout=4))
            self.assertEqual(received[-1][1], envelope["payload"])
            self.assertEqual(
                received[-1][0]["X-GluoMem-Principal"], envelope["scope"]["principal_id"]
            )
            self.assertEqual(q.status()["pending"], 0)
            q.shutdown()
        server.shutdown()
        server.server_close()

    def test_capture_is_opt_in_verbatim_human_only_and_session_scoped(self):
        with (
            tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as home,
            patch("agent.secret_scope.get_secret", return_value="test-key"),
        ):
            p = self.provider()
            p.save_config({"endpoint": "http://127.0.0.1:9"}, home)
            p.initialize(
                "first",
                hermes_home=home,
                user_id="alice",
                cwd="project-a",
                agent_workspace="hermes",
            )
            p.sync_turn(
                "human raw", "SECRET REASONING", messages=[{"role": "system", "content": "hidden"}]
            )
            self.assertEqual(p.status()["pending"], 0)
            p.shutdown()
            p = self.provider()
            p.save_config({"endpoint": "http://127.0.0.1:9", "automatic_capture": True}, home)
            p.initialize(
                "first",
                hermes_home=home,
                user_id="alice",
                cwd="project-a",
                agent_workspace="hermes",
            )
            p.sync_turn(
                "human raw",
                "SECRET REASONING",
                session_id="first",
                turn_author={"id": "bob", "is_bot": False},
                messages=[{"role": "system", "content": "hidden"}],
            )
            p.sync_turn("bot raw", "response", turn_author={"id": "bot", "is_bot": True})
            p.sync_turn(
                "human raw",
                "SECRET REASONING",
                session_id="first",
                turn_author={"id": "bob", "is_bot": False},
            )
            self.assertEqual(p.status()["pending"], 1)
            with p._db() as db:
                raw = db.execute("SELECT envelope FROM outbox").fetchone()[0]
            envelope = json.loads(raw)
            self.assertNotIn("hidden", raw)
            self.assertNotIn("SECRET", raw)
            self.assertEqual(envelope["payload"]["content"], "human raw")
            self.assertEqual(envelope["scope"], p._scope("bob"))
            self.assertNotEqual(p._scope("alice")["principal_id"], p._scope("bob")["principal_id"])
            p.on_session_switch("second")
            p.sync_turn("human raw", "response")
            self.assertEqual(p.status()["pending"], 2)
            p.shutdown()
            q = self.provider()
            q.initialize("cron", hermes_home=home, agent_context="cron")
            q.sync_turn("scheduled content", "response")
            self.assertEqual(q.status()["pending"], 2)
            with self.assertRaises(ValueError):
                q.handle_tool_call("gluomem_remember", {"content": "blocked"})
            q.shutdown()

    def test_recall_is_bounded_fails_open_and_never_reuses_stale_results(self):
        import threading
        import time

        module = importlib.import_module("gluomem_hermes.provider")
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as home:
            p = self.provider()
            p.save_config({"endpoint": "http://127.0.0.1:9"}, home)
            p.initialize(
                "first",
                hermes_home=home,
                user_id="alice",
                cwd="project-a",
                agent_workspace="hermes",
            )
            self.assertEqual(
                p._scope()["namespace"], "workspace-" + module._digest("project-a")[:40]
            )
            with patch.object(
                module,
                "_request",
                return_value={"hits": [{"frame": {"id": 7, "content": "важно " * 4000}}]},
            ):
                result = p.prefetch("remember project details", session_id="first")
                self.assertLessEqual(len(result.encode("utf-8")), 1500)
                self.assertIn("важно", result)
                self.assertEqual(p.recall_status().count, 1)
            with patch.object(module, "_request", side_effect=OSError("offline")):
                self.assertEqual(p.prefetch("different query"), "")
                self.assertIsNone(p.recall_status())
            release = threading.Event()
            with patch.object(
                module, "_request", side_effect=lambda *a, **k: release.wait(4) or {}
            ):
                start = time.monotonic()
                self.assertEqual(p.prefetch("slow query"), "")
                self.assertLess(time.monotonic() - start, 2.0)
                for _ in range(30):
                    self.assertEqual(p.prefetch("other slow query"), "")
                self.assertLessEqual(
                    sum(t.name == "gluomem-recall" for t in threading.enumerate()), 1
                )
                release.set()
            p.shutdown()

    def test_install_native_loader_and_memory_manager_in_isolated_profile(self):
        import subprocess

        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as home:
            env = dict(
                os.environ,
                HERMES_HOME=home,
                GLUOMEM_SERVICE_KEY="isolated-test-key",
                PYTHONPATH=str(PLUGIN),
            )
            install = subprocess.run(
                [sys.executable, "-m", "gluomem_hermes.install", "--hermes-home", home],
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(install.returncode, 0, install.stdout + install.stderr)
            self.assertTrue((Path(home) / "plugins" / "gluomem" / "plugin.yaml").is_file())
            code = """
import json
from hermes_constants import get_hermes_home
from plugins.memory import load_memory_provider, list_memory_provider_names
from agent.memory_manager import MemoryManager
assert 'gluomem' in list_memory_provider_names()
p = load_memory_provider('gluomem', register_skills=False)
assert p is not None
p.save_config({'endpoint': 'http://127.0.0.1:9', 'automatic_capture': True}, str(get_hermes_home()))
assert p.is_available()
m = MemoryManager(external_prefetch_timeout=2)
m.add_provider(p)
m.initialize_all('native-test', platform='cli', user_id='isolated-user', cwd='isolated-workspace')
assert m.has_tool('gluomem_remember')
result = json.loads(m.handle_tool_call('gluomem_remember', {'content': 'candidate only'}))
assert result['status'] == 'queued_local'
m.sync_all('raw user evidence', 'do not capture assistant', session_id='native-test')
assert m.flush_pending(timeout=3)
assert json.loads(m.handle_tool_call('gluomem_status', {}))['pending'] == 2
assert not m.supports_pre_compress_checkpoint()
m.shutdown_all()
assert m.shutdown_drain_state['status'] == 'drained'
print('REAL_HOST_LOADER_AND_MANAGER_OK')
"""
            result = subprocess.run(
                [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=30
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("REAL_HOST_LOADER_AND_MANAGER_OK", result.stdout)

    def test_marked_compression_summary_is_not_automatic_evidence(self):
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as home:
            p = self.provider()
            p.save_config({"endpoint": "http://127.0.0.1:9", "automatic_capture": True}, home)
            p.initialize("first", hermes_home=home)
            try:
                p.sync_turn(
                    "derived summary",
                    "reply",
                    messages=[
                        {"role": "user", "content": "derived summary", "_compressed_summary": True}
                    ],
                )
                self.assertEqual(p.status()["pending"], 0)
            finally:
                p.shutdown()

    def test_turn_author_and_background_context_are_pinned(self):
        import contextvars

        module = importlib.import_module("gluomem_hermes.provider")
        marker = contextvars.ContextVar("profile-test-marker", default="WRONG")
        observed = []

        def transport(endpoint, scope, path, payload, timeout=0.6, *, expected_status=None):
            observed.append((marker.get(), scope, path, payload))
            if path == "/v1/events":
                self.assertEqual(expected_status, 202)
                return job_ack(scope, payload)
            return {"hits": [{"frame": {"content": "recall"}}]}

        with (
            tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as home,
            patch.object(module, "_request", side_effect=transport),
        ):
            p = self.provider()
            p.save_config({"endpoint": "http://127.0.0.1:9"}, home)
            token = marker.set("profile-a")
            p.initialize("session", hermes_home=home, user_id="alice", cwd="project")
            try:
                p.on_turn_start(1, "message", author_id="bob", author_is_bot=False)
                p.handle_tool_call("gluomem_remember", {"content": "bob candidate"})
                self.assertTrue(p.flush(2))
                self.assertEqual(observed[-1][0], "profile-a")
                self.assertEqual(observed[-1][1], p._scope("bob"))
                self.assertIn("recall", p.prefetch("nontrivial recall question"))
                self.assertEqual(observed[-1][0], "profile-a")
                self.assertEqual(observed[-1][1], p._scope("bob"))
                with self.assertRaises(ValueError):
                    p.handle_tool_call(
                        "gluomem_remember", {"content": "c", "scope": {"principal_id": "victim"}}
                    )
            finally:
                p.shutdown()
                marker.reset(token)

    def test_setup_is_profile_local_secret_free_and_capture_off(self):
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as home:
            p = self.provider()
            p.save_config({"endpoint": "http://127.0.0.1:8765", "api_key": "never-store"}, home)
            saved = json.loads((Path(home) / "gluomem.json").read_text())
            self.assertEqual(
                saved, {"endpoint": "http://127.0.0.1:8765", "automatic_capture": False}
            )
            with (
                patch("hermes_constants.get_hermes_home", return_value=Path(home)),
                patch("agent.secret_scope.get_secret", return_value="test-key"),
            ):
                self.assertTrue(p.is_available())
                self.assertFalse(p.get_status_config()["automatic_capture"])
            self.assertTrue(
                any(
                    f.get("env_var") == "GLUOMEM_SERVICE_KEY" and f.get("secret")
                    for f in p.get_config_schema()
                )
            )

    def test_multibyte_input_exceeding_service_budget_is_not_accepted_locally(self):
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as home:
            provider = self.provider()
            provider.save_config({"endpoint": "http://127.0.0.1:9"}, home)
            provider.initialize("budget", hermes_home=home)
            try:
                with self.assertRaisesRegex(ValueError, "UTF-8"):
                    provider.handle_tool_call("gluomem_remember", {"content": "🚀" * 25_001})
                self.assertEqual(provider.status()["pending"], 0)
            finally:
                provider.shutdown()

    def test_gateway_without_authenticated_user_does_not_share_local_identity(self):
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as home:
            provider = self.provider()
            provider.save_config({"endpoint": "http://127.0.0.1:9"}, home)
            try:
                with self.assertRaisesRegex(ValueError, "user_id"):
                    provider.initialize("missing-user", hermes_home=home, platform="telegram")
                self.assertFalse((Path(home) / "gluomem/outbox.sqlite3").exists())
            finally:
                if getattr(provider, "_initialized", False):
                    provider.shutdown()

    def test_gateway_user_identifiers_are_platform_namespaced(self):
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as home:
            scopes = {}
            for platform in ("telegram", "discord", "cli", "desktop", "gui"):
                provider = self.provider()
                provider.save_config({"endpoint": "http://127.0.0.1:9"}, home)
                provider.initialize(
                    platform,
                    hermes_home=home,
                    platform=platform,
                    user_id="123",
                    cwd="same-workspace",
                )
                try:
                    scopes[platform] = provider._scope()
                finally:
                    provider.shutdown()
            self.assertNotEqual(
                scopes["telegram"]["principal_id"], scopes["discord"]["principal_id"]
            )
            self.assertNotEqual(scopes["telegram"]["principal_id"], scopes["cli"]["principal_id"])
            self.assertEqual(scopes["cli"], scopes["desktop"])
            self.assertEqual(scopes["cli"], scopes["gui"])

    def test_flush_checks_only_pending_rows_including_other_endpoints(self):
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as home:
            p = self.provider()
            p.save_config({"endpoint": "http://127.0.0.1:9"}, home)
            p.initialize("flush", hermes_home=home, agent_context="subagent")
            self.addCleanup(p.shutdown)
            with p._db() as db:
                db.execute(
                    "INSERT INTO outbox(event_id,envelope) VALUES (?,?)",
                    ("old", json.dumps({"endpoint": "http://127.0.0.1:8"})),
                )
            with (
                patch.object(p, "status", side_effect=AssertionError("full status")),
                patch("utils.read_json_or_empty", side_effect=AssertionError("config read")),
            ):
                self.assertFalse(p.flush(0))
                self.assertFalse(p.flush(0.03))
                with p._db() as db:
                    db.execute("DELETE FROM outbox")
                self.assertTrue(p.flush(0))
                self.assertTrue(p.flush(0.03))

    def test_completed_prefetch_is_consumed_once_then_recalled_fresh(self):
        module = importlib.import_module("gluomem_hermes.provider")
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as home:
            p = self.provider()
            p.save_config({"endpoint": "http://127.0.0.1:9"}, home)
            p.initialize("first", hermes_home=home, agent_context="subagent")
            self.addCleanup(p.shutdown)
            with patch.object(
                module,
                "_request",
                side_effect=[
                    {"hits": [{"frame": {"content": "queued evidence"}}]},
                    {"hits": [{"frame": {"content": "fresh evidence"}}]},
                ],
            ) as request:
                p.on_turn_start(1, "remember project details")
                p.queue_prefetch("remember project details")
                p._recall_thread.join(2)
                self.assertFalse(p._recall_thread.is_alive())
                self.assertIn("queued evidence", p.prefetch("remember project details"))
                self.assertEqual(request.call_count, 1)
                self.assertIn("fresh evidence", p.prefetch("remember project details"))
                self.assertEqual(request.call_count, 2)

    def test_completed_prefetch_is_invalidated_at_turn_author_and_session_boundaries(self):
        module = importlib.import_module("gluomem_hermes.provider")
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as home:
            for boundary in ("turn", "author", "session"):
                with self.subTest(boundary=boundary):
                    p = self.provider()
                    p.save_config({"endpoint": "http://127.0.0.1:9"}, home)
                    p.initialize(
                        "first", hermes_home=home, user_id="alice", agent_context="subagent"
                    )
                    try:
                        with patch.object(
                            module,
                            "_request",
                            side_effect=[
                                {"hits": [{"frame": {"content": "stale evidence"}}]},
                                {"hits": [{"frame": {"content": "current evidence"}}]},
                            ],
                        ) as request:
                            p.on_turn_start(1, "remember project details", author_id="alice")
                            p.queue_prefetch("remember project details")
                            p._recall_thread.join(2)
                            if boundary == "turn":
                                p.on_turn_start(2, "remember project details", author_id="alice")
                            elif boundary == "author":
                                p.on_turn_start(1, "remember project details", author_id="bob")
                                p.on_turn_start(1, "remember project details", author_id="alice")
                            else:
                                p.on_session_switch("second")
                                p.on_session_switch("first")
                            result = p.prefetch("remember project details")
                            self.assertIn("current evidence", result)
                            self.assertNotIn("stale evidence", result)
                            self.assertEqual(request.call_count, 2)
                    finally:
                        p.shutdown()

    def test_inflight_prefetch_cannot_be_revived_after_a_mismatched_query(self):
        import threading

        module = importlib.import_module("gluomem_hermes.provider")
        entered, release = threading.Event(), threading.Event()

        def recall(*args, **kwargs):
            entered.set()
            self.assertTrue(release.wait(3))
            return {"hits": [{"frame": {"content": "stale evidence"}}]}

        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as home:
            p = self.provider()
            p.save_config({"endpoint": "http://127.0.0.1:9"}, home)
            p.initialize("first", hermes_home=home, agent_context="subagent")
            try:
                with patch.object(module, "_request", side_effect=recall):
                    p.queue_prefetch("remember project details")
                    self.assertTrue(entered.wait(2))
                    self.assertEqual(p.prefetch("different project question"), "")
                    release.set()
                    p._recall_thread.join(2)
                with patch.object(
                    module,
                    "_request",
                    return_value={"hits": [{"frame": {"content": "fresh evidence"}}]},
                ) as request:
                    self.assertIn("fresh evidence", p.prefetch("remember project details"))
                    self.assertEqual(request.call_count, 1)
            finally:
                release.set()
                p.shutdown()

    def test_waiting_prefetch_drops_results_when_turn_author_changes(self):
        import threading

        module = importlib.import_module("gluomem_hermes.provider")
        entered, release, waiting = threading.Event(), threading.Event(), threading.Event()

        def recall(*args, **kwargs):
            entered.set()
            release.wait(3)
            return {"hits": [{"frame": {"content": "alice evidence"}}]}

        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as home:
            p = self.provider()
            p.save_config({"endpoint": "http://127.0.0.1:9"}, home)
            p.initialize("first", hermes_home=home, user_id="alice", agent_context="subagent")
            results = []
            consumer = None
            try:
                with patch.object(module, "_request", side_effect=recall):
                    p.queue_prefetch("remember project details")
                    self.assertTrue(entered.wait(2))
                    done = p._recall_job["done"]
                    original_wait = done.wait

                    def wait(timeout):
                        waiting.set()
                        return original_wait(timeout)

                    with patch.object(done, "wait", side_effect=wait):
                        consumer = threading.Thread(
                            target=lambda: results.append(p.prefetch("remember project details"))
                        )
                        consumer.start()
                        self.assertTrue(waiting.wait(2))
                        p.on_turn_start(2, "remember project details", author_id="bob")
                        release.set()
                        consumer.join(2)
                        p._recall_thread.join(2)
                    self.assertEqual(results, [""])
                    self.assertIsNone(p.recall_status())
                with patch.object(module, "_request", return_value={"hits": []}) as request:
                    self.assertEqual(p.prefetch("remember project details"), "")
                    self.assertEqual(request.call_args.args[1], p._scope("bob"))
            finally:
                release.set()
                if consumer is not None:
                    consumer.join(2)
                p.shutdown()

    def test_provider_identity_is_read_only(self):
        provider = self.provider()
        self.assertEqual(provider.name, "gluomem")
        with self.assertRaises(AttributeError):
            provider.name = "another-provider"
        self.assertEqual(provider.name, "gluomem")


if __name__ == "__main__":
    unittest.main()
