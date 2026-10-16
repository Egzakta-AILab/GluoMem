"""Operator CLI integration tests, executed by a real Hermes host Python."""

import argparse
import contextlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PLUGIN = Path(__file__).resolve().parents[1] / "integrations" / "hermes"
sys.path.insert(0, str(PLUGIN))


@unittest.skipUnless(importlib.util.find_spec("agent"), "requires actual Hermes host")
class NativeCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"])
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        self.env = patch.dict(os.environ, {"HERMES_HOME": str(self.home)})
        self.env.start()
        self.addCleanup(self.env.stop)
        from gluomem_hermes.provider import GluoMemProvider

        self.provider = GluoMemProvider()
        self.provider.save_config({"endpoint": "http://127.0.0.1:9"}, self.home)

    def invoke(self, *argv):
        from gluomem_hermes.cli import gluomem_command, register_cli

        parser = argparse.ArgumentParser()
        register_cli(parser)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = gluomem_command(parser.parse_args(argv))
        self.assertEqual(result, 0)
        return json.loads(output.getvalue())

    def test_actual_native_discovery_registers_status_without_initializing_provider(self):
        from hermes_cli.config import save_config
        from plugins.memory import discover_plugin_cli_commands

        target = self.home / "plugins" / "gluomem"
        shutil.copytree(PLUGIN / "gluomem_hermes", target)
        save_config({"memory": {"provider": "gluomem"}})
        commands = discover_plugin_cli_commands()
        self.assertEqual([item["name"] for item in commands], ["gluomem"])
        parser = argparse.ArgumentParser()
        commands[0]["setup_fn"](parser)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            result = commands[0]["handler_fn"](parser.parse_args(["status"]))
        self.assertEqual(result, 0)
        status = json.loads(output.getvalue())
        self.assertFalse(status["automatic_capture"])
        self.assertEqual(status["pending"], 0)
        self.assertEqual(status["delivery_errors"], [])
        self.assertFalse((self.home / "gluomem" / "outbox.sqlite3").exists())
        self.assertNotIn("dspy", sys.modules)

    def server(self, respond):
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        self.requests = []

        requests = self.requests

        class Handler(BaseHTTPRequestHandler):
            def handle_request(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                body = json.loads(raw) if raw else None
                requests.append((self.command, self.path, dict(self.headers), body))
                status, result = respond(self.command, self.path, body)
                encoded = json.dumps(result).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            do_POST = do_GET = do_DELETE = handle_request

            def log_message(self, format: str, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.provider.save_config({"endpoint": f"http://127.0.0.1:{server.server_port}"}, self.home)
        credential = patch("agent.secret_scope.get_secret", return_value="isolated-key")
        credential.start()
        self.addCleanup(credential.stop)

    def test_list_sends_native_identity_scoped_get_and_bounded_pagination(self):
        self.server(
            lambda method, path, body: (
                200,
                {"active_frames": [], "review_queue": [], "next_cursor": 4},
            )
        )
        result = self.invoke(
            "list",
            "--workspace",
            "C:/project",
            "--platform",
            "telegram",
            "--user-id",
            "alice",
            "--limit",
            "3",
            "--cursor",
            "2",
        )
        self.assertEqual(result["next_cursor"], 4)
        method, path, headers, body = self.requests[0]
        self.assertEqual((method, path, body), ("GET", "/v1/memories?limit=3&cursor=2", None))
        self.provider.initialize(
            "reference",
            hermes_home=self.home,
            cwd="C:/project",
            platform="telegram",
            user_id="alice",
            agent_context="subagent",
        )
        self.addCleanup(self.provider.shutdown)
        for key, value in self.provider._scope().items():
            header = {"tenant_id": "Tenant", "principal_id": "Principal", "namespace": "Namespace"}[
                key
            ]
            self.assertEqual(headers["X-GluoMem-" + header], value)
        self.assertEqual(headers["Authorization"], "Bearer isolated-key")

    def test_remote_scope_does_not_initialize_provider_or_create_outbox(self):
        self.server(lambda method, path, body: (200, {"active_frames": []}))
        with patch.object(type(self.provider), "initialize", side_effect=AssertionError("I/O")):
            self.invoke(
                "list", "--workspace", "project", "--platform", "telegram", "--user-id", " alice "
            )
        self.assertFalse((self.home / "gluomem").exists())
        import hashlib

        def digest(value):
            return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:40]

        headers = self.requests[0][2]
        self.assertEqual(headers["X-GluoMem-Tenant"], "hermes-" + digest(self.home.resolve()))
        self.assertEqual(headers["X-GluoMem-Namespace"], "workspace-" + digest("project"))
        self.assertEqual(
            headers["X-GluoMem-Principal"],
            "user-" + digest(json.dumps(["gateway", "telegram", " alice "])),
        )

    def test_provenance_reads_exact_frame_from_native_lineage_route(self):
        frame = {"id": 7, "content": "original", "provenance": [{"source_ref": "source:a"}]}
        self.server(lambda method, path, body: (200, [frame]))
        result = self.invoke("provenance", "7", "--workspace", "project")
        self.assertEqual(result["frame"], frame)
        self.assertEqual(result["lineage"], [frame])
        self.assertEqual(self.requests[0][:2], ("GET", "/v1/memories/7/lineage"))

    def test_review_accept_requires_confirmation_and_reads_back_exact_frame(self):
        frame = {"id": 7, "content": "candidate", "status": "candidate", "verified_by_user": False}

        def respond(method, path, body):
            if method == "POST":
                self.assertEqual((path, body), ("/v1/memories/7:review", {"action": "accept"}))
                frame.update(status="active", verified_by_user=True)
                return 200, {"action": "accept", "frame": frame}
            return 200, [frame]

        self.server(respond)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            self.invoke("review", "7", "--action", "accept", "--workspace", "project")
        self.assertEqual(self.requests, [])
        result = self.invoke("review", "7", "--action", "accept", "--yes", "--workspace", "project")
        self.assertTrue(result["verified_readback"])
        self.assertEqual(result["frame"]["status"], "active")
        self.assertEqual(
            [r[:2] for r in self.requests],
            [("POST", "/v1/memories/7:review"), ("GET", "/v1/memories/7/lineage")],
        )

    def test_review_edit_and_dismiss_use_service_actions_and_verify_result(self):
        frame = {"id": 7, "status": "candidate", "verified_by_user": False, "content": "old"}

        def respond(method, path, body):
            if method == "POST":
                if body["action"] == "edit":
                    self.assertEqual(body["edited_content"], "replacement")
                    frame.update(content="replacement", status="active", verified_by_user=True)
                else:
                    frame.update(content="[forgotten]", status="tombstoned")
                return 200, {"frame": frame}
            return 200, [frame]

        self.server(respond)
        edited = self.invoke(
            "review",
            "7",
            "--action",
            "edit",
            "--content",
            "replacement",
            "--yes",
            "--workspace",
            "project",
        )
        self.assertEqual(edited["frame"]["content"], "replacement")
        dismissed = self.invoke(
            "review", "7", "--action", "dismiss", "--yes", "--workspace", "project"
        )
        self.assertEqual(dismissed["frame"]["status"], "tombstoned")
        self.assertTrue(dismissed["verified_readback"])

    def test_correct_reads_both_exact_superseded_and_replacement_targets(self):
        old = {"id": 7, "content": "old", "status": "active", "superseded_by": None}
        new = {"id": 8, "content": "new fact", "status": "active", "supersedes": 7}

        def respond(method, path, body):
            if method == "POST":
                self.assertEqual(
                    (path, body), ("/v1/memories/7:correct", {"new_content": "new fact"})
                )
                old.update(content="[forgotten]", status="superseded", superseded_by=8)
                return 200, {"superseded_frame": old, "replacement_frame": new}
            return 200, [old, new]

        self.server(respond)
        result = self.invoke(
            "correct", "7", "--content", "new fact", "--yes", "--workspace", "project"
        )
        self.assertTrue(result["verified_readback"])
        self.assertEqual(result["replacement_frame"], new)
        self.assertEqual(
            [r[:2] for r in self.requests],
            [
                ("POST", "/v1/memories/7:correct"),
                ("GET", "/v1/memories/7/lineage"),
                ("GET", "/v1/memories/8/lineage"),
            ],
        )

    def test_forget_requires_confirmation_and_verifies_scrubbed_exact_target(self):
        frame = {"id": 7, "content": "old", "status": "active"}

        def respond(method, path, body):
            if method == "DELETE":
                self.assertIsNone(body)
                frame.update(content="[forgotten]", status="forgotten")
                return 200, {"frame_id": 7, "reason": "forgotten"}
            return 200, [frame]

        self.server(respond)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            self.invoke("forget", "7", "--workspace", "project")
        self.assertEqual(self.requests, [])
        result = self.invoke("forget", "7", "--yes", "--workspace", "project")
        self.assertTrue(result["verified_readback"])
        self.assertEqual(result["frame"]["status"], "forgotten")
        self.assertEqual(
            [r[:2] for r in self.requests],
            [("DELETE", "/v1/memories/7"), ("GET", "/v1/memories/7/lineage")],
        )

    def test_export_uses_actual_scoped_export_route_and_writes_json_to_stdout(self):
        exported = {
            "active_memories": [{"id": 7, "content": "private"}],
            "review_queue": [],
            "scope_generation": 2,
        }
        self.server(lambda method, path, body: (200, exported))
        result = self.invoke("export", "--workspace", "project", "--yes")
        self.assertEqual(result, exported)
        self.assertEqual(
            [(r[0], r[1], r[3]) for r in self.requests], [("POST", "/v1/memories:export", None)]
        )

    def fail_command(self, *argv):
        from gluomem_hermes.cli import gluomem_command, register_cli

        parser = argparse.ArgumentParser()
        register_cli(parser)
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = gluomem_command(parser.parse_args(argv))
        self.assertEqual(result, 1)
        self.assertEqual(stdout.getvalue(), "")
        return json.loads(stderr.getvalue())

    def test_failed_write_readback_is_nonzero_without_false_success_or_secret_leak(self):
        def respond(method, path, body):
            if method == "POST":
                return 200, {"frame": {"id": 7, "status": "active", "verified_by_user": True}}
            return 401, {"error": "sensitive server body isolated-key"}

        self.server(respond)
        error = self.fail_command(
            "review", "7", "--action", "accept", "--yes", "--workspace", "project"
        )
        self.assertIn("verify before retry", error["error"])
        self.assertNotIn("isolated-key", json.dumps(error))
        self.assertEqual(len(self.requests), 2)

    def test_rejects_untrusted_scope_inputs_and_missing_gateway_identity_before_http(self):
        self.server(lambda method, path, body: (200, {}))
        self.fail_command("list", "--workspace", "project", "--platform", "telegram")
        self.assertFalse((self.home / "gluomem" / "outbox.sqlite3").exists())
        for argv in [
            ("list", "--workspace", "project", "--tenant-id", "evil"),
            ("list", "--workspace", "project", "--principal-id", "evil"),
            ("list", "--workspace", "project", "--limit", "101"),
            ("correct", "7", "--workspace", "project", "--content", "fact"),
            ("export", "--workspace", "project"),
        ]:
            with (
                self.subTest(argv=argv),
                self.assertRaises(SystemExit),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.invoke(*argv)
        self.fail_command(
            "correct", "7", "--workspace", "project", "--content", "語" * 33334, "--yes"
        )
        self.fail_command("provenance", "0", "--workspace", "project")
        self.assertEqual(self.requests, [])

    def test_transport_rejects_redirect_and_oversized_export_without_partial_stdout(self):
        response = [302, {"redirect": "never follow"}]
        self.server(lambda method, path, body: tuple(response))
        self.fail_command("export", "--workspace", "project", "--yes")
        self.assertEqual(len(self.requests), 1)
        response[:] = [200, {"large": "x" * 1_000_001}]
        self.fail_command("export", "--workspace", "project", "--yes")
        self.assertEqual(len(self.requests), 2)

    def test_full_native_cli_process_controls_running_provider_in_scratch_profile(self):
        import subprocess

        from hermes_cli.config import save_config

        shutil.copytree(PLUGIN / "gluomem_hermes", self.home / "plugins" / "gluomem")
        save_config({"memory": {"provider": "gluomem"}})
        self.provider.initialize("running", hermes_home=self.home, cwd="project")
        self.addCleanup(self.provider.shutdown)
        for action, enabled in [("resume", True), ("status", True), ("pause", False)]:
            run = subprocess.run(
                [sys.executable, "-m", "hermes_cli.main", "gluomem", action],
                env=dict(os.environ),
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(json.loads(run.stdout)["automatic_capture"], enabled)
            self.assertEqual(self.provider.status()["automatic_capture"], enabled)

    def test_transport_rejects_nonobject_lineage_items(self):
        from gluomem_hermes.provider import _request

        self.server(lambda method, path, body: (200, [42]))
        with self.assertRaisesRegex(ValueError, "invalid service response"):
            _request(
                self.provider.get_status_config()["endpoint"],
                {"tenant_id": "test", "principal_id": "test", "namespace": "test"},
                "/v1/memories/7/lineage",
                method="GET",
                expect_list=True,
            )

    def test_pause_resume_affects_already_running_provider_next_sync(self):
        self.provider.initialize("running", hermes_home=self.home, cwd="project")
        self.addCleanup(self.provider.shutdown)
        result = self.invoke("resume")
        self.assertTrue(result["automatic_capture"])
        self.provider.sync_turn("first human turn", "ignored assistant")
        self.assertEqual(self.provider.status()["pending"], 1)
        result = self.invoke("pause")
        self.assertFalse(result["automatic_capture"])
        self.assertFalse(self.provider.status()["automatic_capture"])
        self.provider.sync_turn("paused turn", "ignored assistant")
        self.assertEqual(self.provider.status()["pending"], 1)
        result = self.invoke("resume")
        self.provider.sync_turn("second human turn", "ignored assistant")
        self.assertEqual(self.provider.status()["pending"], 2)
        self.assertEqual(self.provider.get_status_config()["automatic_capture"], True)
        self.assertEqual(
            {row["name"] for row in self.provider.get_tool_schemas()},
            {"gluomem_remember", "gluomem_status"},
        )


if __name__ == "__main__":
    unittest.main()
