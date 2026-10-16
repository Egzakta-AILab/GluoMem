"""Real SQLite/HTTP outbox contracts; execute using the actual Hermes Python."""

import importlib
import importlib.util
import json
import os
import sys
import tempfile
import threading
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import patch

PLUGIN = Path(__file__).resolve().parents[1] / "integrations" / "hermes"
sys.path.insert(0, str(PLUGIN))


def job_ack(scope, payload):
    """Actual service JobResponse wire shape; no core/ML imports in the host."""
    return {
        "id": 1,
        "scope": scope,
        "job_type": "ingest_event",
        "dedup_key": f"{payload['event_id']}:{payload['revision']}",
        "status": "queued",
        "attempt_count": 0,
        "max_attempts": 3,
        "scope_generation": 0,
        "last_error": None,
        "created_at": "2026-10-16T00:00:00+00:00",
        "updated_at": "2026-10-16T00:00:00+00:00",
    }


@contextmanager
def service(status=202):
    received = []
    responses: dict[str, Any] = {"status": status}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append(payload)
            code = responses["status"] if payload["content"] != "valid later" else 202
            scope = {
                field: self.headers[header]
                for field, header in (
                    ("tenant_id", "X-GluoMem-Tenant"),
                    ("principal_id", "X-GluoMem-Principal"),
                    ("namespace", "X-GluoMem-Namespace"),
                )
            }
            ack = job_ack(scope, payload)
            ack.update(responses.get("ack", {}))
            for field in responses.get("omit", ()):
                ack.pop(field, None)
            body = json.dumps(ack if code in (200, 202) else {"secret": "BODY-SECRET"}).encode()
            self.send_response(code)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", received, responses
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


@unittest.skipUnless(importlib.util.find_spec("agent"), "requires actual Hermes host")
class OutboxReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"])
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        self.secret = patch("agent.secret_scope.get_secret", return_value="TEST-KEY")
        self.secret.start()
        self.addCleanup(self.secret.stop)

    def provider(self, endpoint, **config):
        p = importlib.import_module("gluomem_hermes").GluoMemProvider()
        p.save_config({"endpoint": endpoint, **config}, str(self.home))
        p.initialize("session", hermes_home=self.home, agent_context="subagent")
        p._primary = True  # Deterministically seed the real DB before starting delivery.
        self.addCleanup(p.shutdown)
        return p

    def start(self, p):
        p._writer = threading.Thread(target=p._deliver, daemon=True)
        p._writer.start()

    def test_invalid_async_ack_is_pending_and_replays_frozen_envelope_after_restart(self):
        import time

        cases = (
            [(202, {"id": value}) for value in (None, "1", True, 0, -1, 1.5)]
            + [
                (
                    202,
                    {
                        "scope": {
                            "tenant_id": "wrong",
                            "principal_id": "wrong",
                            "namespace": "wrong",
                        }
                    },
                ),
                (202, {"scope": None}),
                (202, {"job_type": "sweep_expired"}),
                (202, {"dedup_key": "other:1"}),
                (202, {"status": "pending"}),
                (200, {"http_status": 202}),  # A body cannot spoof transport acceptance.
                (201, {}),
                (204, {}),
            ]
            + [
                (202, {"scope": {field: "wrong"}})
                for field in ("tenant_id", "principal_id", "namespace")
            ]
        )
        for i, (code, ack) in enumerate(cases):
            with self.subTest(code=code, ack=ack), service(code) as (endpoint, received, responses):
                responses["ack"] = ack
                p = self.provider(endpoint)
                p._enqueue("private candidate", event_id=f"malformed-{i}", user_id="alice")
                with p._db() as db:
                    frozen = db.execute("SELECT envelope FROM outbox").fetchone()[0]
                self.start(p)
                deadline = time.monotonic() + 2
                while not p.status()["delivery_errors"] and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertFalse(p.flush(0.05))
                p.shutdown()
                self.assertEqual(p.status()["failed"], 0)
                self.assertEqual(p.status()["pending"], 1)
                with p._db() as db:
                    self.assertEqual(
                        db.execute("SELECT envelope FROM outbox").fetchone()[0], frozen
                    )
                self.assertNotIn("private candidate", json.dumps(p.status()))
                responses["ack"] = {}
                responses["status"] = 202
                q = self.provider(endpoint)
                q._user_id = "bob"
                q._workspace = "changed-workspace"
                self.start(q)
                self.assertTrue(q.flush(2))
                q.shutdown()
                self.assertEqual(received[-1], json.loads(frozen)["payload"])

    def test_missing_ack_identity_never_dequeues_even_with_matching_event_id(self):
        for field in ("id", "scope", "job_type", "dedup_key", "status"):
            with self.subTest(field=field), service() as (endpoint, _, responses):
                p = self.provider(endpoint)
                p._enqueue("private candidate", event_id="matching-event")
                responses["omit"] = [field]
                responses["ack"] = {"event_id": "matching-event"}
                self.start(p)
                self.assertFalse(p.flush(0.1))
                p.shutdown()
                self.assertEqual(p.status()["pending"], 1)
                self.assertEqual(p.status()["failed"], 0)
                responses["omit"] = []
                q = self.provider(endpoint)
                self.start(q)
                self.assertTrue(q.flush(2))
                q.shutdown()

    def test_all_remote_job_states_mean_queue_acceptance_not_memory_completion(self):
        for state in ("queued", "leased", "succeeded", "dead_letter"):
            with self.subTest(state=state), service() as (endpoint, _, responses):
                responses["ack"] = {"status": state}
                p = self.provider(endpoint)
                receipt = p._enqueue("review candidate", event_id=f"state-{state}")
                self.assertFalse(receipt["remote_committed"])
                self.assertEqual(receipt["status"], "queued_local")
                self.start(p)
                self.assertTrue(p.flush(2))
                p.shutdown()
                self.assertEqual(p.status()["pending"], 0)

    def test_legacy_outbox_additive_index_preserves_rows_and_bounds_endpoint_seek(self):
        p = self.provider("http://127.0.0.1:9")
        p._enqueue("frozen private", event_id="legacy")
        p.shutdown()
        with p._db() as db:
            db.execute("DROP INDEX outbox_endpoint_fifo")
            frozen = db.execute("SELECT * FROM outbox").fetchall()
        q = self.provider("http://127.0.0.1:8")
        with q._db() as db:
            self.assertEqual(db.execute("SELECT * FROM outbox").fetchall(), frozen)
            plan = db.execute(
                "EXPLAIN QUERY PLAN SELECT event_id,envelope FROM outbox "
                "WHERE json_extract(envelope, '$.endpoint')=? ORDER BY rowid LIMIT 100",
                (q._config["endpoint"],),
            ).fetchall()
        details = " ".join(row[3] for row in plan)
        self.assertIn("SEARCH outbox USING INDEX outbox_endpoint_fifo", details)
        self.assertNotIn("TEMP B-TREE", details)

    def test_permanent_rejection_receipt_survives_restart_and_fifo_continues(self):
        for code in (400, 409, 410, 413, 422):
            with self.subTest(code=code), service(code) as (endpoint, received, _):
                p = self.provider(endpoint)
                p._enqueue("REJECTED-PLAINTEXT", event_id=f"bad-{code}")
                p._enqueue("valid later", event_id=f"good-{code}")
                self.start(p)
                self.assertTrue(p.flush(2), "permanent rejection must not poison FIFO")
                p.shutdown()
                q = self.provider(endpoint)
                status = q.status()
                self.assertEqual(status["failed"], 1)
                receipt = status["dead_letters"][0]
                self.assertEqual(receipt["http_status"], code)
                self.assertEqual(receipt["attempts"], 1)
                with q._db() as db:
                    dump = "\n".join(db.iterdump())
                self.assertNotIn("REJECTED-PLAINTEXT", dump)
                self.assertNotIn("BODY-SECRET", dump)
                self.assertNotIn("TEST-KEY", dump)
                self.assertEqual(len(received), 2)
                with q._db() as db:
                    db.execute("DELETE FROM dead_letters")
                q.shutdown()

    def test_event_capacity_is_atomic_global_and_duplicate_is_accepted_at_capacity(self):
        from concurrent.futures import ThreadPoolExecutor

        p = self.provider("http://127.0.0.1:9", outbox_max_events=3)
        q = self.provider("http://127.0.0.1:9", outbox_max_events=3)
        gate = threading.Barrier(16)

        def admit(i):
            gate.wait(5)
            try:
                (p if i % 2 else q)._enqueue("accepted durable", event_id=f"event-{i}")
                return i
            except ValueError as exc:
                self.assertIn("capacity", str(exc))
                return None

        with ThreadPoolExecutor(max_workers=16) as pool:
            accepted = [i for i in pool.map(admit, range(16)) if i is not None]
        self.assertEqual(len(accepted), 3)
        self.assertEqual(p.status()["pending"], 3)
        result = p._enqueue("accepted durable", event_id=f"event-{accepted[0]}")
        self.assertEqual(result["status"], "queued_local")
        with self.assertRaisesRegex(ValueError, "idempotency"):
            p._enqueue("different content", event_id=f"event-{accepted[0]}")
        p.shutdown()
        q.shutdown()
        r = self.provider("http://127.0.0.1:9", outbox_max_events=1)
        self.assertEqual(r.status()["pending"], 3, "lower limits cannot evict accepted work")
        with self.assertRaisesRegex(ValueError, "capacity"):
            r._enqueue("not accepted")

    def test_byte_capacity_counts_utf8_envelopes_and_serializes_concurrent_admission(self):
        from concurrent.futures import ThreadPoolExecutor

        p = self.provider("http://127.0.0.1:9", outbox_max_bytes=3000)
        gate = threading.Barrier(12)

        def admit(i):
            gate.wait(5)
            try:
                p._enqueue("界" * 100, event_id=f"event-{i:02}")
                return True
            except ValueError as exc:
                self.assertIn("capacity", str(exc))
                return False

        with ThreadPoolExecutor(max_workers=12) as pool:
            accepted = sum(pool.map(admit, range(12)))
        with p._db() as db:
            count, size = db.execute(
                "SELECT count(*),sum(length(CAST(envelope AS BLOB))) FROM outbox"
            ).fetchone()
        self.assertEqual(count, accepted)
        self.assertGreater(accepted, 0)
        self.assertLess(accepted, 12)
        self.assertLessEqual(size, 3000)
        self.assertEqual(p.status()["pending_bytes"], size)
        with patch("hermes_constants.get_hermes_home", return_value=self.home):
            self.assertEqual(p.get_status_config()["outbox_max_bytes"], 3000)

    def test_native_status_is_readonly_and_supports_legacy_and_dead_letter_databases(self):
        from gluomem_hermes.cli import _local_status

        p = self.provider("http://127.0.0.1:9")
        p._enqueue("PRIVATE-CONTENT", event_id="pending")
        with p._db() as db:
            db.execute("INSERT INTO dead_letters VALUES ('rejected',422,2,123.0)")
        status = _local_status(self.home, p._config)
        self.assertEqual(status["failed"], 1)
        self.assertEqual(status["dead_letters"][0]["event_id"], "rejected")
        self.assertGreater(status["pending_bytes"], 0)
        self.assertNotIn("PRIVATE-CONTENT", json.dumps(status))
        with p._db() as db:
            db.execute("DROP TABLE dead_letters")
        before = p._db_path.read_bytes()
        status = _local_status(self.home, p._config)
        self.assertEqual(status["failed"], 0)
        self.assertEqual(status["dead_letters"], [])
        self.assertEqual(before, p._db_path.read_bytes())
        with p._db() as db:
            self.assertIsNone(
                db.execute("SELECT 1 FROM sqlite_master WHERE name='dead_letters'").fetchone()
            )

    def test_transient_http_errors_remain_durable_and_replay_after_restart(self):
        import time

        for code in (401, 403, 404, 408, 429, 500, 503):
            with self.subTest(code=code), service(code) as (endpoint, received, responses):
                p = self.provider(endpoint)
                p._enqueue("retry plaintext", event_id=f"retry-{code}")
                self.start(p)
                deadline = time.monotonic() + 2
                while not p.status()["delivery_errors"] and time.monotonic() < deadline:
                    time.sleep(0.01)
                status = p.status()
                self.assertEqual(status["failed"], 0)
                self.assertEqual(status["pending"], 1)
                self.assertEqual(status["delivery_errors"][0]["error"], f"service HTTP {code}")
                p.shutdown()
                responses["status"] = 202
                q = self.provider(endpoint)
                self.assertEqual(q.status()["pending"], 1)
                self.start(q)
                self.assertTrue(q.flush(2))
                self.assertGreaterEqual(len(received), 2)
                q.shutdown()

    def test_config_capacity_bounds_reject_invalid_values(self):
        from gluomem_hermes.provider import _validate_config

        for key, maximum in (("outbox_max_events", 100_000), ("outbox_max_bytes", 1_073_741_824)):
            for value in (True, False, 0, -1, 1.5, "100", maximum + 1):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    _validate_config({"endpoint": "http://127.0.0.1:9", key: value})

    def test_rejected_event_id_cannot_be_claimed_as_newly_accepted(self):
        with service(422) as (endpoint, _, _):
            p = self.provider(endpoint)
            p._enqueue("rejected private", event_id="terminal")
            self.start(p)
            self.assertTrue(p.flush(2))
            with self.assertRaisesRegex(ValueError, "rejected"):
                p._enqueue("rejected private", event_id="terminal")
            self.assertEqual(p.status()["pending"], 0)

    def test_native_loader_and_cli_report_failed_receipts_without_private_payload(self):
        import subprocess

        env = dict(
            os.environ,
            HERMES_HOME=str(self.home),
            PYTHONPATH=str(PLUGIN),
            GLUOMEM_SERVICE_KEY="test-only-key",
        )
        result = subprocess.run(
            [sys.executable, "-m", "gluomem_hermes.install", "--hermes-home", str(self.home)],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        with service(422) as (endpoint, _, _):
            code = """
import json, sys
from plugins.memory import load_memory_provider
from hermes_constants import get_hermes_home
from hermes_cli.config import save_config
save_config({'memory': {'provider': 'gluomem'}})
p = load_memory_provider('gluomem', register_skills=False)
assert p is not None
p.save_config({'endpoint': sys.argv[1], 'outbox_max_events': 1}, str(get_hermes_home()))
p.initialize('native', hermes_home=get_hermes_home())
p.handle_tool_call('gluomem_remember', {'content': 'NEVER-PRINT-PRIVATE'})
assert p.flush(3)
assert p.status()['failed'] == 1
p.shutdown()
"""
            run = subprocess.run(
                [sys.executable, "-c", code, endpoint],
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        status = subprocess.run(
            [sys.executable, "-m", "hermes_cli.main", "gluomem", "status"],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(status.returncode, 0, status.stdout + status.stderr)
        parsed = json.loads(status.stdout)
        self.assertEqual(parsed["failed"], 1)
        self.assertEqual(parsed["pending"], 0)
        self.assertEqual(parsed["outbox_max_events"], 1)
        self.assertEqual(parsed["dead_letters"][0]["http_status"], 422)
        self.assertNotIn("NEVER-PRINT-PRIVATE", status.stdout + status.stderr)

    def test_running_instances_share_live_profile_capacity_without_evicting(self):
        p = self.provider("http://127.0.0.1:9", outbox_max_events=3)
        p._enqueue("first", event_id="first")
        q = self.provider("http://127.0.0.1:9", outbox_max_events=1)
        with self.assertRaisesRegex(ValueError, "capacity"):
            p._enqueue("must not accept using stale capacity")
        self.assertEqual(q.status()["pending"], 1)
        self.assertEqual(p.status()["outbox_max_events"], 1)
        self.assertEqual(p._enqueue("first", event_id="first")["status"], "queued_local")

    def test_full_old_endpoint_batch_cannot_starve_current_endpoint_fifo(self):
        import time

        with service() as (old_endpoint, old_received, _), service() as (endpoint, received, _):
            p = self.provider(old_endpoint)
            for i in range(100):
                p._enqueue(f"old private {i}", event_id=f"old-{i:03}")
            with p._db() as db:
                frozen = db.execute("SELECT * FROM outbox ORDER BY rowid").fetchall()
            p.shutdown()
            q = self.provider(endpoint)
            q._user_id = "bob"
            q._workspace = "different"
            q._enqueue("current private", event_id="current-0")
            self.assertEqual(q.status()["pending"], 101)
            self.start(q)
            deadline = time.monotonic() + 2
            while q.status()["pending"] == 101 and time.monotonic() < deadline:
                time.sleep(0.01)
            q.shutdown()
            self.assertEqual([r["event_id"] for r in received], ["current-0"])
            self.assertEqual(old_received, [])
            self.assertEqual(q.status()["pending"], 100)
            self.assertFalse(q.flush(0))
            with q._db() as db:
                self.assertEqual(
                    db.execute("SELECT * FROM outbox ORDER BY rowid").fetchall(), frozen
                )
            # A second bounded batch preserves eligible FIFO as well as global capacity.
            r = self.provider(endpoint, outbox_max_events=102)
            r._enqueue("next one", event_id="current-1")
            r._enqueue("next two", event_id="current-2")
            with self.assertRaisesRegex(ValueError, "capacity"):
                r._enqueue("over global limit")
            self.start(r)
            deadline = time.monotonic() + 2
            while r.status()["pending"] > 100 and time.monotonic() < deadline:
                time.sleep(0.01)
            r.shutdown()
            self.assertEqual(
                [a["event_id"] for a in received], ["current-0", "current-1", "current-2"]
            )
            with r._db() as db:
                self.assertEqual(
                    db.execute("SELECT * FROM outbox ORDER BY rowid").fetchall(), frozen
                )

    def test_endpoint_change_never_retargets_accepted_private_work(self):
        with service() as (endpoint, received, _):
            p = self.provider("http://127.0.0.1:9")
            p._enqueue("old private origin", event_id="old")
            p.shutdown()
            q = self.provider(endpoint)
            q._enqueue("valid later", event_id="new")
            self.start(q)
            self.assertFalse(q.flush(0.3))
            q.shutdown()
            self.assertEqual([r["event_id"] for r in received], ["new"])
            self.assertEqual(q.status()["pending"], 1)


if __name__ == "__main__":
    unittest.main()
