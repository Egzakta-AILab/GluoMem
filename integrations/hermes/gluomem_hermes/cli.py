"""Trusted operator commands discovered by the native Hermes memory loader."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, NotRequired, TypedDict
from urllib.parse import urlencode


class _FrameReadback(TypedDict):
    frame: dict[str, Any]
    lineage: list[dict[str, Any]]
    verified_readback: NotRequired[bool]


def register_cli(subparser):
    commands = subparser.add_subparsers(dest="gluomem_action", required=True)
    commands.add_parser("status", help="Local profile capture and delivery status")
    commands.add_parser("pause", help="Stop new automatic capture at the next sync")
    commands.add_parser("resume", help="Enable automatic capture at the next sync")
    listing = commands.add_parser("list", help="List one bounded page of scoped memories")
    _identity_options(listing)
    listing.add_argument("--limit", type=int, choices=range(1, 101), default=50)
    listing.add_argument("--cursor", type=int, default=0)
    provenance = commands.add_parser("provenance", help="Inspect a frame and its version lineage")
    _identity_options(provenance)
    provenance.add_argument("frame_id", type=int)
    review = commands.add_parser("review", help="Confirm a candidate as an authenticated operator")
    _identity_options(review)
    review.add_argument("frame_id", type=int)
    review.add_argument("--action", choices=["accept", "edit", "dismiss"], required=True)
    review.add_argument("--content", help="Replacement text (only for review edit)")
    review.add_argument(
        "--yes", action="store_true", required=True, help="Confirm privileged write"
    )
    correct = commands.add_parser("correct", help="Supersede a memory with operator-corrected text")
    _identity_options(correct)
    correct.add_argument("frame_id", type=int)
    correct.add_argument("--content", required=True)
    correct.add_argument(
        "--yes", action="store_true", required=True, help="Confirm privileged write"
    )
    forget = commands.add_parser("forget", help="Forget and scrub a memory's affected chain")
    _identity_options(forget)
    forget.add_argument("frame_id", type=int)
    forget.add_argument(
        "--yes", action="store_true", required=True, help="Confirm destructive write"
    )
    export = commands.add_parser("export", help="Export scoped private memories as JSON to stdout")
    _identity_options(export)
    export.add_argument(
        "--yes", action="store_true", required=True, help="Confirm private data export"
    )


def _identity_options(parser):
    parser.add_argument(
        "--workspace", required=True, help="Exact logical workspace used by the host"
    )
    parser.add_argument(
        "--platform", default="cli", help="Trusted host platform (e.g. cli, telegram)"
    )
    parser.add_argument(
        "--user-id", help="Authenticated host user ID; required for gateway platforms"
    )


def _operator_scope(home, args):
    from .provider import _LOCAL_PLATFORMS, _scope_for

    if not args.platform.strip() or not args.workspace.strip() or getattr(args, "cursor", 0) < 0:
        raise ValueError("nonempty platform/workspace and nonnegative cursor required")
    missing = args.user_id is None or not args.user_id.strip()
    if missing and args.platform not in _LOCAL_PLATFORMS:
        raise ValueError("gateway operations require an authenticated --user-id")

    return _scope_for(
        home, args.platform, "local-user" if missing else args.user_id, args.workspace
    )


def _local_status(home, config):
    from .provider import _outbox_status

    delivery = {
        "pending": 0,
        "pending_bytes": 0,
        "failed": 0,
        "delivery_errors": [],
        "dead_letters": [],
    }
    path = home / "gluomem" / "outbox.sqlite3"
    if path.exists():
        db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=0.25)
        try:
            delivery = _outbox_status(db)
        finally:
            db.close()
    return {"provider": "gluomem", **config, **delivery}


def _read_frame(endpoint, scope, frame_id) -> _FrameReadback:
    from .provider import _request

    if frame_id < 1:
        raise ValueError("frame ID must be positive")
    lineage = _request(
        endpoint,
        scope,
        f"/v1/memories/{frame_id}/lineage",
        method="GET",
        expect_list=True,
    )
    matches = [frame for frame in lineage if frame.get("id") == frame_id]
    if len(matches) != 1:
        raise ValueError("exact frame missing from lineage readback")
    return {"frame": matches[0], "lineage": lineage}


def _content(value):
    if not isinstance(value, str) or not value.strip() or len(value.encode("utf-8")) > 100_000:
        raise ValueError("content must be nonempty and at most 100000 UTF-8 bytes")
    return value.strip()


def gluomem_command(args):
    try:
        return _execute(args)
    except Exception as exc:
        # Never print exception payloads, response bodies, credentials or tracebacks.
        message = (
            f"GluoMem command failed ({type(exc).__name__}); check profile, inputs and service"
        )
        if args.gluomem_action in {"review", "correct", "forget"}:
            message += "; a write may have committed: verify before retry"
        print(json.dumps({"error": message}), file=sys.stderr)
        return 1


def _execute(args):
    from hermes_constants import get_hermes_home

    from .provider import GluoMemProvider

    home = Path(get_hermes_home()).resolve()
    provider = GluoMemProvider()
    config = provider.get_status_config()
    if args.gluomem_action in {"list", "provenance", "review", "correct", "forget", "export"}:
        from .provider import _request

        scope = _operator_scope(home, args)
        if args.gluomem_action == "list":
            path = "/v1/memories?" + urlencode({"limit": args.limit, "cursor": args.cursor})
            result = _request(config["endpoint"], scope, path, method="GET")
        elif args.gluomem_action == "provenance":
            result = _read_frame(config["endpoint"], scope, args.frame_id)
        elif args.gluomem_action == "export":
            result = _request(config["endpoint"], scope, "/v1/memories:export")
        elif args.gluomem_action == "correct":
            result = _correct(config["endpoint"], scope, args)
        elif args.gluomem_action == "forget":
            if args.frame_id < 1:
                raise ValueError("frame ID must be positive")
            _request(config["endpoint"], scope, f"/v1/memories/{args.frame_id}", method="DELETE")
            result = _read_frame(config["endpoint"], scope, args.frame_id)
            frame = result["frame"]
            if frame.get("status") != "forgotten" or frame.get("content") != "[forgotten]":
                raise ValueError("write acknowledged but forget readback did not match")
            result["verified_readback"] = True
        else:
            if args.frame_id < 1:
                raise ValueError("frame ID must be positive")
            payload = {"action": args.action}
            if args.action == "edit":
                payload["edited_content"] = _content(args.content)
            elif args.content is not None:
                raise ValueError("--content is only valid with review --action edit")
            _request(config["endpoint"], scope, f"/v1/memories/{args.frame_id}:review", payload)
            result = _read_frame(config["endpoint"], scope, args.frame_id)
            frame = result["frame"]
            valid = (
                frame.get("status") == "tombstoned" and frame.get("content") == "[forgotten]"
                if args.action == "dismiss"
                else frame.get("status") == "active" and frame.get("verified_by_user") is True
            )
            if args.action == "edit":
                valid = valid and frame.get("content") == payload["edited_content"]
            if not valid:
                raise ValueError("write acknowledged but review readback did not match")
            result["verified_readback"] = True
        print(json.dumps(result))
        return 0
    if args.gluomem_action in {"pause", "resume"}:
        provider.save_config(
            {**config, "automatic_capture": args.gluomem_action == "resume"}, str(home)
        )
        config = provider.get_status_config()  # Read back the exact written target.
    print(json.dumps(_local_status(home, config)))
    return 0


def _correct(endpoint, scope, args):
    from .provider import _request

    if args.frame_id < 1:
        raise ValueError("frame ID must be positive")
    content = _content(args.content)
    ack = _request(
        endpoint, scope, f"/v1/memories/{args.frame_id}:correct", {"new_content": content}
    )
    replacement_id = ack.get("replacement_frame", {}).get("id")
    if type(replacement_id) is not int or replacement_id < 1 or replacement_id == args.frame_id:
        raise ValueError("invalid correction acknowledgement; verify before retry")
    old = _read_frame(endpoint, scope, args.frame_id)["frame"]
    new = _read_frame(endpoint, scope, replacement_id)["frame"]
    if (
        old.get("status") != "superseded"
        or old.get("superseded_by") != replacement_id
        or new.get("supersedes") != args.frame_id
        or new.get("content") != content
        or new.get("status") != "active"
    ):
        raise ValueError("write acknowledged but correction readback did not match")
    return {"superseded_frame": old, "replacement_frame": new, "verified_readback": True}
