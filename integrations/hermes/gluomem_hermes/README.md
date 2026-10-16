# GluoMem for Hermes

A thin native provider for a separately operated GluoMem service and worker.
It uses Hermes host APIs and the standard library; it does not load the embedded
SDK, DSPy, models, or the memory database into Hermes.

## Installation

From `integrations/hermes`, using Python 3.11+:

```sh
python -m gluomem_hermes.install --hermes-home /absolute/path/to/profile
```

The installer copies the package into the selected profile's `plugins/gluomem/`.
It refuses to overwrite an existing plugin and changes neither configuration nor
credentials. Back up that directory before upgrading; retain the profile's
`gluomem.json` and `gluomem/` data directory.

For owner-managed environments, the independent `gluomem-hermes` wheel registers
`hermes_agent.memory_providers`. Do not modify a managed Hermes environment merely
to enable a directory provider.

Start a private GluoMem service and worker, then configure the intended profile:

```sh
HERMES_HOME=/absolute/path/to/profile hermes memory setup gluomem
HERMES_HOME=/absolute/path/to/profile hermes memory status
```

Hermes stores the service key in profile-scoped secret configuration. Non-secret
settings live in `gluomem.json`:

```json
{"endpoint": "http://127.0.0.1:8765", "automatic_capture": false}
```

Only loopback endpoints permit HTTP; remote endpoints require HTTPS. URL
credentials, paths, queries, fragments, redirects and ambient proxies are rejected.
The shared service key belongs on a trusted host, never in a browser client.

## Operator commands

Select the intended profile for every invocation. Supply the exact logical
workspace supplied by the host, not an inferred filesystem directory.

```sh
hermes gluomem status
hermes gluomem pause
hermes gluomem resume
hermes gluomem list --workspace example
hermes gluomem provenance 7 --workspace example
hermes gluomem review 7 --action accept --yes --workspace example
hermes gluomem correct 7 --content 'Replacement fact' --yes --workspace example
hermes gluomem forget 7 --yes --workspace example
hermes gluomem export --yes --workspace example > private-export.json
```

See `hermes gluomem --help` for pagination and remote-author options. Remote
platforms require an authenticated author ID. Review/correct/forget verify the
exact target after writing; failed readback does not prove rejection. Exports
contain private data and are limited to 1 MB per response.

## Safety and delivery

- Capture defaults off. Automatic capture sends only completed human user text,
  excluding assistant/tool output, compression summaries, bots and delegated work.
- Automatic and model-authored observations require review. Model tools cannot
  select scopes or assert user confirmation; trusted operators promote candidates.
- Scope derives from the host profile, authenticated author/platform and logical
  workspace. Scope hashing does not replace server-side authorization.
- `queued_local` means a durable SQLite outbox entry, not a remembered fact.
  Dequeue requires an actual HTTP 202 matching the frozen scope, event/revision
  and async job identity. Even that acknowledgement means queue acceptance only.
- Pending envelopes retain their original endpoint and identity. Pausing stops new
  automatic capture, not delivery of accepted work or explicit proposals.
- Profile-wide admission defaults to 1000 events / 16 MiB; configure
  `outbox_max_events` and `outbox_max_bytes` in `gluomem.json`. Accepted entries
  are not evicted when limits shrink. Permanent rejections keep metadata-only
  receipts; transient failures remain pending. Inspect status after flushing.
- The outbox contains sensitive plaintext. Its byte budget is not a disk/WAL quota;
  logical deletion does not erase offline copies, backups or storage media.
- Recall waits at most 1.8 seconds and injects at most 1500 UTF-8 bytes. Native
  compression-checkpoint durability and built-in-memory mirroring are not provided.

## Development

Run `tests/test_hermes_*.py` with an actual Hermes interpreter; host-dependent tests
skip in a core-only environment. Type-check this independent package with Hermes
source available instead of adding Hermes to the core distribution's dependencies.
