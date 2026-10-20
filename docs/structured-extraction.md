# Structured extraction policy

Structured-output controls are opt-in. Existing defaults remain `adapter="chat"`,
`canonicalize=False`, `schema_mode="auto"`, `schema_strict=False`,
`validation_retries=0`, and `output_field_limits={}`. Model selection, generation
budgets, and the frame, storage, provenance, privacy, and temporal algorithms are
unchanged.

With the `llm` extra installed, configure a declared JSON-schema transport:

```python
from gluomem.config.app_config import AppConfig, DspyConfig

cfg = AppConfig(
    dspy=DspyConfig(
        model="my-model",  # Required: a name registered in cfg.models.
        adapter="json",  # "baml" is also supported
        canonicalize=True,
        schema_mode="required",
        output_field_limits={"consolidate": {"decision.rationale": 8192}},
    )
)
```

Register `my-model` in `AppConfig.models` before using this example, or replace it
with a registered model name. There is no built-in selected generation model:
`dspy.model` is required, and task-level `model` values are optional explicit
overrides. The old `dspy.default_model` field has been removed.

- `canonicalize` normalizes unambiguous declared field names and string Literal
  values. It preserves free text, rationale text, and arbitrary mapping keys.
  Structured parsing rejects duplicate, unknown, ambiguous, and missing semantic
  Literal fields, including in required-schema mode with canonicalization disabled;
  defaults must not manufacture an absent consolidation action. Exact JSON fences
  and matching single-output envelopes are accepted, not surrounding prose.
- `schema_mode="required"` requires a model provider that explicitly advertises
  response-schema support and sends the native output schema. Open mappings and
  native nullability are retained. `schema_strict=True` additionally sets the wire
  strict flag and requires `schema_mode="required"`; it does not convert every
  nested object into an all-fields-required, closed-object dialect. Provider
  compatibility must be established separately.
- `validation_retries=1` permits one additional attempt on an adapter parse failure,
  using the same LM, original inputs, categorical validation feedback, and
  temperature zero. The default does not retry. Other transport/runtime failures
  do not trigger this validation retry. Validation retries and enabled DSPy Refine
  are mutually exclusive.
- `output_field_limits` maps task names to declared native-model field paths and
  positive integer maximum Unicode character counts. It requires the declared
  JSON/BAML schema transport. Unknown tasks/fields, Literal fields, nullable/union
  fields, arrays, open-map traversal, and top-level scalar outputs are rejected.
  Unrelated fields and their native defaults remain unchanged. Configuration
  validates primitive values and option compatibility without importing DSPy;
  task and field paths are resolved when constructing a DSPy extractor.

The configured bounds are carried by cloned native Pydantic model annotations,
so schema generation and parser validation use the same types. They are applied
once after building or loading JSON-state or compiled-program artifacts, preserving compiled
instructions and demonstrations. An empty policy leaves the original signature
unchanged. Consolidation rationale remains a returned field; accepted rationale
text is not truncated, rewritten, or removed.

Each extractor snapshots its configuration at construction. Changing the caller's
configuration does not mutate a running extractor; construct a new extractor to
activate a changed policy. Bounded tasks are prepared before any model call so
invalid field paths fail during construction, not halfway through ingestion.

Extraction failures use a categorical outer error, with the original exception
retained as its cause. Applications should avoid exposing chained exception
tracebacks to untrusted callers.

## Transient attempts and logging

Set `dspy.retry` independently of the required model selection:

```yaml
dspy:
  model: my-model  # Explicit registered model; no built-in selection.
  retry:
    max_attempts: 3
    initial_delay_seconds: 1
    max_delay_seconds: 30
```

`max_attempts` includes the initial extraction attempt; its default is **1**
(no automatic retry). Backoff doubles after each failure, capped by
`max_delay_seconds`. Valid provider `Retry-After` hints are honored within that
cap; a longer hint stops retrying rather than retrying too early. Backoff releases
the extraction concurrency slot. Each attempt uses the same model, inputs and
generation settings; failed attempts may still incur provider charges.

Only typed transient server, transport, timeout and rate-limit errors are retried.
Authentication, invalid requests, schema/parse failures and cancellation are not.
Multi-attempt transient retries cannot be combined with validation retries,
enabled DSPy Refine (including nested compiled Refine modules) or streaming extraction.
SDK-internal retries are disabled,
including DSPy LMs pinned in loaded programs, so they cannot multiply this policy.
The policy bounds extraction-task attempts, not individual requests inside an
arbitrary multi-predictor compiled program; external workload limits still apply.

Logger `gluomem.llm.retry` emits one `llm_attempt` event per completed attempt:
`operation_id`, task, attempt number, configured maximum, outcome, elapsed seconds,
safe error category, HTTP status when available, and scheduled retry delay.
Successful attempts are INFO; failed/retrying attempts are WARNING. No prompts,
responses, endpoints, credentials or arbitrary exception messages/tracebacks are
included. Missing HTTP status remains null, not an invented success/error code.
Attempt counts do not imply known token usage for failed requests.

The service/worker JSON logging includes these fields. Embedded applications can
call `gluomem.logging.configure_logging()` or attach their own structured handler
and durable log collector; the library does not silently create log files.

## Verification boundary

Authored unit and loopback SDK tests exercise schema transport, parsing, bounded
rationale, compiled artifacts, and error handling. They are not real-model quality
measurements or proof that a backend honors schema constraints during decoding.
A rationale `maxLength` limits the string value, not whitespace outside that
string, generation-token budgets, or backend termination. Benchmark completion and
codec quality require separate evidence.
