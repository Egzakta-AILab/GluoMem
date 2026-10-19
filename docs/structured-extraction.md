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
        adapter="json",  # "baml" is also supported
        canonicalize=True,
        schema_mode="required",
        output_field_limits={"consolidate": {"decision.rationale": 8192}},
    )
)
```

- `canonicalize` normalizes unambiguous declared field names and string Literal
  values. It preserves free text, rationale text, and arbitrary mapping keys.
  Duplicate, unknown, ambiguous, and missing semantic Literal fields are rejected;
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
  Unrelated fields and their native defaults remain unchanged.

The configured bounds are carried by cloned native Pydantic model annotations,
so schema generation and parser validation use the same types. They are reapplied
when loading JSON-state or compiled-program artifacts, preserving compiled
instructions and demonstrations. An empty policy leaves the original signature
unchanged. Consolidation rationale remains a returned field; accepted rationale
text is not truncated, rewritten, or removed.

Extraction failures use a categorical outer error, with the original exception
retained as its cause. Applications should avoid exposing chained exception
tracebacks to untrusted callers.

## Verification boundary

Authored unit and loopback SDK tests exercise schema transport, parsing, bounded
rationale, compiled artifacts, and error handling. They are not real-model quality
measurements or proof that a backend honors schema constraints during decoding.
A rationale `maxLength` limits the string value, not whitespace outside that
string, generation-token budgets, or backend termination. Benchmark completion and
codec quality require separate evidence.
