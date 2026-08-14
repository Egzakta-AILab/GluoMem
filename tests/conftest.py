"""Root test conftest — neutralize dspy/litellm import-time side effects.

Both env vars MUST be set before the first ``import dspy`` anywhere in the
test session:

- ``DSPY_CACHEDIR``: importing dspy CREATES its disk-cache directory at import
  time (defaults to ``~/.dspy_cache``); point it into a session temp dir so
  tests never touch the user's home.
- ``LITELLM_LOCAL_MODEL_COST_MAP``: litellm otherwise fetches its model-cost
  map over HTTP at import — a network call that would break the offline gate.

``setdefault`` keeps explicit operator overrides working.
"""

from __future__ import annotations

import os
import tempfile

_dspy_cache = tempfile.mkdtemp(prefix="gluomem-dspy-cache-")
os.environ.setdefault("DSPY_CACHEDIR", _dspy_cache)
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
