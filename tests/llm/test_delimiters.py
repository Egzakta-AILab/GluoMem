import json

import pytest
from dspy.utils.exceptions import AdapterParseError

from gluomem.llm.router import _make_adapter
from gluomem.llm.signatures import TASK_SIGNATURES


@pytest.mark.parametrize("wrapper", ["```json\n{}\n```", "<decision>\n{}\n</decision>"])
def test_exact_structural_delimiters_only(wrapper):
    adapter = _make_adapter("json", canonicalize=True)
    body = json.dumps({"action": "ADD", "target_frame_id": None, "rationale": "Exact quote"})
    assert (
        adapter.parse(TASK_SIGNATURES["consolidate"], wrapper.format(body))["decision"].rationale
        == "Exact quote"
    )
    for extra in [
        "private outside " + wrapper.format(body),
        wrapper.format(body) + " trailing",
        "<unknown>" + body + "</unknown>",
    ]:
        with pytest.raises(AdapterParseError):
            adapter.parse(TASK_SIGNATURES["consolidate"], extra)
