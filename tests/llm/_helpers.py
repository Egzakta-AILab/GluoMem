"""Authored transport doubles shared by LLM contract tests."""

from dspy.utils.dummies import DummyLM


class Scripted(DummyLM):
    def __init__(self, responses):
        super().__init__([])
        self.responses = iter(responses)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        value = next(self.responses)
        if isinstance(value, BaseException):
            raise value
        return [value]

    async def acall(self, prompt=None, *, messages=None, **kwargs):
        return self(prompt=prompt, messages=messages, **kwargs)
