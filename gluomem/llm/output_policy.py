"""Apply output bounds to native signatures after program construction or loading."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Annotated, Any, cast

import dspy
from pydantic import BaseModel, Field, create_model

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


def _constrain(annotation: Any, parts: list[str], limit: int) -> Any:
    if not parts:
        if annotation is not str:
            raise ValueError("output limit requires a native string field")
        return Annotated[str, Field(max_length=limit)]
    if not isinstance(annotation, type) or not issubclass(annotation, BaseModel):
        raise ValueError("output limit path must traverse declared model fields")
    name, *rest = parts
    if name not in annotation.model_fields:
        raise ValueError("unknown output limit path")
    field = deepcopy(annotation.model_fields[name])
    child = _constrain(field.annotation, rest, limit)
    return cast("Callable[..., type[BaseModel]]", create_model)(
        annotation.__name__,
        __base__=annotation,
        __doc__=annotation.__doc__,
        **{name: (child, field)},
    )


def configured_signature(
    signature: type[dspy.Signature], limits: Mapping[str, int]
) -> type[dspy.Signature]:
    """Clone bounded native model fields, preserving all other field metadata.

    Arrays, unions, open mappings and top-level scalar outputs are unsupported.
    An empty policy returns the original signature.
    """
    if not limits:
        return signature
    fields: dict[str, tuple[Any, Any]] = {
        name: (field.annotation, deepcopy(field)) for name, field in signature.fields.items()
    }
    for path, limit in sorted(limits.items()):
        if not isinstance(path, str) or not path or type(limit) is not int or limit <= 0:
            raise ValueError(
                "output limits require dot paths and positive integer character counts"
            )
        name, *parts = path.split(".")
        if name not in signature.output_fields:
            raise ValueError("unknown output limit path")
        if not parts:
            raise ValueError("output limits require a string inside a native model field")
        annotation, field = fields[name]
        fields[name] = (_constrain(annotation, parts, limit), field)
    return dspy.make_signature(fields, signature.instructions, signature_name=signature.__name__)


def apply_output_limits(module: dspy.Module, limits: Mapping[str, int]) -> dspy.Module:
    """Finalize a loaded or newly built program before it is exposed to execution."""
    if limits:
        predictors = module.predictors()
        if not predictors:
            raise ValueError("bounded task requires native typed predictors")
        for predictor in predictors:
            if predictor.signature is None:
                raise ValueError("bounded task requires native typed predictors")
            predictor.signature = configured_signature(predictor.signature, limits)
    return module
