from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, ConfigDict

# JSON-schema primitive type -> acceptable Python type(s). bool is excluded from
# the numeric checks because in Python bool is a subclass of int.
_JSON_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "object": dict,
    "array": list,
}

PermissionTier = Literal["read_only", "state_changing", "destructive"]


def validate_arguments(input_schema: dict, arguments) -> bool:
    """Generic validation of tool-call arguments against a Skill's input schema.

    Reusable across any skill: checks required fields are present, primitive
    types match, and required string fields are non-empty (a present-but-empty
    required string is semantically no argument at all — e.g. app_name="").
    """
    if not isinstance(arguments, dict):
        return False
    properties = input_schema.get("properties", {})
    required = input_schema.get("required", [])
    for field in required:
        if field not in arguments:
            return False
    for field, value in arguments.items():
        spec = properties.get(field)
        if spec is None:
            continue  # extra fields tolerated
        expected = spec.get("type")
        if not _type_ok(expected, value):
            return False
        if expected == "string" and field in required and not value.strip():
            return False
    return True


def _type_ok(expected: str | None, value) -> bool:
    if expected is None:
        return True
    python_type = _JSON_TYPES.get(expected)
    if python_type is None:
        return True  # unknown schema type: don't reject
    if expected in ("integer", "number") and isinstance(value, bool):
        return False  # bool is an int subclass but not a number here
    return isinstance(value, python_type)


class SkillMeta(BaseModel):
    """Everything the registry and the LLM need to know about a skill."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    input_schema: dict
    requires_network: bool
    permission_tier: PermissionTier
    # Shown verbatim when the skill is blocked by offline mode. Optional so
    # non-network skills don't carry a message they can never emit.
    offline_message: str | None = None


class Skill(ABC):
    """A local capability the model can invoke via tool calling."""

    meta: SkillMeta

    @abstractmethod
    async def execute(self, **kwargs) -> str:
        """Run the skill and return a result string for the model (errors included)."""
