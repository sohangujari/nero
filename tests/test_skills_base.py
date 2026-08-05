import pytest
from pydantic import ValidationError

from nero.skills.base import Skill, SkillMeta, validate_arguments

SCHEMA = {
    "type": "object",
    "properties": {"app_name": {"type": "string"}},
    "required": ["app_name"],
}


class TestSkillMeta:
    def test_holds_all_fields(self):
        meta = SkillMeta(
            name="open_app",
            description="Opens an app.",
            input_schema=SCHEMA,
            requires_network=False,
            permission_tier="state_changing",
        )
        assert meta.name == "open_app"
        assert meta.requires_network is False
        assert meta.permission_tier == "state_changing"
        assert meta.offline_message is None

    def test_rejects_unknown_permission_tier(self):
        with pytest.raises(ValidationError):
            SkillMeta(
                name="x", description="d", input_schema={},
                requires_network=False, permission_tier="sudo",
            )

    def test_rejects_unknown_keys(self):
        with pytest.raises(ValidationError):
            SkillMeta(
                name="x", description="d", input_schema={},
                requires_network=False, permission_tier="read_only", bogus=1,
            )


class TestSkillContract:
    def test_subclass_must_implement_execute(self):
        class Incomplete(Skill):
            meta = SkillMeta(
                name="x", description="d", input_schema={},
                requires_network=False, permission_tier="read_only",
            )

        with pytest.raises(TypeError):
            Incomplete()


class TestValidateArguments:
    def test_valid(self):
        assert validate_arguments(SCHEMA, {"app_name": "Safari"}) is True

    def test_missing_required_field(self):
        assert validate_arguments(SCHEMA, {}) is False

    def test_empty_required_string(self):
        assert validate_arguments(SCHEMA, {"app_name": "   "}) is False

    def test_wrong_type(self):
        assert validate_arguments(SCHEMA, {"app_name": 5}) is False

    def test_non_dict_arguments(self):
        assert validate_arguments(SCHEMA, "nope") is False

    def test_extra_fields_tolerated(self):
        assert validate_arguments(SCHEMA, {"app_name": "Safari", "x": 1}) is True

    def test_generic_required_int(self):
        schema = {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        }
        assert validate_arguments(schema, {"count": 3}) is True
        assert validate_arguments(schema, {"count": "3"}) is False
        assert validate_arguments(schema, {}) is False
