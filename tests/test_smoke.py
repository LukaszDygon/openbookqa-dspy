from openbookqa_dspy.config import Settings


def test_settings_defaults() -> None:
    s = Settings.load()
    assert isinstance(s.model, str)
