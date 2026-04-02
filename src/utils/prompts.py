"""Prompt template loader.

Loads prompt text from ``config/prompts/*.txt`` files and substitutes
named placeholders at call time using Python's str.format_map.
"""

from pathlib import Path

PROMPTS_DIR = Path(__file__).parent.parent.parent / "config" / "prompts"


def load_prompt(name: str, **kwargs: str) -> str:
    """Load a prompt template by name and fill in placeholders.

    Prompt files live in ``config/prompts/{name}.txt``.  Placeholders use
    ``{key}`` Python format-string syntax.

    Args:
        name: Prompt file stem (e.g. ``"query_analysis"``).
        **kwargs: Placeholder values to substitute into the template.

    Returns:
        Fully rendered prompt string.

    Raises:
        FileNotFoundError: If no prompt file with the given name exists.
        KeyError: If a required placeholder is missing from kwargs.
    """
    pass


def list_prompts() -> list[str]:
    """Return the names of all available prompt templates.

    Returns:
        Sorted list of prompt file stems.
    """
    pass
