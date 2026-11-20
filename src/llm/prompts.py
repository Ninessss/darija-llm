
"""Prompts pour l'expérience de classification des dialectes."""

LABELS = [
    "algerian",
    "moroccan",
    "tunisian",
    "standard_arabic",
]

TASK_INSTRUCTIONS = """
You are an Arabic dialect classification assistant.

Your task is to identify the variety of Arabic used in the input text.

Choose exactly one of these labels:
- algerian: Algerian Darija
- moroccan: Moroccan Darija
- tunisian: Tunisian Darija
- standard_arabic: Modern Standard Arabic

Rules:
- Choose only one label from the list.
- Do not translate or rewrite the input.
- If the text is ambiguous, choose the most likely label.
"""

COT_INSTRUCTIONS = """
Briefly state the linguistic clues supporting your classification.
Then provide exactly one final label using this format:
Final label: <label>

Do not provide multiple candidate labels.
"""


def build_prompt(
        text: str,
        examples: list[dict] | None = None,
        cot: bool = False,
) -> str:
    """Construit un prompt zero-shot, few-shot ou CoT."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Le texte à classifier ne peut pas être vide.")

    examples = examples or []

    if len(examples) not in [0, 4, 12, 20]:
        raise ValueError(
            "Le nombre d'exemples doit être 0, 4, 12 ou 20 "
            "(0, 1, 3 ou 5 par dialecte)."
        )

    prompt = TASK_INSTRUCTIONS.strip() + "\n\n"

    if examples:
        prompt += "Here are some labeled examples:\n\n"

        for example in examples:
            if "text" not in example or "label" not in example:
                raise ValueError(
                    "Chaque exemple doit contenir text et label."
                )

            example_text = example["text"]
            example_label = example["label"]

            if not isinstance(example_text, str) or not example_text.strip():
                raise ValueError("Le texte d'un exemple est vide.")

            if example_label not in LABELS:
                raise ValueError(f"Label invalide : {example_label}")

            prompt += (
                f"Input: {example_text.strip()}\n"
                f"Label: {example_label}\n\n"
            )

    if cot:
        prompt += COT_INSTRUCTIONS.strip() + "\n\n"
        prompt += f"Input: {text.strip()}\nFinal label:"
    else:
        prompt += (
            "Return only the label, without any explanation.\n\n"
            f"Input: {text.strip()}\nLabel:"
        )

    return prompt
