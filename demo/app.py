
"""Application Gradio de démonstration du classificateur de dialectes."""

import html
import os

import gradio as gr

from src.llm.prompts import build_prompt
from src.llm.inference import generate_response
from src.llm.evaluation import parse_prediction
from src.llm.load_models import load_model


COLORS = [
    "#dbeafe",
    "#dcfce7",
    "#fef3c7",
    "#fce7f3",
    "#ede9fe",
    "#ffedd5",
]


def display_tokens(tokenizer, text: str) -> tuple[str, int]:
    """Affiche les tokens avec des couleurs et retourne leur nombre."""

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    colored_tokens = []

    for index, token in enumerate(tokens):
        color = COLORS[index % len(COLORS)]
        safe_token = html.escape(token)

        colored_tokens.append(
            f'<span style="background-color:{color}; '
            f'padding:4px; margin:2px; border-radius:5px; '
            f'display:inline-block;">{safe_token}</span>'
        )

    return " ".join(colored_tokens), len(token_ids)


def predict_dialect(
        text: str,
        cot: bool,
        model,
        tokenizer,
) -> tuple[str, str, str, int]:
    """Classe une phrase et affiche sa tokenisation."""

    if not isinstance(text, str) or not text.strip():
        raise gr.Error("Veuillez saisir une phrase.")

    prompt = build_prompt(
        text=text,
        cot=cot,
    )

    response = generate_response(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=100 if cot else 10,
    )

    prediction = parse_prediction(response)

    if prediction is None:
        prediction_display = "Réponse invalide ou non reconnue."
    else:
        prediction_display = prediction

    token_html, token_count = display_tokens(tokenizer, text)

    return (
        prediction_display,
        token_html,
        response,
        token_count,
    )



def build_demo(model, tokenizer, mini_gpt=None):
    """Construit l'interface Gradio."""

    with gr.Blocks(title="Les LLM parlent-ils maghrébin ?") as demo:
        gr.Markdown(
            """
            # Les LLM parlent-ils maghrébin ?

            Saisissez une phrase en arabe ou en Arabizi
            pour identifier son dialecte.

            Dialectes possibles : algérien, marocain,
            tunisien et arabe standard.
            """
        )

        text_input = gr.Textbox(
            label="Phrase à analyser",
            placeholder="Exemple : واش راك؟",
            lines=3,
        )

        cot_input = gr.Checkbox(
            label="Activer le mode Chain-of-Thought (CoT)",
            value=False,
        )

        submit_button = gr.Button(
            "Analyser la phrase",
            variant="primary",
        )

        prediction_output = gr.Textbox(
            label="Dialecte détecté",
            interactive=False,
        )

        token_count_output = gr.Number(
            label="Nombre de tokens",
            precision=0,
            interactive=False,
        )

        token_output = gr.HTML(
            label="Visualisation des tokens",
        )

        response_output = gr.Textbox(
            label="Réponse brute du modèle",
            lines=5,
            interactive=False,
        )

        submit_button.click(
            fn=lambda text, cot: predict_dialect(
                text, cot, model, tokenizer
            ),
            inputs=[text_input, cot_input],
            outputs=[
                prediction_output,
                token_output,
                response_output,
                token_count_output,
            ],
        )

        gr.Markdown("## Génération de texte avec Mini-GPT")

        if mini_gpt is None:
            gr.Markdown(
                "*Le module Mini-GPT sera disponible "
                "lorsque son entraînement sera terminé.*"
            )
        else:
            gr.Markdown(
                "Le Mini-GPT est chargé et prêt à être intégré."
            )

    return demo



def main():
    """Charge le modèle configuré et lance la démonstration."""

    model_id = os.getenv("DARIJA_MODEL_ID")

    if not model_id:
        raise RuntimeError(
            "Définis la variable d'environnement DARIJA_MODEL_ID "
            "avec l'identifiant Hugging Face du modèle."
        )

    load_in_4bit = (
            os.getenv("DARIJA_LOAD_IN_4BIT", "false").lower() == "true"
    )

    model, tokenizer = load_model(
        model_id=model_id,
        load_in_4bit=load_in_4bit,
    )

    demo = build_demo(model, tokenizer)

    demo.launch()


if __name__ == "__main__":
    main()
