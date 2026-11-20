"""Fonctions d'inférence pour les modèles de langage."""

import torch


def generate_response(
        model,
        tokenizer,
        prompt: str,
        max_new_tokens: int = 10,
) -> str:
    """
    Envoie un prompt à un modèle et récupère sa réponse.
    """

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    # Place les entrées sur le même appareil que le modèle.
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # On récupère uniquement les nouveaux tokens générés.
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return response.strip()