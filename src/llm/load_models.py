"""Chargement générique des modèles de langage."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_device() -> str:
    """Détermine le matériel disponible."""

    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def load_model(
        model_id: str,
        load_in_4bit: bool = False,
):
    """
    Charge un modèle causal et son tokenizer.

    Args:
        model_id: identifiant Hugging Face du modèle.
        load_in_4bit: active la quantification 4 bits sur CUDA.

    Returns:
        model, tokenizer
    """

    device = get_device()

    if load_in_4bit and device != "cuda":
        raise ValueError(
            "La quantification 4 bits avec bitsandbytes "
            "nécessite un GPU CUDA."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "dtype": "auto",
    }

    if device == "cuda":
        model_kwargs["device_map"] = "auto"

        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        **model_kwargs,
    )

    if device != "cuda":
        model = model.to(device)

    model.eval()

    return model, tokenizer