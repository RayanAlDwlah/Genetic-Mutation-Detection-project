"""ESM2 model loading helpers."""

from transformers import AutoModel, AutoTokenizer


def load_esm2(model_name: str = "facebook/esm2_t6_8M_UR50D"):
    """Load tokenizer and model for ESM2 embeddings."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model
