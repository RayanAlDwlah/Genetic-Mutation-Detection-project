"""ESM-2 protein language model integration for mutation classification.

NOTE: This module is a structural placeholder reserved for Phase 2 (post-midterm).
The full implementation will include:
  - Extraction of per-residue embeddings from ESM-2 (facebook/esm2_t6_8M_UR50D)
  - Fine-tuning the model on ClinVar missense variants using LoRA / full fine-tuning
  - A classification head on top of the mutation-site embedding
  - Evaluation against the XGBoost baseline using identical gene-level splits

Data requirement: UniProt human proteome FASTA (data/raw/uniprot/human_proteome.fasta)
"""

from transformers import AutoModel, AutoTokenizer


def load_esm2(model_name: str = "facebook/esm2_t6_8M_UR50D"):
    """Load the ESM-2 tokenizer and encoder from HuggingFace.

    Default model: esm2_t6_8M_UR50D — the smallest ESM-2 variant (8M parameters),
    suitable for local development. Replace with esm2_t33_650M_UR50D for
    production-grade embeddings (requires ~4 GB GPU memory).

    Returns:
        tokenizer: ESM-2 BPE tokenizer for amino acid sequences.
        model: Pre-trained ESM-2 encoder (all weights frozen by default).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model
