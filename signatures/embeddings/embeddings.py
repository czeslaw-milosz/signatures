import logging
from typing import Any

import torch
import transformers
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig


def get_dna_embeddings(dna_batch: torch.Tensor, model: torch.Module, tokenizer: Any, embedding_type: str = "mean") -> torch.Tensor:
    assert embedding_type in {"mean", "max"}, f"Unsupported embedding type: {embedding_type}"
    tokenized_batch = tokenizer(dna_batch, return_tensors="pt")["input_ids"]
    hidden_states = model(tokenized_batch)[0]  # shape: [batch_size, seq_length, embedding_dim]
    return torch.mean(hidden_states, dim=1) if embedding_type == "mean" else torch.max(hidden_states, dim=1)[0]



def load_dnabert2(force_download: bool = False):
    config = BertConfig.from_pretrained(
        "zhihan1996/DNABERT-2-117M",
        force_download=force_download
    )
    model = AutoModel.from_pretrained(
        "zhihan1996/DNABERT-2-117M",
        trust_remote_code=True, 
        config=config, force_download=force_download
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "zhihan1996/DNABERT-2-117M",
        trust_remote_code=True,
        force_download=force_download
    )
    return model, tokenizer



    

