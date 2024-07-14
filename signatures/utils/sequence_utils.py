import re

import pandas as pd


def add_pre_post_seq_columns(df: pd.DataFrame) -> pd.DataFrame:
    assert "mutation_type" in df.columns, "DataFrame must contain a 'mutation_type' column!"
    df[["seq_pre", "seq_post"]] = pd.DataFrame(
        df["mutation_type"].apply(extract_pre_post_sequences).tolist(), 
        index=df.index
    )
    return df


def extract_pre_post_sequences(mutation_type: str) -> tuple[str, str]:
    left_context, mutation, right_context = re.split("\[|\]", mutation_type)
    pre, post = mutation.split(">")
    pre_seq, post_seq = f"{left_context}{pre}{right_context}", f"{left_context}{post}{right_context}"
    return pre_seq, post_seq
