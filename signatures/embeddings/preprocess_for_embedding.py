import logging
import os

import click
import pandas as pd

from signatures.utils import sequence_utils


logging.basicConfig(level=logging.INFO)


@click.command()
@click.option("--data_dir", default="resources/data/icgc")
def preprocess_for_embeddings(**kwargs):
    data_dir = kwargs["data_dir"]
    logging.info(f"Preprocessing data in directory: {data_dir}")
    for filename in os.listdir(data_dir):
        if filename.endswith(".tsv"):
            logging.info(f"Processing file: {filename}")
            df = pd.read_csv(os.path.join(data_dir, filename), sep="\t")
            df = sequence_utils.add_pre_post_seq_columns(df)
            df.to_csv(os.path.join(data_dir, filename), sep="\t", index=False)
            logging.info(f"Processed file: {filename}")


if __name__ == "__main__":
    preprocess_for_embeddings()
