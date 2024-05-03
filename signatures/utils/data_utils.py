import pandas as pd
# import polars as pl

from signatures import config

def read_cosmic_signatures(input_path: str = config.COSMIC_FILE_PATH) -> pd.DataFrame:
    return pd.read_csv(input_path, sep="\t").set_index("Type")
