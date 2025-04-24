from pathlib import Path

import pandas as pd


def merge_and_save_dataframes(*dfs, path: str, name: str):
    """
    Merges multiple dataframes and saves the result to an Excel file.
    """
    merged_df = pd.concat(*dfs, ignore_index=True)
    merged_df.to_excel(Path(path) / name, index=False)
