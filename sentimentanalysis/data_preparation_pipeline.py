import json
import logging
import os
import zipfile

import pandas as pd
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def extract_data(file_path: str) -> pd.DataFrame:
    """
    Extracts data from a given JSONL file path.

    This function unzips the file and reads the JSONL data into a pandas DataFrame.

    Parameters
    ----------
    file_path : str
        path to the zipped JSONL file

    Returns
    -------
    pd.DataFrame
        loaded data
    """
    # Open the zipped file
    with zipfile.ZipFile(file_path, "r") as zf:
        # Open the JSONL file inside the zip
        with zf.open(zf.namelist()[0], "r") as f:
            # Read the JSONL data into a pandas DataFrame
            data: pd.DataFrame = pd.read_json(f, lines=True)
    # Return the loaded data
    return data


def transform(data: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the data by adding a sentiment column based on the rating.

    This function evaluates each rating in the data and assigns a sentiment
    label: "positive" for ratings 4 and above, "negative" for ratings 2 and
    below, and "neutral" for ratings 3.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing a "rating" column to be transformed.

    Returns
    -------
    pd.DataFrame
        The DataFrame with an added "sentiment" column.
    """
    data["sentiment"] = data["rating"].apply(
        lambda x: "positive" if x >= 4 else ("negative" if x <= 2 else "neutral")
    )
    return data


def load(data: pd.DataFrame, data_path: str) -> None:
    """
    Saves the data to a specified CSV file path.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to be saved.
    data_path : str
        The path where the CSV file will be saved.

    Returns
    -------
    None
    """
    data.to_csv(data_path)


def main():
    """
    Main function to prepare and save sentiment analysis data.

    This function loads the environment variables, reads the configuration
    file, extracts raw data from a specified path, transforms the data by
    adding a sentiment column, and saves the final data to a specified path.
    """

    # Load environment variables
    load_dotenv()
    env = os.getenv("ENVIRONMENT")
    config_path = os.getenv("CONFIG_PATH")

    # Load configuration based on the environment
    with open(config_path, "r") as fin:
        configuration = json.load(fin)
    configuration = configuration[env]
    raw_file_path = configuration["raw_data_path"]
    data_path = configuration["data_path"]

    # Extract raw data
    raw_data = extract_data(raw_file_path)

    # Transform data by adding sentiment column
    final_data = transform(raw_data)

    # Save the transformed data
    load(final_data, data_path)


if __name__ == "__main__":
    main()
