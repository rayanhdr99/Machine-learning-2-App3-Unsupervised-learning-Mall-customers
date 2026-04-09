# data_loader.py - loads the mall customers csv and validates it
import logging
import pandas as pd

logger = logging.getLogger(__name__)

# these are the columns we expect to find in the dataset
REQUIRED_COLUMNS = ["Customer_ID", "Gender", "Age", "Annual_Income", "Spending_Score"]


# load the mall customers dataset from a csv file
def load_data(filepath: str) -> pd.DataFrame:
    logger.info("Loading data from: %s", filepath)

    # try reading the csv file
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError as e:
        logger.error("Data file not found: %s", filepath)
        raise FileNotFoundError(f"Data file not found: {filepath}") from e
    except Exception as e:
        logger.error("Failed to read CSV: %s", e)
        raise

    # make sure the dataframe isn't empty
    if df.empty:
        raise ValueError("The loaded dataset is empty.")

    # check that all the required columns are present
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    logger.info("Data loaded. Shape: %s", df.shape)
    return df
