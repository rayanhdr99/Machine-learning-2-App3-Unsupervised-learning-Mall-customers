# preprocessor.py - picks the features we need for KMeans clustering
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# extract the columns we want to use for clustering (income and spending score by default)
def get_features(df: pd.DataFrame, include_age: bool = False) -> pd.DataFrame:
    # we only need Annual_Income and Spending_Score for clustering
    cols = ["Annual_Income", "Spending_Score"]

    # optionally include Age as a third feature
    if include_age:
        cols = ["Age", "Annual_Income", "Spending_Score"]

    logger.info("Selected clustering features: %s", cols)

    # return a copy so we don't accidentally modify the original dataframe
    return df[cols].copy()
