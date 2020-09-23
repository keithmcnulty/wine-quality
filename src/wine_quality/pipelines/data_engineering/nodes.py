import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

def _create_categories(x):
    if x >= 6:
        return "High"
    elif x >= 4:
        return "Medium"
    else:
        return "Low"


def quality_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a column converting wine rating into a quality category
    :param df: Pandas Dataframe
    :return: Pandas Dataframe
    """

    df['quality_category'] = df['quality'].apply(_create_categories)
    return df

def concat_dfs(red: pd.DataFrame, white: pd.DataFrame) -> pd.DataFrame:
    """
    Combine red and white dataframes, add type column, drop NAs
    :param red: Pandas DataFrame
    :param white: Pandas DataFrame
    :return: Pandas DataFrame
    """

    red['type'] = 'red'
    white['type'] = 'white'
    return pd.concat([red, white]).dropna()

def create_dummies(df: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """
    Create dummy variables from categorical columns
    :param df: Pandas DataFrame
    :param parameters: columns to dummy
    :return: Pandas DataFrame
    """
    for col in parameters["dummy_cols"]:
        binarizer = LabelBinarizer()
        dummies = binarizer.fit_transform(df[col])
        if len(binarizer.classes_)==2:
            dummy_df = pd.DataFrame(dummies, columns=binarizer.classes_[1:])
        else:
            dummy_df = pd.DataFrame(dummies, columns=binarizer.classes_)
        df = pd.concat([df.drop(col, axis=1), dummy_df], axis=1)

    return df



