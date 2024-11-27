from os import listdir
from os.path import isfile, join
import json
import pandas as pd
import numpy as np
import datetime
from dateutil.parser import parse


def convert_json_to_df(dir):
    """
    Turn json files into merged dataframe

    Args:
        dir(str): directory where json files are saved
    Returns:
        dfs(pd.DataFrame): dataframe which are merged from json files
    """
    files = [f for f in listdir(dir) if isfile(join(dir, f))]
    
    df_list = []
    for f in files:
        with open(join(dir, f), 'r', encoding="utf-8") as file:
            df = pd.DataFrame(json.loads(line) for line in f)
        df_list.append(df)
    
    return pd.concat(df_list, ignore_index=True)
    
def convert_date_format(df, date_column, convert_mode):
    if convert_mode == "seconds_to_datetime":
        df[date_column] = pd.to_datetime(df[date_column], unit='s')
    
    if convert_mode == "from_javatime":
        def invert_javatime(javatime):
            seconds = javatime / 1000
            sub_seconds = (javatime % 1000.0) / 1000.0
            date = datetime.datetime.fromtimestamp(seconds, tz=datetime.timezone.utc)
            return date + datetime.timedelta(seconds=sub_seconds)
        df[date_column] = df[date_column].apply(lambda x: invert_javatime(x))
    
    if convert_mode == "custom_publish_date":
        df[date_column] = df[date_column].astype(str)
        df[date_column] = pd.to_datetime(df[date_column].str[:8].str.strip(),
                                         format="%Y%m%d",
                                         errors="coerce")
        df[date_column].fillna(pd.Timestamp('2099-12-31'), inplace=True)

        start_date = pd.Timestamp('1800-01-01')
        end_date = pd.Timestamp('2020-12-31')
        df = df[
            (df[date_column] >= start_date) &
            (df[date_column] <= end_date)
        ]

    return df

def convert_to_null_value(df, column_name, conditions):
    """
    Replaces values in a specified DataFrame column with NaN based on given conditions.

    Parameters:
    df (pd.DataFrame): The target DataFrame.
    column_name (str): The name of the column to process.
    conditions (list of functions): A list of functions representing conditions. Each function takes a single argument and returns True if the condition is met.

    Returns:
    pd.DataFrame: The DataFrame with specified values replaced by NaN.
    """
    for condition in conditions:
        df[column_name] = df[column_name].apply(
            lambda x: np.nan if condition(x) else x
        )
    return df

def convert_to_discrete_col(df, column_name, convert_mode,
                            cut_list=None, cut_num=None, 
                            default_date=None, days_list=None, date_column=None):
    """
    Convert a specified column in the DataFrame to discrete categories based on the chosen mode.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column_name (str): The name of the column to convert.
    convert_mode (str): The mode of conversion ('cut', 'qcut', or 'date_range').
    cut_list (list, optional): Bin edges for 'cut' mode.
    cut_num (int, optional): Number of quantiles for 'qcut' mode.
    default_date (str, optional): Reference date for 'date_range' mode.
    days_list (list, optional): List of day intervals for 'date_range' mode.
    date_column (str, optional): Name of the new column to store date range categories.

    Returns:
    pd.DataFrame: The DataFrame with the converted column.
    """
    if convert_mode == "cut":
        df[column_name] = pd.cut(df[column_name], cut_list,
                                labels=[str(i) for i in range(len(cut_list) - 1)], right=False)
    
    if convert_mode == "qcut":
        df[column_name] = pd.qcut(df[column_name], cut_num, labels=[str(i) for i in range(cut_num)])

    if convert_mode == "date_range":
        def date_range(days=None, default_date="2020-05-01"):
            default_date = pd.to_datetime(default_date)
            delta = datetime.timedelta(days=days)
            before_date = default_date - delta
            after_date = default_date + delta
            return pd.date_range(before_date, after_date, periods=2)
        
        df[date_column] = len(days_list)
        for i, days in enumerate(days_list):
            date_rng = date_range(days=days, default_date=default_date)
            df.loc[df[column_name].isin(date_rng), date_column] = i

    return df
