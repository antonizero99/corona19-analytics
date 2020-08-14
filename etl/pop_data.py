import pandas as pd
import os


DATA_LOCATION = '\\data\\'


def get_fact_pop_clean() -> pd.DataFrame:
    df_pop = __load_pop_data()
    df_pop_clean = __clean_pop_data(df_pop)
    return df_pop_clean


def __load_pop_data() -> pd.DataFrame:
    df_pop = pd.read_csv(os.path.dirname(os.getcwd()) + DATA_LOCATION + 'population.csv')
    return df_pop


def __clean_pop_data(df_pop: pd.DataFrame) -> pd.DataFrame:
    con_len_above_3 = df_pop.Code.str.len() == 3
    con_not_na = df_pop.Code.notna()
    df_pop_clean = df_pop.loc[con_len_above_3 & con_not_na]
    return df_pop_clean