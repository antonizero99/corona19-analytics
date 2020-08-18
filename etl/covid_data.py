import requests
import pathlib
import pandas as pd
import numpy as np
import json

DATA_LOCATION = 'data'
COVID_DATA_OWID = 'owid-covid-data.csv'
COVID_DATA_JHU_RECOVER = 'time_series_covid19_recovered_global.csv'
COVID_DATA_JHU_DEATH = 'time_series_covid19_deaths_global.csv'
COVID_DATA_JHU_CONFIRM = 'time_series_covid19_confirmed_global.csv'

GEOJSON_WORLD_MED_SOL = 'world_geo_json.json'

MAPPING_LOCATION_OWID = 'mapping_location_covid.csv'
MAPPING_LOCATION_JHU = 'mapping_location_recover.csv'


def download_data(url: str, file_name: str):
    try:
        download = requests.get(url)
        save_file = pathlib.Path.cwd().parent / DATA_LOCATION / file_name
        open(save_file, 'wb').write(download.content)
    except requests.exceptions.Timeout:
        print('Download {} timed out'.format(file_name))
    except requests.exceptions.RequestException:
        print('Download {} error'.format(file_name))



def update_data():
    source_url_covid = 'https://covid.ourworldindata.org/data/owid-covid-data.csv'
    source_url_recover = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data' \
                         '/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
    source_url_death = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data' \
                       '/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
    source_url_confirmed = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data' \
                           '/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'

    download_data(url=source_url_covid, file_name=COVID_DATA_OWID)
    download_data(url=source_url_recover, file_name=COVID_DATA_JHU_RECOVER)
    download_data(url=source_url_death, file_name=COVID_DATA_JHU_DEATH)
    download_data(url=source_url_confirmed, file_name=COVID_DATA_JHU_CONFIRM)


def get_dim_location() -> pd.DataFrame:
    df = __load_csv_data(file_name=COVID_DATA_OWID)
    df_geo = df[['iso_code', 'continent', 'location',
                 'population', 'population_density', 'gdp_per_capita', 'hospital_beds_per_thousand', 'life_expectancy']] \
        .drop_duplicates(subset=['iso_code', 'continent', 'location'], keep='first', ignore_index=True)
    df_geo['iso_code'].replace('OWID_KOS', 'KOS', inplace=True)
    df_geo = df_geo[~df_geo.iso_code.isna()]
    return df_geo


def __load_csv_data(file_name: str) -> pd.DataFrame:
    df = pd.read_csv(pathlib.Path.cwd().parent / DATA_LOCATION / file_name)
    return df


def __load_json_data(file_name: str) -> dict:
    with open(pathlib.Path.cwd().parent / DATA_LOCATION / file_name) as json_file:
        dc_json = json.load(json_file)
    return dc_json


def etl_geojson_data() -> pd.DataFrame:
    dc_geojson = __load_json_data(GEOJSON_WORLD_MED_SOL)
    lst_world_geo_json = [item['properties'] for item in dc_geojson['features']]
    df_world_geo_json = pd.DataFrame(lst_world_geo_json)
    return df_world_geo_json


def etl_covid_data_jhu(jhu_df: pd.DataFrame, total_col_label: str, trans_col_label: str) -> pd.DataFrame:
    df_mapping_location_jhu = __load_csv_data(file_name=MAPPING_LOCATION_JHU)
    df_geo = get_dim_location()

    # Transform jhu_df
    # Add World rows in jhu_df
    jhu_df = jhu_df.append(jhu_df.sum(numeric_only=True, axis=0).append(pd.Series(['World'], index=['Country/Region'])),
                           ignore_index=True)
    # Create Mapping Key
    jhu_df['Mapping Key'] = jhu_df['Country/Region'].replace(np.nan, '') + jhu_df['Province/State'].replace(np.nan, '')
    # Join with flat file to get location by field Mapping Key
    jhu_df_new_location = pd.merge(jhu_df, df_mapping_location_jhu[['Mapping Key', 'Mapping location']], how='left',
                                   on='Mapping Key')
    jhu_df_new_location['location'] = jhu_df_new_location['Mapping location']

    # Remove unnecessary fields
    jhu_df_new_location.drop(
        labels=['Province/State', 'Country/Region', 'Lat', 'Long', 'Mapping Key', 'Mapping location'], axis=1,
        inplace=True)

    # Group by new location field
    jhu_df_new_location = jhu_df_new_location.groupby(by='location', as_index=False).sum()

    # Unpivot
    jhu_df = jhu_df_new_location.melt(id_vars='location', var_name='date', value_name=total_col_label)

    # Convert data type to datetime
    jhu_df['date'] = pd.to_datetime(jhu_df['date'], format='%m/%d/%y')

    # Add new column: transaction column
    jhu_df = jhu_df.sort_values(by=['location', 'date'], ascending=[1, 1]).reset_index(drop=True)
    jhu_df[trans_col_label] = jhu_df.groupby('location').diff()[total_col_label]

    # Add iso_code, continent columns
    jhu_df = pd.merge(jhu_df, df_geo, how='left', on='location')

    # Add group by continent data
    jhu_df = jhu_df.append(jhu_df[['continent', 'date', total_col_label, trans_col_label]].
                           groupby(by=['continent', 'date'], as_index=False).sum().
                           rename(columns={'continent': 'location'}), ignore_index=True)

    # Done ETL jhu_df

    return jhu_df


def get_fact_confirm() -> pd.DataFrame:
    df_confirm = __load_csv_data(file_name=COVID_DATA_JHU_CONFIRM)
    df_confirm = etl_covid_data_jhu(df_confirm, total_col_label='confirm', trans_col_label='new_confirm')
    return df_confirm


def get_fact_death() -> pd.DataFrame:
    df_death = __load_csv_data(file_name=COVID_DATA_JHU_DEATH)
    df_death = etl_covid_data_jhu(df_death, total_col_label='death', trans_col_label='new_death')
    return df_death


def get_fact_recover() -> pd.DataFrame:
    df_recover = __load_csv_data(file_name=COVID_DATA_JHU_RECOVER)
    df_recover = etl_covid_data_jhu(df_recover, total_col_label='recover', trans_col_label='new_recover')
    return df_recover


def get_fact_jhu_full() -> pd.DataFrame:
    df_confirm = get_fact_confirm()
    df_death = get_fact_death()
    df_recover = get_fact_recover()

    df_full = pd.merge(df_confirm, df_death[['location', 'date', 'death', 'new_death']], how='inner',
                       on=['location', 'date'])
    df_full = pd.merge(df_full, df_recover[['location', 'date', 'recover', 'new_recover']], how='inner',
                       on=['location', 'date'])

    # Add new column: Number of current active cases
    df_full['active'] = df_full['confirm'] - df_full['death'] - df_full['recover']

    # Add countries' properties
    df_geojson = get_dim_countries_details()
    df_full = pd.merge(df_full, df_geojson[['adm0_a3', 'economy', 'income_grp']], how='left',
                       left_on='iso_code', right_on='adm0_a3')

    # Sort whole data set by location and date
    df_full.sort_values(by=['location', 'date'], axis=0, ascending=[1, 1], inplace=True, ignore_index=True)
    return df_full


def get_dim_countries_details() -> pd.DataFrame:
    df_geojson = etl_geojson_data()
    return df_geojson
