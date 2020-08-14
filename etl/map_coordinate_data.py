import json
import os


DATA_LOCATION = '\\data\\'


def get_world_geo_json() -> dict:
    geo_json = __load_world_geo_json_data()
    return geo_json


def __load_world_geo_json_data() -> dict:
    with open(os.path.dirname(os.getcwd()) + DATA_LOCATION + 'world_geo_json.json') as json_file:
        world_geo_json = json.load(json_file)
    return world_geo_json