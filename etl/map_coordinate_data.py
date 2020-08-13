import json


def get_world_geo_json() -> dict:
    geo_json = __load_world_geo_json_data()
    return geo_json


def __load_world_geo_json_data() -> dict:
    with open('D:\Programming\Python\Corona Analytics Project\data\world_geo_json.json') as json_file:
        world_geo_json = json.load(json_file)
    return world_geo_json