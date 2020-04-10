import json
import pathlib
import os


def get_config_from_json(path):
    with open(path, 'r') as f:
        config = json.load(f)
    return config


twitter_config = get_config_from_json(os.path.join(pathlib.Path(__file__).parent.absolute(), "config.json"))
