from flask import Flask, render_template

from twitter import DataHandler, Visualizer
from utils import load_yaml
import os
import json

app = Flask(__name__)


def execute_data_visualization(lan, config, pins_num):
    config = load_yaml(config)[lan]
    handler = DataHandler(directory=config['path'], max_capacity_per_file=config['csv_size'])
    df = handler.load_all_data()
    pins = df[df['location'].notnull()].tail(pins_num)
    return ",".join(pins.location.values)


def get_pins(lan, config, pins_num):
    config = load_yaml(config)[lan]
    handler = DataHandler(directory=config['path'], max_capacity_per_file=config['csv_size'])
    visualizer = Visualizer(mapping=config['mapping'], handler=handler)
    if not os.path.exists(config['image_dir']):
        os.makedirs(config['image_dir'])
    if not os.path.exists(config['hash_addresses_dir']):
        os.makedirs(config['hash_addresses_dir'])
    pins = visualizer.get_last_pins(num_of_pins=pins_num,
                                    addresses_path=os.path.join(config['hash_addresses_dir'],
                                                                config['hash_addresses_path']))
    return json.dumps(pins)


@app.route("/")
def call():
    pins = get_pins('en', "configs/configuration.yaml", 10)
    return render_template("intro.html", pins=pins)


if __name__ == '__main__':
    app.run(debug=True)