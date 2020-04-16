from flask import Flask, render_template

from twitter import DataHandler
from utils import load_yaml

app = Flask(__name__)


def execute_data_visualization(lan, config, pins_num):
    config = load_yaml(config)[lan]
    handler = DataHandler(directory=config['path'], max_capacity_per_file=config['csv_size'])
    df = handler.load_all_data()
    pins = df[df['location'].notnull()].tail(pins_num)
    return ",".join(pins.location.values)


@app.route("/")
def call():
    locs = execute_data_visualization('en', "configs/configuration.yaml", 10)
    print(locs)
    return locs

if __name__ == '__main__':
    app.run(debug=True)