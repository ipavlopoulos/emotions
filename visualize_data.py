from twitter import DataHandler, Visualizer
from utils import load_yaml
import os
import click


cli = click.Group()


@cli.command()
@click.option('--lan', default='en')
@click.option('--config', default="configs/configuration.yaml")
@click.option('--pins_num', default=1000)
def execute_data_visualization(lan, config, pins_num):
    config = load_yaml(config)[lan]
    handler = DataHandler(directory=config['path'], max_capacity_per_file=config['csv_size'])
    visualizer = Visualizer(mapping=config['mapping'], handler=handler)
    if not os.path.exists(config['image_dir']):
        os.makedirs(config['image_dir'])
    if not os.path.exists(config['hash_addresses_dir']):
        os.makedirs(config['hash_addresses_dir'])
    visualizer.pin(num_of_data=pins_num, image_path=os.path.join(config['image_dir'], config['image_path']),
                   geohashing_path=os.path.join(config['hash_addresses_dir'], config['hash_addresses_path']))


if __name__ == "__main__":
    execute_data_visualization()
