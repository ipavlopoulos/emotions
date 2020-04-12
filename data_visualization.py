from twitter import DataHandler, Visualizer
from utils import load_yaml
import click
cli = click.Group()


@cli.command()
@click.option('--lan', default='en')
@click.option('--config', default="configs/configuration.yaml")
def execute_data_visualization(lan, config):
    config = load_yaml(config)[lan]
    handler = DataHandler(directory=config['path'], max_capacity_per_file=config['csv_size'])
    visualizer = Visualizer(mapping=config['mapping'], handler=handler)
    visualizer.pin(num_of_data=10, image_path=config['image_path'])


if __name__ == "__main__":
    execute_data_visualization()
