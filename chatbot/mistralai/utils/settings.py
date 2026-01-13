import os

import yaml


def load_config():
    """
    Load the application configuration from the 'config.yaml' file.

    Returns:
        A dictionary with the application settings.
    """
    root_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(root_dir, "../config.yaml")) as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
