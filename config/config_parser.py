import yaml
import json
import os


def load_config(config_path: str) -> dict:
    """
    Loads a configuration file (YAML or JSON) and returns a dictionary.
    :param config_path: Path to the configuration file.
    :return: dict: Parsed configuration as a dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Configuration file not found at {config_path}')

    file_extension = os.path.splitext(config_path)[1]

    with open(config_path, 'r') as file:
        if file_extension == '.yaml' or file_extension == '.yml':
            config = yaml.safe_load(file)
        elif file_extension == '.json':
            config = json.load(file)
        else:
            raise ValueError(f'Unsupported configuration file format: {file_extension}')

    return config
