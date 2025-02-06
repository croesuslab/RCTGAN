import yaml
import logging

def load_yaml_to_dict(file_path: str) -> dict:
    """
    Loads YAML data from a file into a Python dictionary.

    Args:
        file_path (str): The path to the YAML file to be loaded.

    Returns:
        dict: A dictionary representation of the YAML data.
    """
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except yaml.YAMLError as exc:
        logging.error(f"Error parsing YAML file: {file_path} - {exc}")
        raise