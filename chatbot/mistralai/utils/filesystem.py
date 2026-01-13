import os
from .settings import load_config


def create_dir(path: str) -> None:
    """
    Create a directory if it does not exist.

    Args:
        path (str): Path of the directory to be created.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def remove_existing_file(file_path: str) -> None:
    """
    Delete a file if it exists.

    Args:
        file_path (str): Path of the file to be deleted.
    """
    if os.path.exists(file_path):
        os.remove(file_path)


def get_file_path():
    """
    Gets the path to the JSONL database file specified in the application's
    configuration.

    Returns:
        The path to the JSONL database file.
    """
    config = load_config()

    root_dir = os.path.dirname(os.path.abspath(__file__))
    requested_dir = os.path.join(root_dir, "../", config["jsonl_database_path"])

    return os.path.join(requested_dir)
