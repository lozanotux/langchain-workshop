import os
import sys

import jsonlines
import yaml
from langchain.schema import Document


class DocsJSONLLoader:
    """
    Document loader for documentation in JSONL format.

    Args:
        file_path (str): Path to the JSONL file to be loaded.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self):
        """
        Loads the documents from the file path specified during
        initialization.

        Returns:
            A list of Document objects.
        """
        with jsonlines.open(self.file_path) as reader:
            documents = []
            for obj in reader:
                page_content = obj.get("text", "")
                metadata = {
                    "title": obj.get("title", ""),
                    "repo_owner": obj.get("repo_owner", ""),
                    "repo_name": obj.get("repo_name", ""),
                }
                documents.append(
                    Document(
                        page_content=page_content,
                        metadata=metadata))
        return documents


def load_config():
    """
    Load the application configuration from the 'config.yaml' file.

    Returns:
        A dictionary with the application settings.
    """
    root_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(root_dir, "config.yaml")) as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def get_openai_api_key():
    """
    Obtains the OpenAI API key from the environment. If it is not available,
    it stops the program from running.

    Returns:
        The OpenAI API key.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Please create an environment variable OPENAI_API_KEY.")
        sys.exit()
    return openai_api_key


def get_cohere_api_key():
    """
    Obtains the Cohere API key from the environment. If it is not available,
    requests the user to enter it.

    Returns:
        The Cohere API key.
    """
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not cohere_api_key:
        cohere_api_key = input("Please enter your COHERE_API_KEY: ")
    return cohere_api_key


def get_file_path():
    """
    Gets the path to the JSONL database file specified in the application's
    configuration.

    Returns:
        The path to the JSONL database file.
    """
    config = load_config()

    root_dir = os.path.dirname(os.path.abspath(__file__))
    requested_dir = os.path.join(root_dir, config["jsonl_database_path"])

    return os.path.join(requested_dir)


def get_query_from_user() -> str:
    """
    Request a query from the user.

    Returns:
        The query entered by the user.
    """
    try:
        query = input()
        return query
    except EOFError:
        print("Error: Unexpected input. Please try again.")
        return get_query_from_user()


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
