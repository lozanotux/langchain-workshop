import datetime
import json
import os
import re
from typing import Dict

import emoji
import requests
from dotenv import load_dotenv
from termcolor import colored

from utils.filesystem import create_dir, remove_existing_file
from utils.settings import load_config


def preprocess_text(text: str) -> str:
    """
    Preprocess the text by removing certain patterns and characters.

    Args:
        text (str): Text to be preprocessed.

    Returns:
        The preprocessed text.
    """
    text = re.sub(r"<[^>]*>", "", text)
    text = re.sub(r"http\S+|www.\S+", "", text)
    text = re.sub(r"Copyright.*", "", text)
    text = text.replace("\n", " ")
    text = emoji.demojize(text)
    text = re.sub(r":[a-z_&+-]+:", "", text)
    return text


def download_file(url: str, repo_info: dict, jsonl_file_name: str) -> None:
    """
    Download a file from a URL and save it to a JSONL file.

    Args:
        url (str): URL from where the file is downloaded.
        repo_info (dict): Information about the repository from where the file
        is downloaded.
        jsonl_file_name (str): Name of the JSONL file where the downloaded
        file is saved.
    """
    response = requests.get(url)
    filename = url.split("/")[-1]
    text = response.text

    if text is not None and isinstance(text, str):
        text = preprocess_text(text)
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        file_dict = {
            "title": filename,
            "repo_owner": repo_info["owner"],
            "repo_name": repo_info["repo"],
            "text": text,
        }

        with open(jsonl_file_name, "a") as jsonl_file:
            jsonl_file.write(json.dumps(file_dict) + "\n")
    else:
        print(f"Texto no esperado: {text}")


def process_directory(
    path: str,
    repo_info: Dict,
    headers: Dict,
    jsonl_file_name: str,
) -> None:
    """
    Processes a directory from a GitHub repository and downloads the files
    in it.

    Args:
        path (str): Path of the directory to be processed.
        repo_info (Dict): Information about the repository that contains
        the directory.
        headers (Dict): Headers for the request to the GitHub API.
        jsonl_file_name (str): Name of the JSONL file where the downloaded
        files will be saved.
    """
    # If the directory name is 'zh', it skips it and returns immediately. This
    # feature is implemented so that Chinese translations are not downloaded.
    if os.path.basename(path) == "zh":
        print(
            colored(
                f"The 'zh' directory (Chinese translations) skipped: {path}", "yellow"
            )
        )
        return

    base_url = (
        f"https://api.github.com/repos/{repo_info['owner']}"
        f"/{repo_info['repo']}/contents/"
    )
    print(
        colored(
            f"Processing directory: {path}" f" from the repo: {repo_info['repo']}",
            "blue",
        )
    )
    response = requests.get(base_url + path, headers=headers)

    if response.status_code == 200:
        files = response.json()
        for file in files:
            if file["type"] == "file" and (
                file["name"].endswith(".mdx") or file["name"].endswith(".md")
            ):
                print(colored(f"Downloading document: {file['name']}", "green"))
                print(colored(f"Download URL: {file['download_url']}", "cyan"))
                download_file(
                    file["download_url"],
                    repo_info,
                    jsonl_file_name,
                )
            elif file["type"] == "dir":
                process_directory(
                    file["path"],
                    repo_info,
                    headers,
                    jsonl_file_name,
                )
        print(colored("Success in extracting documents from the directory.", "green"))
    else:
        print(
            colored(
                "Failed to retrieve files. Please check your GitHub token"
                " and repository details.",
                "red",
            )
        )


def main():
    """
    Main function that runs when the script starts.
    """
    config = load_config()
    github_token = os.getenv("GITHUB_TOKEN")

    if github_token is None:
        raise ValueError("GITHUB_TOKEN is not set in the environment variables.")

    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github.v3.raw",
    }

    current_date = datetime.date.today().strftime("%Y_%m_%d")
    jsonl_file_name = f"data/docs_en_{current_date}.jsonl"

    create_dir("data/")
    remove_existing_file(jsonl_file_name)

    for repo_info in config["github"]["repos"]:
        process_directory(
            repo_info["path"],
            repo_info,
            headers,
            jsonl_file_name,
        )


if __name__ == "__main__":
    load_dotenv()
    main()
