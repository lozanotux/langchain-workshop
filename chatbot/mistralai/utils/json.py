import jsonlines
from langchain_core.documents import Document


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
                documents.append(Document(page_content=page_content, metadata=metadata))
        return documents
