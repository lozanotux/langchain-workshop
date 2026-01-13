import re
import urllib.request
import xml.etree.ElementTree as ET
from multiprocessing.pool import ThreadPool
from typing import Any, Generator, List

import requests
import tiktoken
from bs4 import BeautifulSoup, Doctype, NavigableString, SoupStrainer, Tag
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from tenacity import retry, stop_after_attempt, wait_random_exponential


class LangchainDocsLoader(BaseLoader):
    """A loader for the Langchain documentation.

    The documentation is available at https://docs.langchain.com/oss/python/.
    """

    _sitemap: str = "https://docs.langchain.com/sitemap.xml"
    _filter_urls: List[str] = ["https://docs.langchain.com/oss/python/"]

    def __init__(
        self,
        number_threads: int = 50,
        include_output_cells: bool = True,
        include_links_in_header: bool = False,
    ) -> None:
        """Initialize the loader.

        Args:
            number_threads (int, optional): Number of threads to use
                for parallel processing. Defaults to 50.
        """
        self._number_threads = number_threads
        self._include_output_cells = include_output_cells
        self._include_links_in_header = include_links_in_header

    def load(self) -> List[Document]:
        """Load the documentation.

        Returns:
            List[Document]: A list of documents.
        """

        urls = self._get_urls()
        docs = self._process_urls(urls)
        return docs

    def _get_urls(self) -> List[str]:
        """Get the urls from the sitemap."""
        with urllib.request.urlopen(self._sitemap) as response:
            xml = response.read()

        root = ET.fromstring(xml)

        namespaces = {"sitemap": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        urls = [
            url.text
            for url in root.findall(".//sitemap:loc", namespaces=namespaces)
            if url.text is not None and "https://docs.langchain.com/oss/python/" in url.text
        ]

        return urls

    def _process_urls(self, urls: List[str]) -> List[Document]:
        """Process the urls in parallel."""
        with ThreadPool(self._number_threads) as pool:
            docs = pool.map(self._process_url, urls)
            return docs

    @retry(
        stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=10)
    )
    def _process_url(self, url: str) -> Document:
        """Process a url."""
        r = requests.get(url, allow_redirects=False)
        html = r.text
        metadata = self._metadata_extractor(html, url)
        page_content = self.langchain_docs_extractor(
            html=html,
            include_output_cells=self._include_output_cells,
            # remove the first part of the url
            path_url="/".join(url.split("/")[3:])
            if self._include_links_in_header
            else None,
        )
        return Document(page_content=page_content, metadata=metadata)

    def _metadata_extractor(self, raw_html: str, url: str) -> dict[Any, Any]:
        """Extract metadata from raw html using BeautifulSoup."""
        metadata = {"source": url}

        soup = BeautifulSoup(raw_html, "lxml")
        if title := soup.find("title"):
            metadata["title"] = title.get_text()
        if description := soup.find("meta", attrs={"name": "description"}):
            if isinstance(description, Tag):
                content = description.get("content", None)
                if isinstance(content, str):
                    metadata["description"] = content
            else:
                metadata["description"] = description.get_text()
        if html := soup.find("html"):
            if isinstance(html, Tag):
                lang = html.get("lang", None)
                if isinstance(lang, str):
                    metadata["language"] = lang

        return metadata

    @staticmethod
    def langchain_docs_extractor(
        html: str,
        include_output_cells: bool = True,
        path_url: str | None = None,
    ) -> str:
        # 1. Extraemos solo el contenedor principal por ID
        soup = BeautifulSoup(
            html,
            "lxml",
            parse_only=SoupStrainer(id="content-container"),
        )

        # 2. Eliminamos elementos de navegación y ruido específicos del nuevo formato
        # 'content-side-layout' suele contener el Table of Contents (TOC) a la derecha
        # 'table-of-contents' es otra variante común en estos frameworks
        SCAPE_IDS = ["content-side-layout", "table-of-contents", "header"]
        for div_id in SCAPE_IDS:
            for tag in soup.find_all(id=div_id):
                tag.decompose()

        # Eliminamos tags generales innecesarios
        SCAPE_TAGS = ["nav", "footer", "aside", "script", "style", "noscript", "button"]
        for tag in soup.find_all(SCAPE_TAGS):
            tag.decompose()

        def get_text(tag: Tag) -> Generator[str, None, None]:
            for child in tag.children:
                if isinstance(child, Doctype):
                    continue

                if isinstance(child, NavigableString):
                    yield child.get_text()
                
                elif isinstance(child, Tag):
                    # --- Encabezados ---
                    if child.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                        yield f"\n\n{'#' * int(child.name[1:])} "
                        yield child.get_text(strip=True)
                        yield "\n\n"

                    # --- Enlaces ---
                    elif child.name == "a":
                        href = child.get('href', '')
                        text = child.get_text(strip=True)
                        if text and href:
                            yield f"[{text}]({href})"
                        else:
                            yield text

                    # --- Formato Básico ---
                    elif child.name in ["strong", "b"]:
                        yield f"**{child.get_text(strip=True)}**"
                    elif child.name in ["em", "i"]:
                        yield f"_{child.get_text(strip=True)}_"
                    elif child.name == "br":
                        yield "\n"

                    # --- Bloques de Código ---
                    # El nuevo formato suele envolver el pre en un div con clase code-block
                    elif child.name == "pre":
                        # Intentar detectar el lenguaje del contenedor padre o clases
                        language = ""
                        parent = child.find_parent("div")
                        if parent and parent.has_attr("language"):
                            language = parent.get("language")
                        elif not language:
                            # Fallback a clases antiguas o detección simple
                            classes = child.get("class", [])
                            for cls in classes:
                                if cls.startswith("language-"):
                                    language = cls.split("-")[1]
                                    break
                        
                        # Extraer solo el texto del código
                        code_content = child.get_text()
                        yield f"\n``` {language}\n{code_content}\n```\n\n"

                    # --- Párrafos (Incluye soporte para Mintlify span data-as="p") ---
                    elif child.name == "p" or (child.name == "span" and child.get("data-as") == "p"):
                        yield from get_text(child)
                        yield "\n\n"

                    # --- Listas ---
                    elif child.name == "ul":
                        for li in child.find_all("li", recursive=False):
                            yield "- "
                            yield from get_text(li)
                            yield "\n"
                        yield "\n"
                    
                    elif child.name == "ol":
                        for i, li in enumerate(child.find_all("li", recursive=False)):
                            yield f"{i + 1}. "
                            yield from get_text(li)
                            yield "\n"
                        yield "\n"

                    # --- Tablas ---
                    elif child.name == "table":
                        thead = child.find("thead")
                        if thead:
                            headers = thead.find_all("th")
                            if headers:
                                yield "| " + " | ".join(h.get_text(strip=True) for h in headers) + " |\n"
                                yield "| " + " | ".join("---" for _ in headers) + " |\n"
                        
                        tbody = child.find("tbody")
                        if tbody:
                            for row in tbody.find_all("tr"):
                                cells = row.find_all(["td", "th"])
                                yield "| " + " | ".join(c.get_text(strip=True) for c in cells) + " |\n"
                        yield "\n"

                    # --- Recursividad General ---
                    else:
                        yield from get_text(child)

        joined = "".join(get_text(soup))
        
        # Limpieza final de espacios en blanco excesivos
        return re.sub(r"\n\n+", "\n\n", joined).strip()


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def load_langchain_docs_splitted() -> List[Document]:
    loader = LangchainDocsLoader(include_output_cells=True)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN,
        chunk_size=1000,
        chunk_overlap=50,
        length_function=num_tokens_from_string,
    )

    return text_splitter.split_documents(docs)


def load_langchain_docs_for_app() -> List[Document]:
    loader = LangchainDocsLoader(
        include_output_cells=True, include_links_in_header=True
    )

    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=50,
        length_function=num_tokens_from_string,
    )

    docs = text_splitter.split_documents(docs)

    docs = [
        doc for doc in docs if doc.page_content not in ("```", "```text", "```python")
    ]

    return docs
