from langchain_docs_loader import LangchainDocsLoader

loader = LangchainDocsLoader(include_output_cells=True)
docs = loader.load()
f"Loaded {len(docs)} documents"