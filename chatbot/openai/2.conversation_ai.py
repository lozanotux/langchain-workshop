from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils import DocsJSONLLoader, load_config, get_file_path, get_openai_api_key, get_query_from_user
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from rich.console import Console


config = load_config()

console = Console()

recreate_chroma_db = config["recreate_chroma_db"]
chat_type = config["chat_type"] # Options: "qa" or "memory_chat"


def load_documents(file_path: str):
    # Function to load documents from the given file path
    loader = DocsJSONLLoader(file_path)
    data = loader.load()

    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size=1600,
        length_function=len,
        chunk_overlap=160,
    )

    return text_spliter.split_documents(data)


def get_chroma_db(embeddings, documents, path):
    if recreate_chroma_db:
        console.print("Recreating Chroma DB...")
        return Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=path,
        )
    else:
        console.print("Loading existing Chroma DB...")
        return Chroma(
            persist_directory=path,
            embedding_function=embeddings,
        )


def run_conversation(vectorstore, chat_type, llm):
    console.print(
        "\n[blue]AI: Hi! What do you want to ask me about Transformers"
        "or artificial intelligence?[/blue]\n"
    )
    if chat_type == "qa":
        console.print(
            "\n[green]You are using the chatbot in Q&A mode. This chatbot"
            " generates answers based on the query exclusively without"
            " considering the conversation history.[/green]"
        )
    elif chat_type == "memory_chat":
        console.print(
            "\n[green]You are using the chatbot in memory mode. This chatbot"
            " generates answers based on the entire conversation history."
        )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    chat_history = []

    while True:
        console.print("\n[blue]You:[/blue] ", end="")
        query = get_query_from_user()

        if query.lower() in {"exit", "quit"}:
            console.print("\n[red]Exiting the chatbot. Goodbye![/red]")
            break
    
        if chat_type == "qa":
            response = process_qa_query(retriever, query, llm)
        elif chat_type == "memory_chat":
            response = process_memory_chat_query(retriever, query, llm, chat_history)

        console.print(f"\n[yellow]AI:[/yellow] {response}\n")


def process_qa_query(retriever, query, llm):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
    )
    console.print("\n[green]AI is thinking...[/green]")
    return qa_chain.run(query)


def process_memory_chat_query(retriever, query, llm, chat_history):
    memory_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        verbose=True,
    )
    console.print("\n[green]AI is thinking...[/green]")
    result = memory_chain(
        {
            "question": query,
            "chat_history": chat_history,
        }
    )
    chat_history.append((query, result["answer"]))
    return result["answer"]


def main():
    documents = load_documents(get_file_path())
    get_openai_api_key()
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    vectorstore_chroma = get_chroma_db(
        embeddings, documents, "chroma_docs"
    )

    console.print(
        f"[green]Documents {len(documents)} loaded successfully.[/green]"
    )

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.2,
        max_tokens=1000,
    )

    run_conversation(vectorstore_chroma, chat_type, llm)


if __name__ == "__main__":
    main()
