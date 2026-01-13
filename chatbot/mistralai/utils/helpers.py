import os


def get_mistral_api_key():
    """
    Obtains the Mistral API key from the environment. If it is not available,
    requests the user to enter it.

    Returns:
        The Mistral API key.
    """
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_api_key:
        mistral_api_key = input("Please enter your MISTRAL_API_KEY: ")
    return mistral_api_key


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
