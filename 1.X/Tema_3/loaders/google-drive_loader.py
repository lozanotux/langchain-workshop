from langchain_google_community import GoogleDriveLoader
from dotenv import load_dotenv

load_dotenv()

credentials_path = "ruta/al/archivo/credentials.json"
token_path = "ruta/al/archivo/token.json"

loader = GoogleDriveLoader(
    folder_id="completar_con_el_id_de_tu_carpeta",
    credentials_path=credentials_path,
    token_path=token_path,
    recursive=True,
    scopes=["https://www.googleapis.com/auth/drive.readonly"],
    file_types=["pdf", "application/pdf"]
)

documents = loader.load()

print(len(documents))