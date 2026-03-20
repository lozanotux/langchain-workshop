from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv
import numpy as np


load_dotenv()

embeddings = MistralAIEmbeddings(model="mistral-embed")

texto1 = "La capital de Francia es París."
texto2 = "Paris es la ciudad capital de Francia."

vector1 = embeddings.embed_query(texto1)
vector2 = embeddings.embed_query(texto2)

print("Dimension de los vectores:", len(vector1))  # 1024

cos_sim = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
print(f"Similitud coseno entre los vectores: {cos_sim:.3f}")  # 0.950 (mas cercano a 1, mas similares)
