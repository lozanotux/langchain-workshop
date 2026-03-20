# Configuración de modelos
EMBEDDING_MODEL = "mistral-embed"
QUERY_MODEL = "magistral-small-latest"
GENERATION_MODEL = "mistral-large-latest"

# Configuración del vector store
CHROMA_DB_PATH = "/Users/jlozano/Desktop/Laboratory/udemy-langchain/Tema_3/chroma_db"

# Configuración del retriever
SEARCH_TYPE = "mmr"
MMR_DIVERSITY_LAMBDA = 0.7
MMR_FETCH_K = 20
SEARCH_K = 2

# Configuracion alternativa para retriever hibrido
ENABLE_HYBRID_SEARCH = True
SIMILARITY_THRESHOLD = 0.70