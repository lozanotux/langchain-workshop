# Helpdesk System RAG

Es una aplicación construida con LangChain y LangGraph para poner en practica todo lo aprendido en el modulo 4.

## Como usar este repositorio?

1. Inicializa un entorno virtual:

```sh
python3 -m venv venv
```

```sh
# macOS & GNU/Linux
source venv/bin/activate
```

```sh
# Windows
.\venv\Scripts\activate
```

2. Instala las dependencias:

```sh
pip install -r requirements.txt
```

3. Genera la base de datos vectorial primero:

```sh
python setup_rag.py
```

4. Ejecutar aplicación:

```sh
streamlit run app.py
```