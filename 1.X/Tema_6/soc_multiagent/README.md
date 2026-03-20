# SOC - Sistema de Seguridad Multi-agente

Este sistema utiliza LangChain, LangGraph, FastAPI y Streamlit para poner en practica todos los conceptos vistos en el curso. Se trata de un sistema de seguridad que permite reportar brechas de seguridad mediante el uso de agentes de IA.

## Como usar este proyecto?

1. Completar el contenido del archivo `.env.example` con las APIs de los servicios a utilizar y renombrarlo a `.env`.

2. Crear un entorno virtual y activarlo:

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

3. Instalar dependencias:
```sh
pip install -r requirements.txt
```

4. Ejecución del proyecto:

- **Para la API:**

```sh
python Tema_6/soc_multiagent/webhook_server.py
```

- **Para el dashboard visual (app streamlit):**

```sh
streamlit run Tema_6/soc_multiagent/dashboard.py 
```

## Consumir la API:

Para crear un ticket de seguridad y enviarlo por correo ejecuta la siguiente llamada HTTP:

```sh
curl -X POST "http://localhost:8000/webhook/alert" \
  -H "Content-Type: application/json" \
  -d '{ "source": "proxy", "alert_type": "Malicious URL detection", "severity": "High", "message": "Actividad sospechosa detectada desde el proxy con una URL", "source_ip": "110.37.40.215", "url": "http://110.37.40.215:34887/bin.sh", "file_hash": "4293c1d8574dc87c58360d6bac3daa182f64f7785c9d41da5e0741d2b1817fc7", "timestamp": "2026-03-19T17:11:57", "email_recipient": "lozanotux@gmail.com", "real_apis": true }'
```
