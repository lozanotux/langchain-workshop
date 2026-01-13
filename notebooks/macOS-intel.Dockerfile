# macOS intel doesn't support PyTorch above version 2.2.2, so we need this container to bypass that limitation
FROM python:3.11-slim-bullseye

# Install Jupyter Notebook
RUN pip3 install jupyter ipykernel ipython ipywidgets

# Install PyTorch CPU version
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cpu

# Install LangChain
RUN pip3 install langchain=="0.0.209" langchain-community=="0.0.38" langchain-huggingface=="0.0.3"

# Install Hugging Face Transformers
RUN pip3 install transformers=="4.57.3" einops=="0.8.1" accelerate=="1.12.0"

# Create system user
RUN useradd -ms /bin/bash jupyter

# Change to this new user
USER jupyter

# Set the working directory
WORKDIR /home/jupyter

# Expose port for Jupyter Notebook
EXPOSE 8888

# Start Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=*"]