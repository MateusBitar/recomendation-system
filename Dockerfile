FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 1) Instalar compiladores do sistema (necessário para o gcc)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 2) Atualizar ferramentas de build do Python
# Isso evita problemas com versões antigas do wheel/setuptools
RUN pip install --upgrade pip setuptools wheel

# 3) Instalar as dependências de compilação PRIMEIRO
# O scikit-surprise precisa do numpy instalado para compilar suas extensões em C
RUN pip install --no-cache-dir "numpy<2" "cython<3"

# 4) Instalar scikit-surprise SEM isolamento de build
# O flag --no-build-isolation obriga o pip a usar o numpy instalado no passo anterior
RUN pip install --no-build-isolation --no-cache-dir "scikit-surprise==1.1.3"

# 5) Copiar e instalar o restante das dependências
COPY requirements.txt .
# DICA: Remova numpy e scikit-surprise do requirements.txt para evitar conflitos,
# ou garanta que as versões sejam idênticas.
RUN pip install --no-cache-dir -r requirements.txt

# 6) Copiar o código
COPY . .

ENV DATA_DIR=/app/data \
    MODELS_DIR=/app/models \
    MOVIE_DATA_FILENAME=movies.dat \
    MODEL_FILENAME=svd_model.pkl

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]