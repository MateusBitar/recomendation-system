
# 📚 Sistema de Recomendação com FastAPI + SVD

Este projeto implementa um **sistema de recomendação baseado em Filtragem Colaborativa**, utilizando o algoritmo **SVD (Singular Value Decomposition)** da biblioteca *scikit-surprise*.
A API foi desenvolvida em **FastAPI**, possui documentação interativa (Swagger) e está completamente containerizada com **Docker**.

---

## 🚀 Funcionalidades

* Treinamento de modelo de recomendação usando **Surprise SVD**
* Carregamento de dados do MovieLens (ratings, movies, users)
* API REST com FastAPI para:

  * Gerar recomendações para um usuário → `/recommendations/{user_id}`
  * Verificar informações básicas do sistema → `/health`
* Containerização completa com Docker + Docker Compose
* Separação limpa entre:

  * Código da API
  * Código de pré-processamento
  * Código de treinamento do modelo
* Utilização de variáveis de ambiente para caminhos e parâmetros

---

# 📁 Estrutura do Projeto

```
.
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── main.py
├── movie_data_loader.py
├── train_model.py
├── data/               <-- arquivos .dat (IGNORADOS no Git)
└── models/             <-- modelo treinado .pkl
```

---

# 🧠 Modelo de Recomendação

O modelo utiliza:

* **Filtragem Colaborativa** (Collaborative Filtering)
* **SVD – Singular Value Decomposition**
* Treinamento via Surprise:

  ```python
  from surprise import SVD, Dataset, Reader
  ```
* O modelo treinado é salvo como:

  ```
  models/svd_model.pkl
  ```

Fluxo geral:

1. Carrega dados (ratings.dat)
2. Processa entradas com Surprise Reader
3. Treina o algoritmo SVD
4. Salva o modelo para uso na API

---

# ⚙️ Como Executar o Projeto

## 🧪 1. Rodar Localmente (sem Docker)

### Instale dependências

```bash
pip install -r requirements.txt
```

### Inicie a API

```bash
uvicorn main:app --reload
```

### Acesse a documentação

➡️ [http://localhost:8000/docs](http://localhost:8000/docs)

---

# 🐳 Executando com Docker

## 1️⃣ Build da imagem

(necessário apenas no primeiro uso ou quando o Dockerfile for alterado)

```bash
docker compose build
```

## 2️⃣ Subir o container

```bash
docker compose up
```

A API ficará disponível em:

➡️ [http://localhost:8000/docs](http://localhost:8000/docs)

### 3️⃣ Rodar em background

```bash
docker compose up -d
```

### 4️⃣ Parar tudo

```bash
docker compose down
```

---

# 🔁 Treinar ou Re-treinar o Modelo

Execute:

```bash
python train_model.py
```

O novo modelo será salvo automaticamente em:

```
models/svd_model.pkl
```

E será carregado pela API ao iniciar.

---

# 📡 Endpoints da API

## 🔹 GET `/health`

Retorna status básico da API.
Exemplo:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

## 🔹 GET `/recommendations/{user_id}`

Retorna **N recomendações personalizadas** para um usuário.

### Parâmetros:

* **user_id** (int)
* **top_n** (int) → opcional, padrão = 5

### Exemplo:

```
GET http://localhost:8000/recommendations/10?top_n=10
```

### Exemplo de resposta:

```json
{
  "user_id": 10,
  "recommendations": [
    { "movie_id": 1196, "title": "Star Wars: Episode V", "predicted_rating": 4.83 },
    { "movie_id": 1210, "title": "Star Wars: Episode VI", "predicted_rating": 4.77 },
    ...
  ]
}
```

---

# 🧩 Configuração por Variáveis de Ambiente

Você pode customizar caminhos usando um arquivo `.env`:

```
DATA_DIR=./data
MODELS_DIR=./models
MOVIE_DATA_FILENAME=movies.dat
MODEL_FILENAME=svd_model.pkl
```

No Docker, essas variáveis já são definidas automaticamente.

---

# 🤝 Contribuições

Sinta-se à vontade para abrir issues ou PRs.
Projetos de recomendação são facilmente extensíveis com:

* Modelos híbridos (conteúdo + colaborativo)
* Métricas avançadas (RMSE, MAE, MAP@K)
* Filtros contextuais (gênero, ano, popularidade)
* Banco de dados para armazenar avaliações

---

# 🏁 Conclusão

Este projeto demonstra a construção completa de um sistema de recomendação:

* Treinamento
* Deploy
* API
* Containerização

É um excelente ponto de partida para sistemas mais avançados, aplicações reais e uso em produção.

