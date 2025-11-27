# 📚 Sistema de Recomendação com FastAPI + SVD (Filtragem Colaborativa)

Este projeto implementa um **sistema completo de recomendação**, passando por:

1. **Treinamento de modelo** usando Filtragem Colaborativa com **SVD (Singular Value Decomposition)**
2. **Serviço de API** construído com **FastAPI** para disponibilizar recomendações
3. **Containerização em Docker**, garantindo ambiente reprodutível
4. **Documentação completa** de instalação, execução e arquitetura

O objetivo é entregar um sistema modular, funcional e escalável, que pode ser aplicado a qualquer cenário de recomendação baseado em usuários.

---

# 📌 Decisões de Design e Arquitetura

## 🎯 Escolha do modelo: Filtragem Colaborativa (SVD – Surprise)

O modelo escolhido foi o **SVD (Singular Value Decomposition)** da biblioteca *scikit-surprise*.
As decisões envolvidas:

### ✔ Por que usar Filtragem Colaborativa?

* Permite prever preferências **mesmo sem dados de conteúdo** (gênero, descrição, etc.)
* Aprende padrões ocultos de comportamento entre usuários
* Escalável para grandes bases como MovieLens
* Requer apenas a matriz usuário–item

### ✔ Por que o algoritmo **SVD**?

* É o algoritmo mais usado para recomendação baseada em rating
* Representa usuários e itens em um espaço de fatores latentes
* Tem robustez contra sparsidade
* Supera métodos baseados em média/knn em grande parte dos datasets

### ✔ Por que usar a biblioteca **Surprise**?

* Possui implementação otimizada do SVD
* Facilita leitura de datasets como MovieLens
* Suporta salvamento/carregamento de modelos
* É ideal para projetos acadêmicos e protótipos

### ✔ Por que salvar o modelo em `.pkl`?

* Evita re-treinamento toda vez que a API sobe
* Reduz o tempo de startup
* Facilita deploy em ambientes com menos recursos

---

## 🏛 Decisões de Arquitetura

### ✔ Separação de responsabilidades

* `train_model.py` → Treina e salva o modelo
* `movie_data_loader.py` → Carrega e pré-processa os dados
* `main.py` → API FastAPI
* `models/` e `data/` → separados para organização e versionamento

### ✔ Carregamento do modelo na inicialização da API

* Reduz latência nas requisições
* Evita computações repetidas
* Permite detectar erros antes da API ficar disponível

### ✔ Uso de Docker

* Surprise depende de compilação de C e versões específicas de NumPy → Docker garante compatibilidade
* Cria ambiente 100% reproduzível
* Permite deploy fácil em qualquer servidor

### ✔ Python 3.10 slim

* Compatível com Surprise
* Leve o suficiente para ambientes cloud
* Evita problemas com NumPy 2.0+

---

# 🧠 Funcionamento do Modelo de Recomendação (Explicação Técnica)

### ✔ Conceito

O SVD aproxima a matriz original de ratings:

```
Usuários x Itens
```

por três matrizes menores:

```
R ≈ P × Qᵀ
```

Onde:

* **P** → matriz de fatores dos usuários
* **Q** → matriz de fatores dos filmes
* **R** → matriz de ratings observados

### ✔ Predição

A predição de um rating é feita assim:

```
rating ≈ viés_global + viés_usuario + viés_item + (P × Qᵀ)
```

Ou seja, o modelo prevê:

* Quanto o usuário tende a gostar de itens no geral
* Quanto cada item tende a ser bem avaliado
* A interação entre fatores latentes

### ✔ Recomendação

A API filtra:

1. Itens que o usuário **ainda não avaliou**
2. Calcula o **rating previsto**
3. Ordena do maior para o menor
4. Retorna os **Top N recomendados**

---

# 🧪 Como Rodar o Projeto (Localmente)

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Iniciar API

```bash
uvicorn main:app --reload
```

### 3. Acessar documentação

👉 [http://localhost:8000/docs](http://localhost:8000/docs)

---

# 🐳 Como Rodar com Docker

### 1. Build

```bash
docker compose build
```

### 2. Executar

```bash
docker compose up
```

### 3. API disponível em:

👉 [http://localhost:8000/docs](http://localhost:8000/docs)

---

# 🔁 Como Re-Treinar o Modelo

```bash
python train_model.py
```

O novo modelo será salvo em:

```
models/svd_model.pkl
```

---

# 📡 Endpoints

### 🔹 `/recommendations/{user_id}`

Retorna recomendações personalizadas.

### 🔹 `/health`

Verifica status da API e carregamento do modelo.

---

# 🌱 Possíveis Extensões Futuras

* Versão híbrida (Conteúdo + CF)
* KNN com similaridade entre usuários
* Métricas de avaliação (RMSE, Precision@K)
* Deploy em Kubernetes
* Armazenar ratings em banco PostgreSQL

---

# 🎉 Conclusão

Este projeto apresenta um ciclo completo de construção de sistemas de recomendação:

* Treinamento
* Persistência do modelo
* API REST
* Deploy com Docker
* Documentação completa
