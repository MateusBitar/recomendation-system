from fastapi import FastAPI, HTTPException
from surprise.dump import load
from typing import List, Dict
from pydantic import BaseModel
from movie_data_loader import load_movie_data
import os
import pandas as pd
from dotenv import load_dotenv

# 1. Carrega variáveis do .env
load_dotenv()

DATA_DIR = os.getenv("DATA_DIR")
MODELS_DIR = os.getenv("MODELS_DIR")

RATINGS_FILENAME = os.getenv("RATINGS_FILENAME", "ratings.dat")
MOVIE_DATA_FILENAME = os.getenv("MOVIE_DATA_FILENAME", "movies.dat")
MODEL_FILENAME = os.getenv("MODEL_FILENAME", "svd_model.pkl")

RATINGS_PATH = os.path.join(DATA_DIR, RATINGS_FILENAME)
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_FILENAME)

# --- Variáveis globais do modelo ---
model = None
movie_titles: Dict[int, str] = {}
known_user_ids = set()
min_user_id: int | None = None
max_user_id: int | None = None

# Mock simples para guardar preferências/enriquecimento futuro
user_items: Dict[int, list] = {}


# --- Definição do Esquema de Resposta (Pydantic) ---
class Recommendation(BaseModel):
    """Esquema para um item de recomendação."""
    movie_id: int
    title: str
    estimated_rating: float

# --- Funções de Inicialização ---

def load_recommender_model():
    """
    Carrega o modelo SVD treinado, os títulos de filmes e
    a lista de UserIDs válidos do arquivo ratings.dat.
    """
    global model, movie_titles, known_user_ids, min_user_id, max_user_id

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Arquivo de modelo não encontrado em: {MODEL_PATH}")

    try:
        # Carrega modelo treinado (surprise.dump.load)
        _, loaded_model = load(MODEL_PATH)
        model = loaded_model
        print("✅ Modelo de recomendação SVD carregado com sucesso.")

        # Carrega títulos (dict MovieID -> Título)
        movie_titles = load_movie_data()

        # Carrega ratings para descobrir quais usuários existem
        if not os.path.exists(RATINGS_PATH):
            raise RuntimeError(f"Arquivo de ratings não encontrado em: {RATINGS_PATH}")

        ratings_df = pd.read_csv(
            RATINGS_PATH,
            sep="::",
            engine="python",
            names=["UserID", "MovieID", "Rating", "Timestamp"],
        )

        # Conjunto de UserIDs válidos (da base MovieLens)
        user_ids_series = ratings_df["UserID"].astype(int)
        known_user_ids = set(user_ids_series.tolist())
        min_user_id = int(user_ids_series.min())
        max_user_id = int(user_ids_series.max())

        print(
            f"Usuários conhecidos no modelo: {len(known_user_ids)} "
            f"(intervalo: {min_user_id}–{max_user_id})"
        )

    except Exception as e:
        print(f"ERRO: Falha ao carregar o modelo ou dados. Detalhes: {repr(e)}")
        raise RuntimeError(f"Falha crítica: Modelo não pôde ser carregado: {e}")



# --- Inicialização do FastAPI ---
app = FastAPI(title="Sistema de Recomendação - MovieLens SVD")

@app.on_event("startup")
async def startup_event():
    load_recommender_model()


# --- Endpoints da API (Fase 3.2 & 3.3) ---

@app.get("/", summary="Status da API")
async def root():
    """Endpoint de saúde/status da API."""
    return {"message": "API de Recomendação de Filmes em execução!", 
            "model_status": "Loaded" if model else "Failed to Load"}


@app.get(
    "/recommendations/{user_id}", 
    response_model=List[Recommendation],
    summary="Obter as N principais recomendações para um usuário"
)
async def get_recommendations(user_id: int, n: int = 10):
    """
    Gera as principais recomendações (N filmes) para um UserID específico.
    """
    if not model:
        raise HTTPException(status_code=503, detail="Modelo de recomendação não carregado.")
    
    if user_id not in model.trainset.all_users():
        # Caso o UserID não esteja no conjunto de treinamento
        # No MovieLens 1M, os UserIDs vão de 1 a 6040
        raise HTTPException(status_code=404, detail=f"UserID {user_id} não encontrado no modelo.")

    # 1. Obter todos os MovieIDs conhecidos pelo modelo
    all_movie_ids = list(movie_titles.keys())
    
    # 2. Encontrar filmes que o usuário JÁ AVALIOU (para filtragem)
    # Na prática, você usaria um banco de dados para isso. Aqui, simulamos com o trainset.
    # Como o modelo está treinado no full_trainset, podemos usar ele para obter itens avaliados.
    # Note: Para fazer isso de forma correta e robusta, precisaríamos de uma referência
    # a todos os ratings originais. Simplificamos usando o trainset do modelo.
    inner_user_id = model.trainset.to_inner_uid(user_id)
    rated_items_inner = model.trainset.ur[inner_user_id]
    rated_movie_ids = {model.trainset.to_raw_iid(inner_iid) for inner_iid, _ in rated_items_inner}


    # 3. Gerar previsões para todos os filmes que o usuário NÃO AVALIOU
    predictions = []
    for movie_id in all_movie_ids:
        if movie_id not in rated_movie_ids:
            # Estima a avaliação (r_ui=None significa que não há avaliação real)
            pred = model.predict(uid=user_id, iid=movie_id, r_ui=None)
            predictions.append((pred.iid, pred.est))
    
    # 4. Classificar por avaliação estimada e selecionar os N principais
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_n = predictions[:n]

    # 5. Formatar a resposta
    recommendations = []
    for movie_id, est_rating in top_n:
        recommendations.append(Recommendation(
            movie_id=movie_id,
            title=movie_titles.get(movie_id, "Título Desconhecido"),
            estimated_rating=round(est_rating, 4)
        ))
        
    return recommendations


# --- Endpoints de Gerenciamento (Fase 3.3 - Mockup Básico) ---
# Estes são "stubs" para cumprir os requisitos de Adicionar e Atualizar.
# Em um sistema real, essas operações exigiriam um banco de dados e o retreinamento do modelo.

class UserPreference(BaseModel):
    """Esquema para atualizar as preferências de um usuário."""
    movie_id: int
    rating: float

@app.post("/users/{user_id}/preferences", summary="Atualizar as preferências de um usuário")
async def update_preferences(user_id: int, preference: UserPreference):
    """
    Mock: Registra uma nova preferência/avaliação de filme.
    NOTA: Em um sistema de produção, isso acionaria a atualização de um banco de dados 
    e o retreinamento periódico ou incremental do modelo.
    """
    if user_id not in user_items:
        user_items[user_id] = []
    
    # Log da preferência (Mock)
    user_items[user_id].append(preference.movie_id)
    
    return {"message": f"Preferência registrada para UserID {user_id}", 
            "movie": movie_titles.get(preference.movie_id, "Desconhecido"),
            "rating": preference.rating}

@app.post("/users/add/{user_id}", summary="Adicionar um novo usuário ao sistema")
async def add_user(user_id: int):
    """
    Mock: Adiciona um novo usuário ao sistema.
    NOTA: O novo usuário só receberá recomendações úteis após avaliar filmes.
    """
    if user_id in user_items:
        return {"message": f"UserID {user_id} já existe."}
    
    user_items[user_id] = []
    return {"message": f"UserID {user_id} adicionado com sucesso. Ele pode começar a avaliar filmes."}


class Recommendation(BaseModel):
    movie_id: int
    title: str
    predicted_rating: float

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[Recommendation]


@app.get(
    "/recommendations/{user_id}",
    response_model=RecommendationResponse,
    summary="Obter recomendações para um usuário existente na base"
)
async def get_recommendations(user_id: int, top_n: int = 10):
    """
    Gera recomendações de filmes para um usuário presente no conjunto
    de treino (MovieLens). Para novos usuários, use o endpoint de
    adição/avaliação (futuro).
    """
    global model, movie_titles, known_user_ids, min_user_id, max_user_id

    if model is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado.")

    # 1) Validação de existência do usuário na base
    if user_id not in known_user_ids:
        # mensagem útil para você testar
        faixa = (
            f"Tente um UserID entre {min_user_id} e {max_user_id}."
            if min_user_id is not None and max_user_id is not None
            else ""
        )
        raise HTTPException(
            status_code=404,
            detail=f"UserID {user_id} não encontrado na base de treinamento. {faixa}"
        )

    # 2) Geração de recomendações usando o trainset interno da Surprise
    trainset = model.trainset
    raw_uid = str(user_id)  # Surprise geralmente trata IDs como string internamente

    try:
        inner_uid = trainset.to_inner_uid(raw_uid)
    except ValueError:
        # Caso o usuário exista em ratings.dat mas não tenha entrado no trainset por algum motivo
        raise HTTPException(
            status_code=404,
            detail=f"UserID {user_id} não está presente no modelo treinado."
        )

    # Itens que o usuário já avaliou (lista de (inner_iid, rating))
    user_rated_items_inner = set(j for (j, _) in trainset.ur[inner_uid])

    # Recomendar entre todos os itens que ele *não* avaliou
    all_items_inner = range(trainset.n_items)

    predictions = []
    for inner_iid in all_items_inner:
        if inner_iid in user_rated_items_inner:
            continue  # pula filmes que ele já avaliou

        raw_iid = trainset.to_raw_iid(inner_iid)  # MovieID original (string)
        # Surprise trabalha bem com strings, então usamos raw_uid/raw_iid como string
        pred = model.predict(raw_uid, raw_iid)

        movie_id_int = int(raw_iid)
        predictions.append(
            Recommendation(
                movie_id=movie_id_int,
                title=movie_titles.get(movie_id_int, "Desconhecido"),
                predicted_rating=round(pred.est, 3),
            )
        )

    # Ordena pelas maiores notas previstas
    predictions_sorted = sorted(
        predictions, key=lambda x: x.predicted_rating, reverse=True
    )

    return RecommendationResponse(
        user_id=user_id,
        recommendations=predictions_sorted[:top_n],
    )
