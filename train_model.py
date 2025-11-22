import pandas as pd
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
from surprise.dump import dump
import os
from dotenv import load_dotenv

# 1. Carrega variáveis do .env
load_dotenv()

DATA_DIR = os.getenv("DATA_DIR")
MODELS_DIR = os.getenv("MODELS_DIR")
MOVIE_DATA_FILENAME = os.getenv("MOVIE_DATA_FILENAME", "movies.dat")
MODEL_FILENAME = os.getenv("MODEL_FILENAME", "svd_model.pkl")

if not DATA_DIR or not MODELS_DIR:
    print("❌ ERRO: Variáveis DATA_DIR ou MODELS_DIR não encontradas no .env")
    print("Verifique se o arquivo .env está na mesma pasta que este script.")
    exit()

# Constrói os caminhos COMPLETOS
RATINGS_PATH = os.path.join(DATA_DIR, "ratings.dat")
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_FILENAME)

# Cria a pasta 'models' se ela não existir
os.makedirs(MODELS_DIR, exist_ok=True)


## 🚀 Etapa 2.1: Leitura e Preparação dos Dados

print("1. Iniciando a leitura e preparação dos dados...")
print(f"   -> Lendo de: {RATINGS_PATH}")

try:
    # O MovieLens 1M usa "::" como separador
    ratings_df = pd.read_csv(
        RATINGS_PATH, 
        sep='::', 
        engine='python', 
        names=['UserID', 'MovieID', 'Rating', 'Timestamp'],
        encoding='latin-1' # Adicionado para evitar erros de caracteres
    )
except FileNotFoundError:
    print(f"\n❌ ERRO: Arquivo não encontrado em: {RATINGS_PATH}")
    print("Verifique se 'ratings.dat' está dentro da pasta indicada no .env (DATA_DIR).")
    exit()

# Manter apenas as colunas essenciais
ratings_df = ratings_df[['UserID', 'MovieID', 'Rating']]

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[['UserID', 'MovieID', 'Rating']], reader)

print(f"   -> Dados carregados com sucesso: {len(ratings_df)} avaliações.")
print("------------------------------------------------------------------")


## ⚙️ Etapa 2.2: Treinamento e Avaliação do Modelo

algo = SVD(n_factors=100, n_epochs=20, random_state=42)

print("2. Avaliando o modelo SVD com Cross-Validation (5 folds)...")

cv_results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True, n_jobs=-1)

avg_rmse = cv_results['test_rmse'].mean()
avg_mae = cv_results['test_mae'].mean()

print(f"\n   -> RMSE Médio: {avg_rmse:.4f}")
print(f"   -> MAE Médio: {avg_mae:.4f}")
print("------------------------------------------------------------------")


## 💾 Etapa 2.3: Treinamento Final e Persistência

print("3. Treinando o modelo final...")

full_trainset = data.build_full_trainset()
algo.fit(full_trainset)

dump(MODEL_PATH, algo=algo)

print(f"   -> Modelo salvo em: {MODEL_PATH}")
print("\n✅ Treinamento concluído!")