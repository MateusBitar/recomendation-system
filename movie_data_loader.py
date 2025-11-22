import pandas as pd
import os
from dotenv import load_dotenv

# 1. Carrega variáveis do .env
load_dotenv()

DATA_DIR = os.getenv("DATA_DIR")
MOVIE_DATA_FILENAME = os.getenv("MOVIE_DATA_FILENAME", "movies.dat")

if not DATA_DIR:
    print("❌ ERRO: Variável DATA_DIR não encontrada no .env")
    print("Verifique se o arquivo .env está na mesma pasta que este script.")
    exit()

# Caminho completo para o arquivo de filmes
MOVIE_DATA_PATH = os.path.join(DATA_DIR, MOVIE_DATA_FILENAME)

def load_movie_data():
    """
    Carrega os dados dos filmes e cria um dicionário de mapeamento
    de MovieID para o Título do Filme.
    """
    print(f"Carregando dados dos filmes de: {MOVIE_DATA_PATH}")
    try:
        # MovieLens 1M: separador "::", sem cabeçalho
        movies_df = pd.read_csv(
            MOVIE_DATA_PATH,
            sep="::",
            engine="python",
            names=["MovieID", "Title", "Genres"],
            encoding="latin-1",
        )
        movie_title_map = movies_df.set_index("MovieID")["Title"].to_dict()
        print("Dados dos filmes carregados com sucesso.")
        return movie_title_map
    except FileNotFoundError:
        print(f"ERRO: Arquivo de filmes não encontrado em: {MOVIE_DATA_PATH}")
        return {}

if __name__ == "__main__":
    titles = load_movie_data()
    print(f"Total de filmes carregados: {len(titles)}")
    print(f"Título para MovieID 1: {titles.get(1)}")
