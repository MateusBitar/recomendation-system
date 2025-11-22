import pandas as pd
import os

# Defina o caminho para o arquivo movies.dat do MovieLens 1M
# Ajuste o caminho se necessário, usando o mesmo prefixo de diretório base
MOVIE_DATA_PATH = 'D:/mateu/Documents/Sistema de recomendação/final/data/movies.dat' 

def load_movie_data():
    """
    Carrega os dados dos filmes e cria um dicionário de mapeamento
    de MovieID para o Título do Filme.
    """
    print(f"Carregando dados dos filmes de: {MOVIE_DATA_PATH}")
    try:
        # O arquivo movies.dat usa "::" como separador e não tem cabeçalho
        # Colunas: MovieID, Title, Genres
        movies_df = pd.read_csv(
            MOVIE_DATA_PATH, 
            sep='::', 
            engine='python', 
            names=['MovieID', 'Title', 'Genres']
        )
        # Cria um dicionário de mapeamento {MovieID: Title}
        movie_title_map = movies_df.set_index('MovieID')['Title'].to_dict()
        print("Dados dos filmes carregados com sucesso.")
        return movie_title_map
    except FileNotFoundError:
        print(f"ERRO: Arquivo de filmes não encontrado em: {MOVIE_DATA_PATH}")
        return {}

if __name__ == '__main__':
    # Teste de carregamento
    titles = load_movie_data()
    print(f"Total de filmes carregados: {len(titles)}")
    # Exemplo: O filme 1 é Toy Story (1995) no MovieLens
    print(f"Título para MovieID 1: {titles.get(1)}")