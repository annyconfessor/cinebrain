import requests
import torch
from config import API_KEY

genre_map = {
  28: 'Ação',
  35: 'Comédia',
  18: 'Drama',
  10749: 'Romance',
  878: 'Ficção',
  21: 'Ficção Cientifica',
  344: 'Triller',
  32: 'Horror',
  12: 'Anime',
  837: 'Sci-fi',
  1312: 'Aventura',
  983: 'Suspense',
  91: 'TV Show'
}

def search_movie(movie_name):
  url = f"https://api.themoviedb.org/3/search/movie?query={movie_name}&api_key={API_KEY}"
  response = requests.get(url)
  results = response.json().get('results')
  if results:
    movie_id = results[0]["id"] 
    return get_movie_details(movie_id)
  else: print("Não existe filme com esse nome.")

def get_movie_details(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}"
    response = requests.get(url)
    return response.json()

def process_input(genres_ids, popularity):
    all_genres = sorted(genre_map.keys())
    # "grids" são as combinações de parâmetros nesse conjunto de dados. No caso, de generos e suas popularidades. É a combinação de hiperparâmetros
    # genre_vector = [1 if gid in genres_ids else 0 for gid in all_genres] 
    genre_vector = []
    for grid in all_genres:
      if grid in genres_ids:
        genre_vector.append(1)
      else:
        genre_vector.append(0)
    
    # input_vector = genre_vector + [popularity]

    # esse unsqueeze é para simular um batch
    torchx = torch.tensor(genre_vector + [popularity], dtype=torch.float32).unsqueeze(0)
    return torchx