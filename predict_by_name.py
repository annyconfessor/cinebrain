import torch
import requests
from config import API_KEY
from model import RecommendedMovies

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

movie_name = input("Entre com o nome do filme: ")
movie_data = search_movie(movie_name)

if movie_data:
   details = get_movie_details(movie_data["id"])
   genres = []
   for g in details.get("genres", []):
      genres.append(g["id"])
  #  genres = [g["id"] for g in details.get("genres", [])]
   popularity = details.get("popularity", 0.0)
   print(f"Gêneros: {genres} | Popularidade: {popularity:.2f}")

   input_tensor = process_input(genres, popularity)

  # De maneira geral, esse bloco de código pega um modelo vazio e preenche com modelos treinados
   model = RecommendedMovies(input_size=14)
  #  carrega e injeta o valor de model.pth no modelo criado, é tipo um rest do js
  # o modelo onde o dado vai ser injetado tem que ser do mesmo tamanho que o dado carregado
   model.load_state_dict(torch.load("model.pth")) 
  #  entra no modo de avaliacao/inferencia
   model.eval()

  # para informar que nao estamos treinando o modelo
   with torch.no_grad():
      predicted_rating = model(input_tensor).item()
      if predicted_rating:
         print(f"Nota prevista para '{movie_name}': {predicted_rating:.2f}")
      else:
         print("Filme não encontrado")

process_input([28,35], 5.0)

