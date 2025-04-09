import torch
from model import RecommendedMovies
from utils import search_movie, get_movie_details, process_input

movie_name = input("Entre com o nome do filme: ")
movie_data = search_movie(movie_name)

if movie_data:
   details = get_movie_details(movie_data["id"])
   print("details", details)
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