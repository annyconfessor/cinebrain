import torch
from model import RecommendedMovies
import pandas as pd

genre_map = {
  28: 'Ação',
  35: 'Comédia',
  18: 'Drama',
  10749: 'Romance',
  878: 'Ficção Cientifica',
}

def process_input(genres_ids, popularity):
  all_genres = sorted(genre_map.keys())
  genre_vector = [1 if grid in genres_ids else 0 for grid in all_genres]
  return torch.tensor(genre_vector + [popularity], dtype=torch.float32).unsqueeze(0)

model = RecommendedMovies(input_size=6)
model.load_state_dict(torch.load("model.py"))
model.eval()

genres = [28,878]
popularity = 50.0

input_tensor = process_input(genres, popularity)

# torch.no_grad() é usado apenas para dizer ao pytorch que vamos fazer uma previsão e não um treinamento
with torch.no_grad():
  predict_rating = model(input_tensor).item()

print(f"Nota prevista para o filme: {predict_rating:.2f}")