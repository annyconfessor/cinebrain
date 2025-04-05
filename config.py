import requests
import pandas as pd
import torch

API_KEY = 'bad4a878b132ffdf9c7132503dc21bc8'
url = f"https://api.themoviedb.org/3/movie/popular?api_key={API_KEY}&language=pt-BR&page=1"

response = requests.get(url)
data = response.json()

for movie in data['results']:
  print(movie['title'], movie['vote_average'], movie['genre_ids'])
  print('movies')

movies = [{'title': m['title'], 'rating': m['vote_average'], 'genres': m['genre_ids']} for m in data['results']]
df = pd.DataFrame(movies)

# Codificando gêneros como features binárias
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
genre_features = mlb.fit_transform(df['genres'])
genre_df = pd.DataFrame(genre_features, columns=mlb.classes_)

# Concatenar com a nota
df_final = pd.concat([genre_df, df['rating']], axis=1)

X = torch.tensor(df_final.drop(columns=['rating']).values, dtype=torch.float32)
y = torch.tensor(df_final['rating'].values, dtype=torch.float32).view(-1, 1)
