import requests
from config import API_KEY

def get_movie(movie_name):
  url = f"https://api.themoviedb.org/3/search/movie?query={movie_name}&api_key={API_KEY}"
  response = requests.get(url)
  results = response.json().get('results')
  if results:
    movie_id = results[0]["id"] 
    movie_details = get_movie_details(movie_id)
    return movie_details
  else: print("Não existe filme com esse nome.")


def get_movie_details(movie_id):
    print('moviee id', movie_id)
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}"
    response = requests.get(url)
    print(response.json())
    return response.json()

movie_name = input("Entre com o nome do filme: ")
movie_data = get_movie(movie_name)