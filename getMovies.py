from config import data

all_movies = []
for page in range(1, 4):
    all_movies.extend(data)

print(all_movies)