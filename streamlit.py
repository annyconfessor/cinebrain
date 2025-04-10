import streamlit as st
import torch
from model import RecommendedMovies
from utils import search_movie, get_movie_details, process_input
from PIL import Image
import requests
from io import BytesIO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RecommendedMovies(input_size=14)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

st.set_page_config(page_title="CineBrain 🎥", layout="centered")

st.title("🎬 Previsor de notas de filmes")
st.write("Digite o nome de um filme e veja qual nota o modelo prevê!")

movie_name = st.text_input("Nome do filme: ")

if movie_name: 
  movie_data = search_movie(movie_name)
  
  if movie_data:
    details = get_movie_details(movie_data["id"])

    if details:
      st.subheader(details["title"])
      st.markdown(f"**Sinopse:** {details['overview']}")
      st.write("📊 Popularidade:", details["popularity"])
      st.write("🎞️ Gêneros:", ", ".join(g["name"] for g in details["genres"]))
      genres_ids = (g["id"] for g in details["genres"])

      input_tensor = process_input(genres_ids, details["popularity"])
      input_tensor = input_tensor.to(device)

      with torch.no_grad():
        output = model(input_tensor)
        predicted_rating = output.item()

    st.success(f"⭐ Nota prevista: **{predicted_rating:.2f} / 10**")
  
  else: st.warning("Filme não encontrado.")

else: st.info("Digite o nome do filme")