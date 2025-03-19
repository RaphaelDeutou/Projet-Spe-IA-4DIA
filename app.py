import os
import pandas as pd
import numpy as np
import faiss
import sqlite3
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Initialisation de Flask
app = Flask(__name__)

# Charger les données des vols
file_path = "data/dataset_nettoye.csv"  # Mets le bon chemin
df = pd.read_csv(file_path)

# Générer des descriptions de vols pour l'indexation
df["description"] = df.apply(lambda row: f"Vol {row['AIRLINE']} de {row['DISTANCE']} km, départ prévu à {row['SCHEDULED_DEPARTURE']}, prix {row['prix']} euros.", axis=1)

# Initialiser Sentence Transformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(df["description"].tolist(), convert_to_numpy=True).astype(np.float32)

# Créer l'index FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Clé API Gemini (Ajoute ta clé ici)
os.environ["GOOGLE_API_KEY"] = "AIzaSyC6PBfRhdUBCkDtva-gpevC4YTKvXhbTow"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Fonction pour rechercher les vols
def retrieve_flights(query, k=3):
    query_embedding = model.encode([query], convert_to_numpy=True).astype(np.float32)
    distances, indices = index.search(query_embedding, k)
    results = [df.iloc[idx].to_dict() for idx in indices[0]]
    return results

# Fonction pour générer la réponse avec Gemini
def generate_response(query):
    flights = retrieve_flights(query, k=3)
    context = "\n".join([str(flight) for flight in flights])

    prompt = f"Voici les vols disponibles :\n{context}\n\nQuestion : {query}"
    model_gemini = genai.GenerativeModel("models/gemini-2.0-flash")
    response = model_gemini.generate_content(prompt, generation_config={"temperature": 0.1})

    return response.text

# Route pour l'interface web
@app.route('/')
def home():
    return render_template("index.html")

# API pour interagir avec le chatbot
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get("message", "")
    response = generate_response(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
