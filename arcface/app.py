import os
import random
import joblib
import numpy as np
import streamlit as st
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# Load data
embeddings_db = joblib.load("embeddings.pkl")
names_db = joblib.load("names.pkl")

# Page config
st.set_page_config(page_title="Which Celebrity Are You?", layout="centered")

st.title("🎭 Which Indian Celebrity Are You?")
st.write("Upload your photo and find your celebrity twin!")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

DATASET_FOLDER = "Indian-actors-faces"


def find_match(query_embedding, threshold=0.4):

    similarities = cosine_similarity([query_embedding], embeddings_db)[0]

    best_index = np.argmax(similarities)
    best_score = similarities[best_index]
    best_name = names_db[best_index]

    if best_score >= threshold:
        return best_name, best_score
    else:
        return "No Match Found", best_score


if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    # Resize while keeping aspect ratio
    image.thumbnail((300, 300))

    # Show uploaded image
    st.image(image, caption="Your Image", width=300)

    with st.spinner("Analyzing face..."):

        embedding = DeepFace.represent(
            img_path=np.array(image),
            model_name="ArcFace",
            enforce_detection=False
        )[0]["embedding"]

        name, score = find_match(embedding)

    st.success(f"✨ You look like: {name}")
    st.write(f"Similarity Score: {round(score, 3)}")

    # If match found, show random image from that celebrity folder
    if name != "No Match Found":

        person_folder = os.path.join(DATASET_FOLDER, name)
        image_files = os.listdir(person_folder)

        if image_files:
            random_image = random.choice(image_files)
            full_path = os.path.join(person_folder, random_image)

            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                st.image(full_path, caption=name, width=250)
