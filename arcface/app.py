import joblib
import numpy as np
import streamlit as st
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

embeddings_db = joblib.load("embeddings.pkl")
names_db = joblib.load("names.pkl")


# Page config
st.set_page_config(page_title="Which Celebrity Are You?", layout="centered")

st.title("🎭 Which Indian Celebrity Are You?")
st.write("Upload your photo and find your celebrity twin!")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

def find_match(query_embedding, threshold=0.4):

    similarities = cosine_similarity([query_embedding], embeddings_db)[0]

    best_index = np.argmax(similarities)
    best_score = similarities[best_index]
    best_name = names_db[best_index]
    matched_path = image_paths[best_index]
    matched_image = Image.open(matched_path)

    if best_score >= threshold:
        return best_name, best_score, best_image
    else:
        return "No Match Found", best_score, None



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

        name, score, celeb_image = find_match(embedding)

    st.success(f"✨ You look like: {name}")
    st.write(f"Similarity Score: {round(score, 3)}")
    col1, col2, col3 = st.columns([1,2,1])

    with col1:
        st.image(celeb_image, caption=name, width=250)

