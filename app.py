import streamlit as st
import base64
import io
import os
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer, util

# --- CONFIGURATION & MODEL LOADING ---
st.set_page_config(page_title="AI Interior Designer", layout="wide")

@st.cache_resource
def load_models():
    # Loading the same model as your Dash app
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

text_embedding_model = load_models()

# --- MOCK FUNCTIONS (Replace these with your actual generation logic) ---
def build_prompt(room, style, color, lighting, furniture, material):
    return f"A {style} {room} with {color} palette, {lighting} lighting, {furniture} furniture, made of {material}."

def generate_design(room, style, color, lighting, furniture, material):
    # This is a placeholder. Replace with your actual model inference code.
    return Image.new('RGB', (800, 600), color=(100, 150, 200))

def calculate_consistency_score(image, prompt):
    # Placeholder for your consistency logic
    return 0.8524

# --- SESSION STATE INITIALIZATION ---
if 'interesting_images' not in st.session_state:
    st.session_state.interesting_images = []
if 'last_generated_img' not in st.session_state:
    st.session_state.last_generated_img = None
if 'last_prompt' not in st.session_state:
    st.session_state.last_prompt = ""

# --- UI LAYOUT ---
st.title("🏠 AI Interior Designer")

tab1, tab2 = st.tabs(["Design Parameters", "Refine & Analyze"])

with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Configure Design")
        room = st.selectbox("Room Type", ["Living Room", "Bedroom", "Kitchen", "Bathroom", "Office"])
        style = st.selectbox("Design Style", ["Modern", "Minimalist", "Industrial", "Bohemian", "Scandinavian"])
        color = st.selectbox("Color Palette", ["Neutral", "Warm", "Cool", "Earthy", "Monochrome"])
        lighting = st.selectbox("Lighting", ["Natural light", "Warm ambient", "Soft LED"])
        furniture = st.selectbox("Furniture", ["Modern", "Classic", "Vintage", "Mid-Century"])
        material = st.selectbox("Primary Material", ["Wood", "Marble", "Glass", "Metal", "Concrete"])
        
        if st.button("Generate Design", use_container_width=True):
            with st.spinner("Generating your dream room..."):
                img = generate_design(room, style, color, lighting, furniture, material)
                prompt = build_prompt(room, style, color, lighting, furniture, material)
                
                st.session_state.last_generated_img = img
                st.session_state.last_prompt = prompt
                st.success("Design Generated! Switch to 'Refine & Analyze' tab.")

with tab2:
    col_img, col_search = st.columns(2)
    
    with col_img:
        st.subheader("Generated Design")
        if st.session_state.last_generated_img:
            st.image(st.session_state.last_generated_img, use_container_width=True)
            
            score = calculate_consistency_score(st.session_state.last_generated_img, st.session_state.last_prompt)
            st.info(f"**Prompt-Image Consistency Score:** {score:.4f}")
            
            if st.button("Mark as Interesting", use_container_width=True):
                # Calculate embedding for storage
                embedding = text_embedding_model.encode(st.session_state.last_prompt, convert_to_tensor=True)
                
                new_entry = {
                    "image": st.session_state.last_generated_img,
                    "prompt": st.session_state.last_prompt,
                    "score": score,
                    "embedding": embedding
                }
                
                # Avoid duplicates
                if not any(d['prompt'] == st.session_state.last_prompt for d in st.session_state.interesting_images):
                    st.session_state.interesting_images.append(new_entry)
                    st.toast("Saved to Interesting Designs!")
                else:
                    st.warning("Already saved!")
        else:
            st.write("No design generated yet. Use the first tab to start.")

    with col_search:
        st.subheader("Similarity Search")
        search_query = st.text_input("Search for similar designs", placeholder="e.g., modern bedroom with wood")
        
        if st.button("Search") and st.session_state.interesting_images:
            query_embedding = text_embedding_model.encode(search_query, convert_to_tensor=True)
            
            similarities = []
            for item in st.session_state.interesting_images:
                sim = util.cos_sim(query_embedding, item['embedding']).item()
                similarities.append((sim, item))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[0], reverse=True)
            
            for score, item in similarities[:3]:
                with st.expander(f"Match (Score: {score:.4f})"):
                    st.image(item['image'])
                    st.caption(item['prompt'])
        elif not st.session_state.interesting_images:
            st.write("Save some 'Interesting' designs first to search through them.")

# --- GALLERY SECTION ---
st.divider()
st.header("🌟 Interesting Designs Gallery")

if st.session_state.interesting_images:
    # Dropdown to select specific image
    options = [f"Design {i+1}: {img['prompt'][:50]}..." for i, img in enumerate(st.session_state.interesting_images)]
    selected_idx = st.selectbox("Select a design to view in detail", range(len(options)), format_func=lambda x: options[x])
    
    if selected_idx is not None:
        selected_item = st.session_state.interesting_images[selected_idx]
        st.image(selected_item['image'], caption=selected_item['prompt'], width=700)

    # Grid Display
    cols = st.columns(4)
    for idx, item in enumerate(st.session_state.interesting_images):
        with cols[idx % 4]:
            st.image(item['image'], use_container_width=True)
            st.caption(f"Score: {item['score']:.2f}")
else:
    st.info("Your gallery is currently empty.")
