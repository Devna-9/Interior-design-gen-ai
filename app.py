import streamlit as st
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ PAGE CONFIG ------------------
st.set_page_config(  page_title="AI Interior Designer",  layout="wide")

st.title("🛋️ AI Interior Designer with Prompt Accuracy")
st.caption("Stable Diffusion Inpainting + CLIP-based prompt accuracy")

# ------------------ DEVICE ------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------ LOAD MODELS ------------------
@st.cache_resource
def load_sd_pipeline():
    pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    return pipe.to(device)

@st.cache_resource
def load_clip():
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return clip_model, clip_processor

pipe = load_sd_pipeline()
clip_model, clip_processor = load_clip()

# ------------------ SIDEBAR ------------------
st.sidebar.header("🧩 Controls")

guidance_scale = st.sidebar.slider("Guidance Scale", 5, 20, 15)
num_images = st.sidebar.slider("Number of Images", 1, 6, 4)
seed = st.sidebar.number_input("Seed (change for variety)", value=0, step=1)

# ------------------ IMAGE INPUT ------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Original Image")
    init_image_file = st.file_uploader("Image", type=["jpg", "png"])

with col2:
    st.subheader("Upload Mask (white = change)")
    mask_image_file = st.file_uploader("Mask", type=["png"])

if not init_image_file or not mask_image_file:
    st.warning("Upload both image and mask to proceed.")
    st.stop()

init_image = Image.open(init_image_file).convert("RGB")
mask_image = Image.open(mask_image_file).convert("RGB")

st.subheader("Original & Mask")
st.image([init_image, mask_image], caption=["Original", "Mask"], width=350)

# ------------------ PROMPT INPUT ------------------
st.subheader("✏️ Prompt for Interior Change")
prompt = st.text_input("Describe what you want to change", placeholder="white calacatta marble kitchen island and wall")

generate_btn = st.button("🎨 Generate Design")

# ------------------ GENERATION ------------------
if generate_btn and prompt.strip():
    generator = torch.Generator(device=device).manual_seed(seed)

    with st.spinner("Generating interior designs..."):
        images = pipe(     prompt=prompt,   image=init_image,   mask_image=mask_image,   guidance_scale=guidance_scale,   num_images_per_prompt=num_images,   generator=generator        ).images

    # ------------------ CLIP ACCURACY ------------------
    def clip_score(image, text):
        inputs = clip_processor(text=[text], images=image, return_tensors="pt", padding=True  ).to(device)

        with torch.no_grad():
            outputs = clip_model(**inputs)

        image_emb = outputs.image_embeds.cpu().numpy()
        text_emb = outputs.text_embeds.cpu().numpy()
        return cosine_similarity(image_emb, text_emb)[0][0]

    scores = [clip_score(img, prompt) for img in images]

    # ------------------ DISPLAY ------------------
    st.subheader("🖼️ Generated Designs")

    selected_idx = st.selectbox("Select Image", options=list(range(len(images))),  format_func=lambda i: f"Design {i+1} | Accuracy: {scores[i]:.3f}")

    st.image(images[selected_idx], width=600)

    st.markdown("### 📊 Prompt–Image Accuracy")
    st.success(f"CLIP Similarity Score: **{scores[selected_idx]:.3f}**")

    st.caption("Higher score means the generated image aligns better with your prompt. " "This is CLIP-based semantic similarity, not pixel matching.")

else:
    st.info("Enter a prompt and click Generate.")
