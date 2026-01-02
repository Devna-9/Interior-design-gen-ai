import streamlit as st
import openai
from PIL import Image
import requests
from io import BytesIO

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(
    page_title="AI Interior Designer",
    layout="centered"
)

# Use environment variable for API key
openai.api_key = st.secrets["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"]
# OR locally:
# openai.api_key = "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"

# -------------------------------
# BUDGET LOGIC
# -------------------------------
def budget_description(budget):
    if budget == "Low":
        return "low-cost materials, compact furniture, simple decor, laminate flooring"
    elif budget == "Medium":
        return "mid-range materials, modular furniture, wooden finishes, balanced decor"
    else:
        return "luxury materials, marble flooring, custom furniture, designer decor"

# -------------------------------
# PROMPT ENGINEERING
# -------------------------------
def build_prompt(room, style, color, lighting, budget_desc, furniture_style, material):
    return f"""
Ultra-realistic {style.lower()} {room.lower()} interior design.
Designed with {budget_desc}.
Dominant {color.lower()} color palette.
{furniture_style.lower()} furniture with accurate proportions.
{lighting.lower()} lighting with realistic shadows.
High-quality {material.lower()} textures and finishes.
Global illumination, DSLR photography, 35mm lens.
Photorealistic, ultra-detailed, 8K resolution.
No people, no text, no watermark.
"""

# -------------------------------
# IMAGE GENERATION
# -------------------------------
def generate_image(prompt):
    response = openai.Image.create(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024"
    )
    return response["data"][0]["url"]

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.title("AI Interior Designer")
st.write("Generate ultra-realistic interior designs using DALL·E 3")

room = st.selectbox("Room Type",
                    ["Living Room", "Bedroom", "Kitchen", "Bathroom", "Office", "Dining Room"])

style = st.selectbox("Design Style",
                     ["Modern", "Minimalist", "Industrial", "Bohemian",
                      "Scandinavian", "Traditional", "Art Deco", "Rustic"])

color = st.selectbox("Color Palette",
                     ["Neutral", "Warm", "Cool", "Earthy", "Monochrome", "Vibrant"])

lighting = st.selectbox("Lighting Style",
                        ["Natural light", "Warm ambient", "Soft LED", "Accent lighting"])

furniture_style = st.selectbox("Furniture Style",
                               ["Modern", "Classic", "Rustic", "Minimalist",
                                "Industrial", "Vintage", "Mid-Century Modern"])

material = st.selectbox("Primary Material",
                         ["Wood", "Marble", "Glass", "Metal", "Concrete", "Fabric"])

budget = st.select_slider(
    "Budget Level",
    options=["Low", "Medium", "High"],
    value="Medium"
)

# -------------------------------
# GENERATE BUTTON
# -------------------------------
if st.button("Generate Design"):
    with st.spinner("Designing your interior..."):
        budget_desc = budget_description(budget)
        prompt = build_prompt(
            room, style, color, lighting,
            budget_desc, furniture_style, material
        )

        image_url = generate_image(prompt)
        image = Image.open(BytesIO(requests.get(image_url).content))

        st.image(image, caption="AI-Generated Interior Design", use_column_width=True)
        st.success("Design generated successfully!")

        st.subheader("Prompt Used")
        st.code(prompt)
