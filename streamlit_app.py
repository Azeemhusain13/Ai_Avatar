# ==============================
# streamlit_app.py
# ==============================

import streamlit as st
import torch
from diffusers.pipelines.stable_diffusion import StableDiffusionImg2ImgPipeline
from PIL import Image
import os
from huggingface_hub import login
import warnings

warnings.filterwarnings("ignore")

# -----------------------
# AUTH
# -----------------------

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)

# -----------------------
# PAGE CONFIG
# -----------------------

st.set_page_config(page_title="AI Avatar Agent", layout="centered")
st.title("🚀 AI Avatar Agent")

# -----------------------
# MODEL SELECTION
# -----------------------

model_choice = st.selectbox(
    "Choose Model",
    [
        "Stable Diffusion (fast)",
        "DreamShaper (better quality)",
        "OpenJourney (Midjourney style)"
    ]
)

if model_choice == "DreamShaper (better quality)":
    model_id = "Lykon/dreamshaper-8"
elif model_choice == "OpenJourney (Midjourney style)":
    model_id = "prompthero/openjourney"
else:
    model_id = "runwayml/stable-diffusion-v1-5"

# -----------------------
# LOAD MODEL (CPU SAFE)
# -----------------------

@st.cache_resource(show_spinner="Loading AI model...")
def load_model(model_id, hf_token):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        token=hf_token
    )

    pipe = pipe.to(device)
    pipe.enable_attention_slicing()

    return pipe

pipe = load_model(model_id, hf_token)

# -----------------------
# IMAGE UPLOAD
# -----------------------

uploaded_file = st.file_uploader("Upload your image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    if uploaded_file.size > 5 * 1024 * 1024:
        st.error("File too large. Please upload under 5MB.")
        st.stop()

    image = Image.open(uploaded_file)
    image = image.convert("RGB")
    image.thumbnail((512, 512))

    input_image = image

    st.image(input_image, caption="Input Image")

    # -----------------------
    # GENERATE AVATAR
    # -----------------------

    if st.button("Generate Avatar"):
        with st.spinner("Generating avatar..."):

            prompt = (
                "high quality avatar portrait of the same person, "
                "sharp facial features, ultra detailed, realistic lighting, cinematic"
            )

            negative_prompt = (
                "blurry, distorted face, bad anatomy, extra limbs, low quality"
            )

            torch.cuda.empty_cache()

            with torch.no_grad():
                avatar = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=input_image,
                    strength=0.55,
                    guidance_scale=7,
                    num_inference_steps=10
                ).images[0]

            avatar_path = "avatar.png"
            avatar.save(avatar_path)

            st.session_state["avatar_path"] = avatar_path

            st.image(avatar, caption="Generated Avatar")

# -----------------------
# DOWNLOAD BUTTON
# -----------------------

if "avatar_path" in st.session_state and os.path.exists(st.session_state["avatar_path"]):
    with open(st.session_state["avatar_path"], "rb") as f:
        st.download_button(
            label="📥 Download Avatar",
            data=f,
            file_name="avatar.png",
            mime="image/png"
        )

# -----------------------
# BEFORE / AFTER
# -----------------------

if "avatar_path" in st.session_state and uploaded_file:
    st.subheader("Before vs After")

    col1, col2 = st.columns(2)

    with col1:
        st.image(input_image, caption="Before")

    with col2:
        latest_avatar = Image.open(st.session_state["avatar_path"]).convert("RGB")
        st.image(latest_avatar, caption="After")

# -----------------------
# COMMAND PARSER
# -----------------------

def parse_command(cmd):
    cmd = cmd.lower()

    if "glass" in cmd or "spec" in cmd:
        return "wearing eyeglasses"
    if "tie" in cmd:
        return "wearing a formal tie"
    if "hair" in cmd:
        return "modern hairstyle"
    if "background" in cmd:
        return "tropical beach background"
    if "clothes" in cmd:
        return "stylish clothes"

    return cmd

# -----------------------
# QUICK BUTTONS
# -----------------------

if "avatar_path" in st.session_state:
    st.subheader("Quick Edits")

    col1, col2, col3 = st.columns(3)

    if col1.button("👓 Glasses"):
        st.session_state["command"] = "add glasses"
    if col2.button("👔 Formal"):
        st.session_state["command"] = "add tie"
    if col3.button("🏝️ Beach"):
        st.session_state["command"] = "change background"

# -----------------------
# EDIT SECTION
# -----------------------

if "avatar_path" in st.session_state:
    st.subheader("Edit Avatar")

    command = st.text_input("Enter edit command", value=st.session_state.get("command", ""))

    if st.button("Apply Edit") and command:

        st.info("⏳ Applying your edit... please wait")

        progress_bar = st.progress(0)

        current_avatar = Image.open(st.session_state["avatar_path"]).convert("RGB").resize((512, 512))

        edit_prompt = parse_command(command)

        final_prompt = (
            f"same person, consistent face, {edit_prompt}, "
            "high detail, realistic, preserve identity"
        )

        torch.cuda.empty_cache()

        with st.spinner("✨ Generating edited avatar..."):
            for i in range(1, 6):
                progress_bar.progress(i * 20)

            with torch.no_grad():
                edited = pipe(
                    prompt=final_prompt,
                    negative_prompt="blurry, distorted face",
                    image=current_avatar,
                    strength=0.6,
                    guidance_scale=7,
                    num_inference_steps=10
                ).images[0]

        progress_bar.progress(100)

        edited.save(st.session_state["avatar_path"])

        st.success("✅ Edit applied successfully!")

        st.image(edited, caption="Updated Avatar")
