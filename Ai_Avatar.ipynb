# AI Avatar Agent (Stable + Persistent + Multi-Model)
# ============================================================

!pip install diffusers transformers accelerate pillow matplotlib -q

import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import matplotlib.pyplot as plt
from google.colab import files

# ------------------------------------------------------------
# DEVICE SETUP
# ------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on:", device)

# ------------------------------------------------------------
# MODEL SELECTION
# ------------------------------------------------------------

print("\nChoose Model:")
print("1. Stable Diffusion (fast)")
print("2. DreamShaper (better quality)")
print("3. OpenJourney (Midjourney style)")

choice = input("Enter choice (1/2/3): ")

if choice == "2":
    model_id = "Lykon/dreamshaper-8"
elif choice == "3":
    model_id = "prompthero/openjourney"
else:
    model_id = "runwayml/stable-diffusion-v1-5"

print("Loading model:", model_id)

# ------------------------------------------------------------
# LOAD MODEL (CPU/GPU SAFE)
# ------------------------------------------------------------

if device == "cuda":
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    ).to(device)
else:
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id).to(device)

pipe.enable_attention_slicing()

# Fix randomness (consistent edits)
generator = torch.manual_seed(42)

# ------------------------------------------------------------
# IMAGE UPLOAD
# ------------------------------------------------------------

print("\nUpload your image")
uploaded = files.upload()

image_path = list(uploaded.keys())[0]

input_image = Image.open(image_path).convert("RGB").resize((512,512))

plt.imshow(input_image)
plt.title("Input Image")
plt.axis("off")
plt.show()

# ------------------------------------------------------------
# GENERATE AVATAR
# ------------------------------------------------------------

print("\nGenerating avatar...")

base_prompt = "high quality cartoon avatar portrait of the same person, sharp face, detailed, realistic lighting"
negative_prompt = "blurry, distorted face, bad anatomy, low quality"

avatar = pipe(
    prompt=base_prompt,
    negative_prompt=negative_prompt,
    image=input_image,
    strength=0.55,
    guidance_scale=7,
    num_inference_steps=20,
    generator=generator
).images[0]

# SAVE INITIAL AVATAR
avatar_path = "avatar.png"
avatar.save(avatar_path)

print("Avatar saved as:", avatar_path)

plt.imshow(avatar)
plt.title("Generated Avatar")
plt.axis("off")
plt.show()

# ------------------------------------------------------------
# COMMAND PARSER
# ------------------------------------------------------------

def parse_command(cmd):

    cmd = cmd.lower()

    if "glass" in cmd or "spec" in cmd:
        return "wearing eyeglasses, clear visible glasses"

    if "tie" in cmd:
        return "wearing a formal tie, clearly visible"

    if "hair" in cmd:
        return "with modern hairstyle, detailed hair"

    if "background" in cmd:
        return "with tropical beach background, bright lighting"

    if "clothes" in cmd or "shirt" in cmd:
        return "wearing stylish colorful clothes"

    return cmd

# ------------------------------------------------------------
# EDIT LOOP (PERSISTENT SYSTEM)
# ------------------------------------------------------------

print("\n🚀 Avatar Editing Agent Ready")
print("Commands you can try:")
print(" add glasses")
print(" add tie")
print(" change hair")
print(" change clothes")
print(" change background to beach")
print(" exit")

while True:

    command = input("\nEnter edit command: ")

    if command.lower() == "exit":
        break

    # 🔥 ALWAYS LOAD LATEST SAVED IMAGE
    current_avatar = Image.open(avatar_path).convert("RGB").resize((512,512))

    edit_prompt = parse_command(command)

    final_prompt = f"portrait of the same person {edit_prompt}, high detail, consistent identity"
    negative_prompt = "blurry, distorted face, bad anatomy, low quality"

    print("Applying:", final_prompt)

    edited = pipe(
        prompt=final_prompt,
        negative_prompt=negative_prompt,
        image=current_avatar,
        strength=0.6,
        guidance_scale=7,
        num_inference_steps=20,
        generator=generator
    ).images[0]

    # 🔥 SAVE UPDATED VERSION
    edited.save(avatar_path)

    print("Updated avatar saved.")

    plt.imshow(edited)
    plt.title("Updated Avatar")
    plt.axis("off")
    plt.show()

print("Session ended.")
