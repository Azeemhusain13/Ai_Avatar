# 🚀 AI Avatar Generation & Editing Agent

Working Project (Hugging Face Space): https://huggingface.co/spaces/Azeem123456/Ai_Avatar_Generator

## 📌 Overview
This project implements an **AI-powered avatar generation and editing agent** that allows users to:

- Generate a stylized avatar from an input image
- Iteratively modify the avatar using natural language commands
- Preserve facial identity while editing attributes like:
  - Glasses
  - Clothing
  - Hairstyle
  - Background

The system is built using **Stable Diffusion models from Hugging Face** and runs on **Google Colab (CPU/GPU)**.

---

## 🎯 Features

### ✅ Avatar Generation
- Upload a user image
- Generate a stylized avatar
- Preserve facial identity

### ✅ Editable Attributes
Modify avatar using commands like:
- `add glasses`
- `add tie`
- `change hair`
- `change clothes`
- `change background to beach.`

### ✅ Iterative Editing
- Supports multiple sequential edits
- Maintains state using persistent image saving

### ✅ Multi-Model Support
Choose from:
- Stable Diffusion (fast)
- DreamShaper (high quality)
- OpenJourney (Midjourney-like style)

---

## 🧠 System Architecture


User Image
↓
Avatar Generator (Stable Diffusion)
↓
Saved Avatar (Persistent State)
↓
Command Parser
↓
Edit Prompt Generator
↓
Diffusion Model (Img2Img)
↓
Updated Avatar


---

## ⚙️ Tech Stack

- Python
- Hugging Face Diffusers
- Stable Diffusion Models
- PyTorch
- Google Colab
- PIL (Image Processing)
- Matplotlib

---

## 🚀 Installation

Run the following command:

```bash
pip install diffusers transformers accelerate pillow matplotlib
▶️ Usage
1. Run the script
python avatar_agent.py
2. Upload your image

Select an image when prompted.

3. Choose a model
1 → Stable Diffusion
2 → DreamShaper
3 → OpenJourney
4. Generate avatar
5. Enter edit commands

Example:

add glasses
add tie
change hair
change background to beach
🔁 How Editing Works

The generated avatar is saved locally

Every new edit:
Loads the latest saved avatar
Applies modification
Saves updated version

This ensures:
✔ Consistency
✔ Identity preservation
✔ Reliable editing

📊 Performance
Task	Time
Avatar Generation	~10–30 sec
Avatar Editing	~5–15 sec

(depending on CPU/GPU)

⚠️ Limitations

Edits are based on prompt conditioning (not true region editing)
CPU execution is slower
Complex edits may affect identity slightly

🔥 Future Improvements
Inpainting (edit specific regions only)
Face-locking using IP-Adapter
Automatic region detection (eyes, hair, clothes)
LLM-based command understanding
Streamlit UI for better interaction

💡 Key Design Decision

The system uses a persistent avatar state by saving the generated image and applying all edits to the latest version. This ensures consistent results and prevents identity drift.

👨‍💻 Author:
Azeem Husain Khan
LinkedIN: https://www.linkedin.com/in/azeem-husain-khan-129a041b5/



