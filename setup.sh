#!/bin/bash

echo "🚀 Starting GPU instance setup..."

# --- Update system ---
echo "📦 Updating system..."
sudo apt update -y

# --- Install Python tools ---
echo "🐍 Installing pip + venv..."
sudo apt install -y python3-pip python3-venv git


# --- Create virtual environment ---
echo "🌱 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# --- Upgrade pip ---
pip install --upgrade pip

# --- Install dependencies ---
echo "📚 Installing dependencies..."
pip install -r requirements.txt
pip install streamlit torchvision accelerate huggingface_hub

# --- Fix known issue (jinja2 error) ---
pip install "jinja2>=3.1.0"

# --- Login Hugging Face ---
echo "🔐 Logging into Hugging Face..."
echo "$HUGGINGFACE_HUB_TOKEN" | huggingface-cli login --token

echo "✅ Setup complete!"