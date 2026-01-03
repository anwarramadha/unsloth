sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev

python3.11 -m venv unsloth
source unsloth/bin/activate

pip install --upgrade pip setuptools wheel

# Core dependencies for dataset generation, RAG, and inference
pip install markdown beautifulsoup4 faiss-cpu sentence-transformers numpy vllm accelerate torch

# Transformers (specific version for SahabatAI compatibility)
pip install transformers==4.45.0

# Install Unsloth for training
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft xformers bitsandbytes

echo ""
echo "✅ Installation complete!"
echo ""
echo "📝 IMPORTANT: To activate the virtual environment, run:"
echo "   source unsloth/bin/activate"
echo ""
echo "💡 Or run this script with 'source' to auto-activate:"
echo "   source ./install-dependencies.sh"
echo ""
