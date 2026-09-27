Neural Lossless Steganography (`NeuralLosslessStego`)
An end-to-end Machine Learning pipeline for lossless image/data steganography using Deep Learning architectures. This project embeds secret payload data into cover images without noticeable visual distortion and retrieves the hidden data with high fidelity.
---
📌 Project Overview
Steganography is the science of hiding secret data within ordinary, non-secret files or media to avoid detection.
`NeuralLosslessStego` leverages neural networks to learn custom encoding and decoding transformations:
Encoder Network: Blends secret input payloads into cover images while minimizing visual reconstruction loss.
Decoder Network: Extracts the embedded payload from stego images back into its original lossless format.
Dockerized Deployment: Fully containerized setup ensuring predictable execution environments.
---
📁 Repository Structure
```text
NeuralLosslessStego/
├── ml_core/          # Core machine learning models, architecture definitions, and helper utilities
├── app1.py           # Main application interface / inference execution script
├── Dockerfile        # Container specification file for containerized execution
├── .dockerignore     # Specifies files and patterns to ignore during Docker builds
├── inputs/           # (Local) Directory for input cover/secret sample assets
└── outputs/          # (Local) Directory for generated stego assets and logs
```
---
🛠️ Tech Stack & Requirements
Language: Python 3.x
Deep Learning Frameworks: PyTorch / TensorFlow / Keras
Containerization: Docker
Libraries: OpenCV, NumPy, PIL (Pillow), Matplotlib
---
🚀 Getting Started
1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/NeuralLosslessStego.git
cd NeuralLosslessStego
```
2. Set Up Python Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```
3. Run the Application
To launch the steganography pipeline:
```bash
python app1.py
```
---
🐳 Running with Docker
You can build and containerize the project using Docker:
```bash
# Build the Docker image
docker build -t neural-lossless-stego .

# Run the Docker container
docker run -p 5000:5000 neural-lossless-stego
```
---
👤 Author
Jaiganapathi Senthil Bhuvaneswari
Student @ Sathyabama Institute of Science & Technology
