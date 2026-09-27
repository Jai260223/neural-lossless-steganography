import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from engine import LosslessStegoINN

# Setup device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_trained_model():
    model = LosslessStegoINN(channels=6, num_blocks=4).to(DEVICE)
    weights_path = "ml_core/best_stego_model.pth"
    if not os.path.exists(weights_path):
        raise FileNotFoundError("Trained model weights not found! Run train.py first.")
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=False))
    model.eval()
    return model

def compute_psnr(original_path, extracted_path):
    orig = np.array(Image.open(original_path).convert("RGB").resize((256, 256)), dtype=np.float32)
    ext = np.array(Image.open(extracted_path).convert("RGB"), dtype=np.float32)
    
    mse = np.mean((orig - ext) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def hide_image(carrier_path, secret_path, output_path):
    model = load_trained_model()
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    carrier_img = Image.open(carrier_path).convert("RGB")
    secret_img = Image.open(secret_path).convert("RGB")
    
    carrier_tensor = transform(carrier_img).unsqueeze(0).to(DEVICE)
    secret_tensor = transform(secret_img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        stego_carrier_tensor, residual_latent = model.hide(carrier_tensor, secret_tensor)
    
    to_pil = transforms.ToPILImage()
    stego_img = to_pil(stego_carrier_tensor.squeeze(0).cpu())
    stego_img.save(output_path)
    
    torch.save(residual_latent.cpu(), "ml_core/residual_latent.pt")
    
    print(f"\n[+] Success! Data hidden securely inside: {output_path}")
    print(f"[+] Residual latent cache saved to: ml_core/residual_latent.pt")

def extract_image(stego_path, output_secret_path):
    model = load_trained_model()
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    stego_img = Image.open(stego_path).convert("RGB")
    stego_tensor = transform(stego_img).unsqueeze(0).to(DEVICE)
    
    latent_path = "ml_core/residual_latent.pt"
    if not os.path.exists(latent_path):
        raise FileNotFoundError("Residual latent cache not found! Cannot extract without it.")
    
    residual_latent = torch.load(latent_path, map_location=DEVICE, weights_only=False)
    
    with torch.no_grad():
        recovered_carrier, recovered_secret = model.extract_and_restore(stego_tensor, residual_latent)
        
    to_pil = transforms.ToPILImage()
    extracted_img = to_pil(recovered_secret.squeeze(0).cpu())
    extracted_img.save(output_secret_path)
    
    # Compute PSNR quality metric with updated 35 dB threshold
    psnr_value = compute_psnr("inputs/secret.png", output_secret_path)
    
    print(f"\n[+] Success! Secret data cleanly extracted to: {output_secret_path}")
    print(f"    Extracted Secret PSNR Quality: {psnr_value:.2f} dB")
    
    if psnr_value > 35:
        print("    [VERIFICATION PASSED] High fidelity structural recovery confirmed!")
    else:
        print("    [VERIFICATION WARNING] PSNR is lower than expected.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "hide":
            hide_image("inputs/carrier.png", "inputs/secret.png", "outputs/stego_image.png")
        elif mode == "extract":
            extract_image("outputs/stego_image.png", "outputs/recovered_secret.png")
    else:
        print("Usage: python ml_core/stego_tool.py [hide|extract]")