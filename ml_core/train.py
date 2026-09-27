import torch
import torch.optim as optim
import torch.nn.functional as F
from engine import LosslessStegoINN

def train_model():
    # Detect device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Training Lossless Stego Model on device: {device}")

    # Initialize model and optimizer
    model = LosslessStegoINN(channels=6, num_blocks=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-4)

    print("-> Starting training simulation loop...")
    model.train()

    # Simulating a training loop for 200 iterations
    for epoch in range(1, 201):
        optimizer.zero_grad()

        # Simulated batch: Batch size 4, 3 RGB channels, 64x64 resolution tensors
        carrier_images = torch.randn(4, 3, 64, 64).to(device)
        secret_payloads = torch.randn(4, 3, 64, 64).to(device)

        # 1. Forward Pass (Hide secret in carrier)
        stego_carrier, residual_latent = model.hide(carrier_images, secret_payloads)

        # 2. Inverse Pass (Extract secret and restore carrier)
        restored_carrier, recovered_secret = model.extract_and_restore(stego_carrier, residual_latent)

        # 3. Multi-objective Loss Function
        # - Visual invisibility constraint
        loss_visual = F.mse_loss(stego_carrier, carrier_images)
        # - Zero-error secret extraction constraint
        loss_secret = F.mse_loss(recovered_secret, secret_payloads)
        # - Lossless carrier restoration constraint
        loss_restore = F.mse_loss(restored_carrier, carrier_images)

        # Total composite loss with heavy weighting on reconstruction accuracy
        total_loss = loss_visual + (50.0 * loss_secret) + (50.0 * loss_restore)

        # Backpropagation & Optimization
        total_loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == 1:
            print(f"   Epoch [{epoch}/200] | Total Loss: {total_loss.item():.6f} | Secret Extraction Error: {loss_secret.item():.12f}")

    # Save the trained weights once done
    torch.save(model.state_dict(), "ml_core/best_stego_model.pth")
    print("\n-> Training complete! Model weights saved successfully to ml_core/best_stego_model.pth")

if __name__ == "__main__":
    train_model()