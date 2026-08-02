import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import os

# Import your model engine for images
from ml_core.engine import LosslessStegoINN

st.set_page_config(page_title="Neural & Binary Steganography Suite", page_icon="🔒", layout="centered")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = LosslessStegoINN(channels=6, num_blocks=4).to(DEVICE)
    weights_path = "ml_core/best_stego_model.pth"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=False))
        model.eval()
    return model

model = load_model()

st.title("🔒 Steganography Control Panel")
st.write("Choose between Neural Image-in-Image Hiding or Binary File LSB Concealment.")

tab1, tab2, tab3 = st.tabs(["Image-in-Image (Neural)", "Extract Neural Image", "File-in-Image (PDF/Docs)"])

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# --- TAB 1: Neural Image Hide ---
with tab1:
    st.header("Encode Secret Image (Neural INN)")
    carrier_file = st.file_uploader("Upload Carrier Image", type=["png", "jpg", "jpeg"], key="carrier_img")
    secret_file = st.file_uploader("Upload Secret Image", type=["png", "jpg", "jpeg"], key="secret_img")
    
    if carrier_file and secret_file:
        col1, col2 = st.columns(2)
        with col1:
            c_img = Image.open(carrier_file).convert("RGB")
            st.image(c_img, caption="Carrier", use_column_width=True)
        with col2:
            s_img = Image.open(secret_file).convert("RGB")
            st.image(s_img, caption="Secret", use_column_width=True)
            
        if st.button("Hide Image Securely", type="primary"):
            with st.spinner("Processing through neural network..."):
                c_tensor = transform(c_img).unsqueeze(0).to(DEVICE)
                s_tensor = transform(s_img).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    stego_tensor, residual = model.hide(c_tensor, s_tensor)
                
                stego_img = transforms.ToPILImage()(stego_tensor.squeeze(0).cpu())
                
                os.makedirs("outputs", exist_ok=True)
                os.makedirs("ml_core", exist_ok=True)
                
                stego_path = "outputs/stego_image.png"
                stego_img.save(stego_path)
                torch.save(residual.cpu(), "ml_core/residual_latent.pt")
                
            st.success("Successfully hidden!")
            st.image(stego_img, caption="Generated Stego Carrier Image", use_column_width=True)
            with open(stego_path, "rb") as file:
                st.download_button("Download Stego Image", file, file_name="stego_image.png", mime="image/png")

# --- TAB 2: Neural Image Extract ---
with tab2:
    st.header("Extract Secret Image (Neural INN)")
    stego_upload = st.file_uploader("Upload Stego Image", type=["png", "jpg", "jpeg"], key="stego_up_neural")
    
    if stego_upload:
        stg_img = Image.open(stego_upload).convert("RGB")
        st.image(stg_img, caption="Stego Carrier", use_column_width=True)
        
        if st.button("Extract Secret Payload", type="primary"):
            latent_path = "ml_core/residual_latent.pt"
            if not os.path.exists(latent_path):
                st.error("Residual latent cache missing! Run the hide step first.")
            else:
                with st.spinner("Extracting via neural inversion..."):
                    stg_tensor = transform(stg_img).unsqueeze(0).to(DEVICE)
                    residual = torch.load(latent_path, map_location=DEVICE, weights_only=False)
                    
                    with torch.no_grad():
                        _, recovered_secret = model.extract_and_restore(stg_tensor, residual)
                        
                    ext_img = transforms.ToPILImage()(recovered_secret.squeeze(0).cpu())
                    ext_path = "outputs/recovered_secret.png"
                    os.makedirs("outputs", exist_ok=True)
                    ext_img.save(ext_path)
                    
                st.success("Successfully extracted!")
                st.image(ext_img, caption="Recovered Secret Image", use_column_width=True)
                with open(ext_path, "rb") as file:
                    st.download_button("Download Recovered Secret", file, file_name="recovered_secret.png", mime="image/png")

# --- TAB 3: File-in-Image (PDF/Docs/Zips) ---
with tab3:
    st.header("Arbitrary File Steganography (LSB)")
    st.write("Hide any file (PDF, TXT, ZIP, etc.) inside a carrier image seamlessly.")
    
    file_subtab1, file_subtab2 = st.tabs(["Hide File", "Extract File"])
    
    with file_subtab1:
        f_carrier = st.file_uploader("Carrier Image", type=["png", "jpg", "jpeg"], key="f_carrier")
        payload_file = st.file_uploader("Secret File (PDF, etc.)", key="payload_file")
        
        if f_carrier and payload_file:
            c_img = Image.open(f_carrier).convert("RGB")
            st.image(c_img, caption="Carrier Image Preview", use_column_width=True)
            
            if st.button("Hide File inside Image", type="primary"):
                with st.spinner("Encoding binary payload into pixels..."):
                    secret_bytes = payload_file.read()
                    ext = payload_file.name.split(".")[-1].encode('utf-8')
                    
                    if len(ext) > 255:
                        st.error("File extension too long.")
                    else:
                        header = len(secret_bytes).to_bytes(4, 'big') + len(ext).to_bytes(1, 'big') + ext
                        payload = header + secret_bytes
                        binary_data = ''.join(format(byte, '08b') for byte in payload)
                        data_len = len(binary_data)
                        
                        pixels = list(c_img.getdata())
                        max_bits = len(pixels) * 3
                        
                        if data_len > max_bits:
                            st.error(f"File is too large! Needs {data_len} bits, but image capacity is {max_bits} bits.")
                        else:
                            data_index = 0
                            new_pixels = []
                            for pixel in pixels:
                                r, g, b = pixel
                                if data_index < data_len:
                                    r = (r & ~1) | int(binary_data[data_index])
                                    data_index += 1
                                if data_index < data_len:
                                    g = (g & ~1) | int(binary_data[data_index])
                                    data_index += 1
                                if data_index < data_len:
                                    b = (b & ~1) | int(binary_data[data_index])
                                    data_index += 1
                                new_pixels.append((r, g, b))
                                
                            encoded_img = Image.new(c_img.mode, c_img.size)
                            encoded_img.putdata(new_pixels)
                            
                            os.makedirs("outputs", exist_ok=True)
                            out_file_path = "outputs/stego_with_file.png"
                            encoded_img.save(out_file_path, "PNG")
                            
                            st.success("File successfully embedded into image!")
                            with open(out_file_path, "rb") as f:
                                st.download_button("Download Stego Image with File", f, file_name="stego_with_file.png", mime="image/png")

    with file_subtab2:
        stego_file_up = st.file_uploader("Upload Stego Image containing File", type=["png", "jpg", "jpeg"], key="stego_file_up")
        
        if stego_file_up:
            stg_img = Image.open(stego_file_up).convert("RGB")
            st.image(stg_img, caption="Stego Image", use_column_width=True)
            
            if st.button("Extract Hidden File", type="primary"):
                with st.spinner("Decoding binary streams from pixels..."):
                    pixels = list(stg_img.getdata())
                    bits = []
                    for pixel in pixels:
                        for val in pixel:
                            bits.append(str(val & 1))
                            
                    binary_string = "".join(bits)
                    all_bytes = bytearray(int(binary_string[i:i+8], 2) for i in range(0, len(binary_string), 8))
                    
                    file_size = int.from_bytes(all_bytes[0:4], 'big')
                    ext_len = all_bytes[4]
                    ext = all_bytes[5:5+ext_len].decode('utf-8')
                    
                    start_idx = 5 + ext_len
                    file_data = all_bytes[start_idx : start_idx + file_size]
                    
                    extracted_filename = f"recovered_file.{ext}"
                    extracted_path = os.path.join("outputs", extracted_filename)
                    os.makedirs("outputs", exist_ok=True)
                    
                    with open(extracted_path, "wb") as f:
                        f.write(file_data)
                        
                    st.success(f"File extracted successfully (. {ext})!")
                    with open(extracted_path, "rb") as f:
                        st.download_button(f"Download Recovered {ext.upper()} File", f, file_name=extracted_filename)