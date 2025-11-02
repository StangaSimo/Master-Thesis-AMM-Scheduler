import time
import numpy as np
from openvino.runtime import Core
from PIL import Image

# --- Configura qui il modello e l'immagine ---
MODEL_XML = r'C:\Users\simos\Documents\j\resnet-50-tf\FP32\resnet-50-tf.xml'
MODEL_BIN = r'C:\Users\simos\Documents\j\resnet-50-tf\FP32\resnet-50-tf.bin'       # percorso del modello OpenVINO
IMAGE_PATH = r'C:\Users\simos\Documents\j\prova.png' # immagine di test
# --- Carica l'immagine e fai preprocessing ---
def preprocess_image(image_path, input_shape):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((input_shape[2], input_shape[1]))  # Width, Height (224x224)
    img = np.array(img)  # Formato HWC (Height, Width, Channels)
    img = img[np.newaxis, :, :, :]  # Aggiungi dimensione batch -> [1, 224, 224, 3]
    img = img.astype(np.float32)
    return img

# --- Benchmark function ---
def benchmark(device="CPU", n_runs=100):
    ie = Core()
    model = ie.read_model(model=MODEL_XML, weights=MODEL_BIN)
    compiled_model = ie.compile_model(model=model, device_name=device)
    
    input_layer = compiled_model.input(0)
    input_shape = input_layer.shape

    # Preprocess immagine
    input_data = preprocess_image(IMAGE_PATH, input_shape)

    # Warm-up
    for _ in range(5):
        compiled_model([input_data])

    # Misura tempo inferenza
    start = time.time()
    for _ in range(n_runs):
        result = compiled_model([input_data])
    end = time.time()

    avg_time = (end - start) / n_runs
    fps = 1 / avg_time
    print(f"\nDevice: {device} | Avg inference time: {avg_time*1000:.2f} ms | FPS: {fps:.2f}")

def main():
    print("OpenVINO Benchmark Tool")
    print("-----------------------")
    
    while True:
        print("\nScegli un'opzione:")
        print("1. Esegui benchmark su CPU")
        print("2. Esegui benchmark su GPU")
        print("3. NPU")
        print("4. Esegui benchmark su TUTTI")
        print("5. Esci")
        
        choice = input("Scelta (1-4): ")
        
        if choice == "1":
            benchmark(device="CPU")
        elif choice == "2":
            benchmark(device="GPU")
        elif choice == "4":
            benchmark(device="CPU")
            benchmark(device="GPU")
            benchmark(device="NPU")
        elif choice == "3":
            benchmark(device="NPU")
            break
        elif choice == "5":
            print("Uscita...")
            break
        else:
            print("Scelta non valida. Riprova.")

if __name__ == "__main__":
    main()
