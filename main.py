import os
import pickle
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from src.preprocessing import extract_features, clean_text
from src.model_arc import build_model
from src.train import data_generator

# --- CONFIGURATION ---
IMAGES_DIR = 'data/Images'
CAPTIONS_FILE = 'data/captions.txt'
FEATURES_SAVE_PATH = 'models/features.pkl'
TOKENIZER_SAVE_PATH = 'models/tokenizer.pkl'
MODEL_SAVE_PATH = 'models/best_model.keras'

def main():
    # --- 1. IMAGE FEATURES EXTRACTION ---
    if not os.path.exists(FEATURES_SAVE_PATH):
        print("Starting Feature Extraction (This will take time...):")
        features = extract_features(IMAGES_DIR)
        pickle.dump(features, open(FEATURES_SAVE_PATH, 'wb'))
        print(f"Done! Features saved to {FEATURES_SAVE_PATH}")
    else:
        print("Features already extracted, loading from file...")
        with open(FEATURES_SAVE_PATH, 'rb') as f:
            features = pickle.load(f)

    # --- 2. CAPTIONS LOADING & CLEANING ---
    print("Loading and cleaning captions...")
    df = pd.read_csv(CAPTIONS_FILE)

    mapping = {}
    for i, row in df.iterrows():
        img_name = row['image']
        caption = row['caption']
        if img_name not in mapping:
            mapping[img_name] = []
        mapping[img_name].append(caption)

    cleaned_mapping = clean_text(mapping)
    print(f"Total Images: {len(features)}")
    print(f"Total Captions: {len(cleaned_mapping)}")

    # --- 3. TOKENIZATION ---
    print("Tokenizing text data...")
    all_captions = []
    for key in cleaned_mapping:
        for caption in cleaned_mapping[key]:
            all_captions.append(caption)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    vocab_size = len(tokenizer.word_index) + 1
    max_length = max(len(caption.split()) for caption in all_captions)

    print(f"Vocabulary Size: {vocab_size}")
    print(f"Max Caption Length: {max_length}")

    with open(TOKENIZER_SAVE_PATH, 'wb') as f:
        pickle.dump(tokenizer, f)

    # --- 4. MODEL BUILDING OR LOADING ---
    # Check if a saved model exists to resume training
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Found existing model at {MODEL_SAVE_PATH}. Resuming training...")
        model = load_model(MODEL_SAVE_PATH)
    else:
        print("No existing model found. Building a new model from scratch...")
        model = build_model(vocab_size, max_length)
    
    model.summary()

    # --- 5. TRAINING ---
    epochs = 30 # Yahan aap jitne chahein epochs set karein
    batch_size = 16
    steps = len(cleaned_mapping) // batch_size

    # Callback taake har epoch ke baad automatic save ho
    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='loss', verbose=1, save_best_only=False, mode='min')

    print(f"Starting Training for {epochs} epochs...")
    
    # Generator initialization
    generator = data_generator(
        cleaned_mapping, features, tokenizer, max_length, vocab_size, batch_size)

    # Training fit call
    # Is baar hum direct fit use kar rahe hain, manually loop ki zaroorat nahi
    model.fit(generator, epochs=epochs, steps_per_epoch=steps, callbacks=[checkpoint], verbose=1)

    print(f"Training Complete! Final model updated in '{MODEL_SAVE_PATH}'")

if __name__ == "__main__":
    main()