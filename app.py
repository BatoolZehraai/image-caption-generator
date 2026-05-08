import streamlit as st
import numpy as np
import pickle
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Image Captioner", layout="centered")
st.title("📸 AI Image Caption Generator")
st.markdown("---")
st.write("Upload an image and let the AI describe it.")

# --- LOAD MODELS ---
@st.cache_resource
def load_ai_models():
    # Model aur Tokenizer load karein
    model = load_model('models/best_model.keras')
    with open('models/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    
    # VGG16 for Feature Extraction
    vgg = VGG16()
    vgg = Model(inputs=vgg.inputs, outputs=vgg.layers[-2].output)
    
    return model, tokenizer, vgg

# Models initialize karein
model, tokenizer, vgg_model = load_ai_models()

# --- BEAM SEARCH FUNCTION ---
def beam_search_predictions(model, image_feature, tokenizer, max_length=34, beam_index=3):
    start = [tokenizer.word_index['startseq']]
    sequences = [[start, 0.0]]
    
    while len(sequences[0][0]) < max_length:
        all_candidates = list()
        for seq, score in sequences:
            if seq[-1] == tokenizer.word_index.get('endseq'):
                all_candidates.append([seq, score])
                continue
                
            padded_seq = pad_sequences([seq], maxlen=max_length)
            preds = model.predict([image_feature, padded_seq], verbose=0)
            
            # Top k candidates
            best_preds = np.argsort(preds[0])[-beam_index:]
            
            for word_idx in best_preds:
                candidate_seq = seq + [word_idx]
                candidate_score = score + np.log(preds[0][word_idx] + 1e-10)
                all_candidates.append([candidate_seq, candidate_score])
        
        # Sort and select top sequences
        ordered = sorted(all_candidates, key=lambda l: l[1], reverse=True)
        sequences = ordered[:beam_index]
        
        if all(s[0][-1] == tokenizer.word_index.get('endseq') for s in sequences):
            break

    final_seq = sequences[0][0]
    final_caption = []
    for idx in final_seq:
        word = None
        for w, index in tokenizer.word_index.items():
            if index == idx:
                word = w
                break
        if word == 'endseq':
            break
        if word != 'startseq':
            final_caption.append(word)
            
    return ' '.join(final_caption)

# --- UI LOGIC ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Image load aur display karein
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Generate button
    if st.button('Generate Caption'):
        with st.spinner('AI is thinking...'):
            # 1. Preprocessing for VGG16
            img = image.convert('RGB') # Ensure 3 channels
            img = img.resize((224, 224))
            img = img_to_array(img)
            img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
            img = preprocess_input(img)
            
            # 2. Feature Extraction
            feature = vgg_model.predict(img, verbose=0)
            
            # 3. Prediction using Beam Search
            description = beam_search_predictions(model, feature, tokenizer, beam_index=3)
            
            # Result Display
            st.success(f"**Generated Caption:** {description}")
            st.info("Note: This model is trained on the Flickr8k dataset and may struggle with abstract logos or text.")

else:
    st.info("Please upload an image to start.")
