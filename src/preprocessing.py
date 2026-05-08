import os
import string
import pickle
from tqdm import tqdm
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

# --- PART 1: IMAGE FEATURE EXTRACTION ---
def extract_features(directory):
    model = VGG16()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    
    features = {}
    print("Extracting features from images...")
    for img_name in tqdm(os.listdir(directory)):
        img_path = os.path.join(directory, img_name)
        # Check if it's a file
        if os.path.isfile(img_path):
            image = load_img(img_path, target_size=(224, 224))
            image = img_to_array(image)
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            image = preprocess_input(image)
            feature = model.predict(image, verbose=0)
            features[img_name] = feature
    return features

# --- PART 2: TEXT CLEANING ---
def clean_text(captions_dict):
    table = str.maketrans('', '', string.punctuation)
    for key, captions in captions_dict.items():
        for i in range(len(captions)):
            caption = captions[i].lower()
            caption = caption.translate(table)
            words = [w for w in caption.split() if len(w) > 1]
            # NLP models ko shuru aur khatam batana zaroori hai
            captions[i] = 'startseq ' + ' '.join(words) + ' endseq'
    return captions_dict