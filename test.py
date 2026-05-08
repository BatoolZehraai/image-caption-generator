import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

# --- CONFIGURATION ---
MODEL_PATH = 'models/best_model.keras'
TOKENIZER_PATH = 'models/tokenizer.pkl'
IMAGES_DIR = 'data/Images'

# 1. Load Model and Tokenizer
print("Loading model and tokenizer...")
model = load_model(MODEL_PATH)
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)

# 2. VGG16 Setup for Feature Extraction
vgg_model = VGG16()
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

def extract_features(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = vgg_model.predict(image, verbose=0)
    return feature

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

# 3. Test on a Random Image
max_length = 34 # Jo training ke waqt print hui thi
all_images = os.listdir(IMAGES_DIR)
random_img_name = all_images[np.random.randint(0, len(all_images))]
img_path = os.path.join(IMAGES_DIR, random_img_name)

print(f"Testing on image: {random_img_name}")
feature = extract_features(img_path)
caption = predict_caption(model, feature, tokenizer, max_length)

# Clean caption for display
final_caption = caption.replace('startseq', '').replace('endseq', '').strip()
print(f"Generated Caption: {final_caption}")

# Display the image
img = load_img(img_path)
plt.imshow(img)
plt.title(final_caption)
plt.axis('off')
plt.show()