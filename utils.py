from os import path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

@st.cache_resource
def get_trained_model(model_dir=None):
    model_path = ''
    if model_dir:
        model_path = path.join(model_dir, 'cataract_classifier_model.keras')
    else:
        model_path = path.join('models', 'cataract_classifier_model.keras')

    return load_model(model_path)

def predict(model, img_path, actual_class='', show_image=True):
    image = load_img(img_path, target_size=(55,94))
    image = img_to_array(image)
    img = np.expand_dims(image, axis=0)
    pred = model.predict(img, verbose=0)
    predicted_class = 'normal' if pred[0] > 0.5 else 'cataract'
    
    results = ''
    if actual_class:
        results = f'( Actual: {actual_class}, Predicted: {predicted_class} )'
    else:
        results = f'( Predicted: {predicted_class} )'
    
    if show_image:
        img = plt.imread(img_path)
        plt.title(results)
        plt.imshow(img)
        plt.axis('off')
    else:
        print(results)
    
    return results

