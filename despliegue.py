import streamlit as st
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

import os
import math
import numpy as np
import matplotlib.pyplot as plt



st.set_page_config(layout="centered", page_title="ACGAN MNIST Generator")
st.title("Titulo")
st.write('Seleccione un dpigito para generar imágenes')

@st.cache_resource
def load_acgan_model():
    model_path = "acgan_mnist.h5"
    if not os.path.exists(model_path):
        st.error(f"Error: El archivo '{model_path}' no se encontró. Reivisar que esté en la misma carpeta que el main")
        st.stop()
    return load_model(model_path)

trained_generator = load_acgan_model()
latent_size = 100
num_classes= 10

st.sidebar.header('Opciones de generación')
selected_label= st.sidebar.slider(
    "Selecciona el dígito a generar",
    min_value=0,
    max_value=9,
    value= 0,
    step=1
)

def generate_and_plot_acgan(generator, latent_size, num_classes, class_label):
    noise_input= np.random.uniform(-1.0, 1.0, size= [16, latent_size])
    noise_class_label = np.ones(16, dtype= 'int32')*class_label

    noise_class_input= noise_class_label.reshape(-1,1)

    with st.spinner ('Generando imágenes...'):
        images = generator.predict([noise_input,noise_class_input], verbose=0)
    
    fig, ax = plt.subplots(figsize=(8,8))
    num_images= images.shape[0]
    image_size = images.shape[1]
    rows = int(math.sqrt(num_images))
    cols = int(math.ceil(num_images)/rows) 

    for i in range (num_images):
        plt.subplot(rows, cols, i + 1)
        image = np.reshape(images[i], [image_size, image_size])
        plt.imshow(image, cmap = 'gray')
        plt.title(f'Label: {noise_class_label[i]}')
        plt.axis('off')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

if st.sidebar.button('Generar imágenes') or st.session_state.get('initial_run',True):
    generate_and_plot_acgan(trained_generator, latent_size, num_classes, selected_label)
    st.session_state['initial_run'] = False
    
st.markdown('---')
st.write('Vamo a dibujar')
