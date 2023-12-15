import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras import layers

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import image_dataset_from_directory
from keras.models import load_model

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score

import streamlit as st

# methods

def get_test_Set(input_image_size):
    test_datagen = ImageDataGenerator(rescale = 1./255)
    test_set = test_datagen.flow_from_directory('test_set',
                                            target_size = (input_image_size, input_image_size),
                                            batch_size = 32,
                                            class_mode = 'categorical')
    return test_set

def show_confusion_matrix(used_model, chosen_size):
    test_set = get_test_Set(chosen_size)
    labels = [x for x, y in test_set.class_indices.items()]

    true_labels = []
    predicted_labels = []
    steps = len(test_set)

    for i in range(steps):
        x_batch, y_batch = test_set[i]
        true_labels.extend(np.argmax(y_batch, axis=1))
        predicted_labels.extend(np.argmax(used_model.predict(x_batch), axis=1))

    cm = confusion_matrix(true_labels, predicted_labels)

    # return cm
    plot = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    plot.plot(cmap='Blues')
    return plot

def show_accuracy(used_model, chosen_size):
    test_set = get_test_Set(chosen_size)
    labels = [x for x, y in test_set.class_indices.items()]

    true_labels = []
    predicted_labels = []
    steps = len(test_set)

    for i in range(steps):
        x_batch, y_batch = test_set[i]
        true_labels.extend(np.argmax(y_batch, axis=1))
        predicted_labels.extend(np.argmax(used_model.predict(x_batch), axis=1))

    ac_sc = accuracy_score(true_labels, predicted_labels)

    return ac_sc

# app

st.title("Taak 3 - Deep Learning - Image Classification")
st.subheader("van Lander Jacobs")
st.write("Hier is het mogelijk om een aantal modellen te zien die ik heb getraind, dit zijn de meest interessante modellen die ik er uit heb gepikt.")

option_box = ["model_dl_task3", "model_dl_task3_50_small", "model_dl_task3_500_epochs", "model_dl_task3_1000_epochs"]
text_option = ["Dit model is getraind op afbeeldingen resized naar 256x256, voor 20 epochs.",
               "Dit model is getraind op afbeeldingen resized naar 64x64, voor 50 epochs.\nHet model begint hierna zeer hard te overfitten, 50 is zeker de limiet.",
               "Dit model is getraind op afbeeldingen resized naar 64x64, voor 500 epochs.\nDit model is overfit maar doet het desondanks nog wel goed op de test_set.",
               "Dit model is getraind op afbeeldingen resized naar 64x64, voor 1000 epochs.\nDit model is zeer overfit maar doet het desondanks nog wel goed op de test_set."]
size_option = [256, 64, 64, 64]
size = 64

selected_option = st.selectbox("Select an option", option_box)

for x in range(len(option_box)):
    if option_box[x] == selected_option:
        st.write(text_option[x])
        size = size_option[x]

model = load_model("saved_models/" + selected_option + ".tf")

score = show_accuracy(model, size)

st.subheader("Confusion matrix met " + str(round(score * 100, 0)) + " % accuraatheid")

cm_display = show_confusion_matrix(model, size)

fig, ax = plt.subplots()
cm_display.plot(ax=ax, colorbar=False, cmap="Blues")

st.pyplot(fig)