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
import time

# methods

def get_sets(input_image_size):
    train_datagen = ImageDataGenerator(validation_split=0.2,
                                   rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
    
    test_datagen = ImageDataGenerator(rescale = 1./255)

    seed_training_images = 0

    training_set = train_datagen.flow_from_directory('training_set',
                                                 subset='training',
                                                 target_size = (input_image_size, input_image_size),
                                                 batch_size = 32,
                                                 class_mode = 'categorical',
                                                 seed=seed_training_images) 

    validation_set = train_datagen.flow_from_directory('training_set',
                                                 subset='validation',
                                                 target_size = (input_image_size, input_image_size),
                                                 batch_size = 32,
                                                 class_mode = 'categorical',
                                                 seed=seed_training_images)

    test_set = test_datagen.flow_from_directory('test_set',
                                            target_size = (input_image_size, input_image_size),
                                            batch_size = 32,
                                            class_mode = 'categorical')
    return [test_set, training_set, validation_set]

def show_confusion_matrix(used_model, chosen_size):
    test_set = get_sets(chosen_size)[0]
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
    test_set = get_sets(chosen_size)[0]
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

def train_model(epoch_count, size):
    training_set = get_sets(size)[1]
    validation_set = get_sets(size)[2]
    number_of_classes = 5

    number_of_filters = 64
    shape_filter_param = 3
    pooling_size_param = 2

    model = tf.keras.Sequential([
    layers.Conv2D(number_of_filters, (shape_filter_param, shape_filter_param), input_shape = (size, size, 3), activation="relu"),
    layers.MaxPooling2D((pooling_size_param, pooling_size_param)),
    layers.Dropout(0.1),
    layers.Conv2D(number_of_filters/2, (shape_filter_param, shape_filter_param), activation="relu"),
    layers.MaxPooling2D((pooling_size_param, pooling_size_param)),
    layers.Dropout(0.2),
    layers.Conv2D(number_of_filters/4, (shape_filter_param, shape_filter_param), activation="relu"),
    layers.MaxPooling2D((pooling_size_param, pooling_size_param)),
    layers.Dropout(0.1),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(number_of_classes, activation="softmax") # because it's multiple classes, we use softmax
    ])

    model.compile(optimizer = optimizers.Adam(learning_rate=0.001), 
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])
    
    model.fit(training_set, validation_data= validation_set, steps_per_epoch=10, epochs=epoch_count)
    return model

# Variables

size = 64

# App

st.title("Zelf trainen")
st.write("Hier kun je zelf wat dingen aanpassen.")

slider_value = st.slider(label="Aantal epochs", min_value=1, max_value=25)

train_value = st.button("Train")

if train_value == True:
    with st.spinner("Trainen"):
        model = train_model(slider_value, size)
        st.subheader("Confusion matrix met " + str(round(show_accuracy(model, size) * 100, 0)) + " % accuraatheid")
        st.subheader("Met " + str(slider_value) + " epochs")

        cm_display = show_confusion_matrix(model, size)

        fig, ax = plt.subplots()
        cm_display.plot(ax=ax, colorbar=False, cmap="Blues")

        st.pyplot(fig)
