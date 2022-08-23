import streamlit as st
from matplotlib import pyplot as plt
from keras.datasets import mnist
from toolz.functoolz import pipe, curry
from toolz.functoolz import compose_left as flow
import numpy as np
import keras
from keras import layers


@curry
def unpack(func, args):
    return func(*args)


def generate_mnist_figure(num_rows, num_columns, images):
    fig, axs = plt.subplots(num_rows, num_columns)
    for i in range(num_rows):
        for j in range(num_columns):
            count = (j + num_columns * i)
            axs[i][j].imshow(images[count], cmap='binary')
            axs[i][j].set_xticks([])
            axs[i][j].set_yticks([])
    return fig


if __name__ == '__main__':
    # Model / data parameters
    num_classes = 10
    input_shape = (28, 28, 1)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    st.set_page_config(page_title="MNIST dataset optimization", layout='wide')
    st.markdown("""# Example MNIST dataset \nBellow you can see an example from MNIST dataset""")
    flow(generate_mnist_figure, st.pyplot)(4, 4, x_train)
    st.write("The training set contains ", x_train.shape[0], " pictures and the testing set contains ", x_test.shape[0],
             " elements with annotations.")
    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    age = st.slider('How old are you?', 1, 32, 16)
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(6, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(12, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.summary(print_fn=lambda x: st.text(x))

    batch_size = 128
    epochs = 15

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    score = model.evaluate(x_test, y_test, verbose=0)
    st.write("Test loss:", score[0])
    st.write("Test accuracy:", score[1])
