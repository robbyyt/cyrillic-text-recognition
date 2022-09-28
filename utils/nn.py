import tensorflow as tf
import getpass
from datetime import date


training_params = {
    "batch_size": 32,
    "img_size": (180, 180),
}

AUTOTUNE = tf.data.AUTOTUNE

def get_checkpoint_path():
    today = date.today()
    return f'training/${getpass.getuser()}-{today.strftime("%x")}/checkpoint.ckpt'


def create_model(num_outputs):
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1. / 255),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_outputs)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    return model


def train(training_dataset, validation_dataset):
    cached_training_dataset = training_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    cached_validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    num_output_classes = len(training_dataset.class_names)
    model = create_model(num_output_classes)
    checkpoint_path = get_checkpoint_path()
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    model.fit(
        cached_training_dataset,
        epochs=1,
        verbose=2,
        validation_data=cached_validation_dataset,
        callbacks=[cp_callback]
    )
