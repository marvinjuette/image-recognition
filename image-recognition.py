# import tensorflow
# import help libs
from time import time

import keras
import tensorflow as tf
from keras.callbacks import TensorBoard

NAME = "cnn-georgs-vs-mg-2-conv2d-pooling-2-dense-{}".format(time())

# initialize tensorboard
tensorboard = TensorBoard(log_dir='./log/{}'.format(NAME))

# initialize image_generator object
test_image_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
train_image_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,
                                                                     shear_range=0.2,
                                                                     zoom_range=0.2,
                                                                     horizontal_flip=True)
train_image_data = train_image_generator.flow_from_directory("images/train_set/buildings",
                                                             target_size=(64, 64),
                                                             batch_size=32,
                                                             class_mode='binary')

test_image_data = test_image_generator.flow_from_directory("images/test_set/buildings",
                                                           target_size=(64, 64),
                                                           batch_size=32,
                                                           class_mode='binary')

# create model
model = keras.Sequential([
    keras.layers.Conv2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    # keras.layers.Conv2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'),
    # keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    # keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# writer = tf.summary.FileWriter('./log')

with tf.Session() as sess:
    model.fit_generator(train_image_data, validation_data=test_image_data, validation_steps=800,
                        steps_per_epoch=8000, epochs=10, callbacks=[tensorboard])

    model.save('./saves/buildings.h5')

# writer.flush()
