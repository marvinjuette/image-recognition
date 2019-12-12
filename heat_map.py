from keras.preprocessing import image
from keras.models import load_model
from keras import models

from matplotlib import pyplot as plt
import numpy as np

layer = 2
model_path = "./saves/buildings-2-conv2d-pooling-2-dense.h5"
img_path = "./images/test_set/buildings/marien/1_IMG_20190302_144405.jpg"
output_path = "./generated_images"
output_file_name = "conv2d_2"

img = image.load_img(img_path, target_size=(64, 64))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])

model = load_model(model_path)

layer_outputs = [layer.output for layer in model.layers[:len(model.layers)]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(img_tensor)

plt.title(type(model.layers[layer]))
tmp = activations[2]
plt.imshow(tmp[0, :, :, 4])
plt.savefig(output_path + output_file_name + ".png", dpi=300)
plt.show()
