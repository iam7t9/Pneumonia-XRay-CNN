from keras.models import load_model
import numpy as np
import tensorflow as tf


# from google.colab import drive
# drive.mount('/content/drive/')
# predictions = (model.predict(x_val))

MODEL_PATH = 'model.h5'
# IMG_PATH = "S:\Projects\Pheno cnn\images\IM-0115-0001.jpeg"
IMG_PATH = "S:\Projects\Pheno cnn\images\person1949_bacteria_4880.jpeg"

model = load_model(MODEL_PATH)
image = tf.keras.preprocessing.image.load_img(IMG_PATH, target_size=(150, 150))

input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])
input_arr = input_arr.astype('float32') / 255.

predictions = model.predict(input_arr,verbose = 0)
print(float(predictions[0]))