import cv2 as cv
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

model = tf.keras.models.load_model('models/my-model')

acc, loss = model.evaluate(x_test, y_test)
print(f'accuracy: {acc}')
print(f'loss: {loss}')

