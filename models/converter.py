import os
from tensorflow import keras
import onnxmltools
os.environ['TF_KERAS'] = '1'


model = keras.models.load_model(r'Unet_autoencoder.h5')
onnx_model = onnxmltools.convert_keras(model)
onnxmltools.utils.save_model(onnx_model, 'Unet_autoencoder.onnx')

print("done")