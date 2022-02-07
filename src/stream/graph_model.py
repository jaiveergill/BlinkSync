import tensorflow.keras as keras
import tensorflow as tf
LOG_DIR = 'src/stream/logs/fit'

model = keras.models.load_model('src/stream/saved_model2')
tb_callback = tf.keras.callbacks.TensorBoard(LOG_DIR)
tb_callback.set_model(model)