import tensorflow as tf
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # type: ignore
import numpy as np
import os
from config import img_width, img_height, class_labels

# Define parameters
model_name = "chess_classifier_10k"
model_path = f"models/{model_name}.keras"
tflite_model_path = f"models/{model_name}.tflite"

# Load the saved model from disk
model = load_model(model_path)
print(f"Loaded model from {model_path}")

# https://github.com/keras-team/keras-core/issues/746
tf_callable = tf.function(
    model.call,
    autograph=False,
    input_signature=[tf.TensorSpec((1, img_width, img_height, 3), tf.float32)],
)
tf_concrete_function = tf_callable.get_concrete_function()
converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [tf_concrete_function], tf_callable
)


# Generate representative dataset for better quantization
def representative_dataset_gen():
    sample_images_dir = "dataset/training"
    for piece_type in class_labels:
        piece_dir = os.path.join(sample_images_dir, piece_type)
        filenames = os.listdir(piece_dir)
        selected_filenames = filenames[
            :10
        ]  # Select the first 10 images for each piece type

        for filename in selected_filenames:
            img_path = os.path.join(piece_dir, filename)
            img = load_img(
                img_path, target_size=(img_width, img_height)
            )  # Adjust target_size if necessary
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # Normalize to [0, 1]
            yield [img_array.astype(np.float32)]


if "_quant" in tflite_model_path:
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8  # or tf.int8 ?
    converter.inference_output_type = tf.uint8  # or tf.int8 ?

tflite_model = converter.convert()

with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"Model converted to TensorFlow Lite and saved to {tflite_model_path}")
