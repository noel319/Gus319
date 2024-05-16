import os
import warnings
import logging

# Suppress TensorFlow warnings
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import tf2onnx
import onnx
import onnxruntime as rt


# Function to decode predictions
def decode_predictions(predictions, class_labels):
    """
    Decode the model predictions.

    Args:
        predictions (numpy.ndarray): Model predictions.
    """
    score = tf.nn.softmax(predictions[0])
    predicted_class = np.argmax(score)
    confidence = np.max(score)
    label = class_labels[predicted_class]

    return label, confidence


# Function to read and preprocess an image
def read_image(img_path, img_size=(224, 224)):
    """
    Read and preprocess an image.

    Args:
        img_path (str): Path to the image.

    Returns:
        numpy.ndarray: Preprocessed image data.
    """
    img = tf.keras.utils.load_img(img_path, target_size=(img_size[0], img_size[1]))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    return img_array


def export_onnx(model_file_path, input_image):
    # Load the model
    loaded_model = load_model(model_file_path)  # load .h5 model
    preds = loaded_model.predict(input_image)

    # Convert the model to ONNX format
    spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
    model_name = os.path.splitext(os.path.basename(model_file_path))[0]
    output_path = f"onnx_model/{model_name}.onnx"
    model_proto, _ = tf2onnx.convert.from_keras(
        loaded_model, input_signature=spec, opset=13, output_path=output_path
    )
    output_names = [n.name for n in model_proto.graph.output]
    print(f"Exported model to: {output_path}")

    # Run inference with ONNX Runtime
    x = input_image.numpy()
    providers = ["CPUExecutionProvider"]
    m = rt.InferenceSession(output_path, providers=providers)
    onnx_pred = m.run(output_names, {"input": x})[0]

    np.testing.assert_allclose(preds, onnx_pred, rtol=1e-5)
    print(
        "Exported model has been tested with ONNXRuntime, and returned the same results!"
    )

    return onnx_pred


if __name__ == "__main__":
    # Print TensorFlow version
    print("""TensorFlow version: """, tf.__version__)

    # model_path = "results-01/efficientnetb3-eye-diseases-01.h5"
    # class_labels = {0: "cataract", 1: "diabetic_retinopathy", 2: "normal"}
    # img_path = "images/cataract.jpg"

    model_path = "results-02/efficientnetb3-eye-diseases-02.h5"
    class_labels = {0: "ARMD", 1: "glaucoma", 2: "normal"}
    img_path = "images/ARMD.png"

    # Read and preprocess the image
    input_image = read_image(img_path)
    print("Input Image Shape:", input_image.shape)

    # Export ONNX model
    onnx_pred = export_onnx(model_path, input_image)

    # Decode the ONNX predictions
    label, conf = decode_predictions(onnx_pred, class_labels)
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence.".format(
            label, 100 * conf
        )
    )
