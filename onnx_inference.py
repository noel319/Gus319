import onnxruntime as rt
from convert_onnx import read_image, decode_predictions


class InferenceModel:
    def __init__(self, onnx_model_path, class_dict):
        """
        Initialize the InferenceModel object.

        Args:
            model_path (str): Path to the ONNX model.
            class_dict (dict): Dictionary of class labels.
        """
        self.session = rt.InferenceSession(onnx_model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.class_dict = class_dict

    def prepare_input(self, image_path):
        """
        Prepare the input for the model.

        Args:
            image_path (str): Path to the image.

        Returns:
            numpy.ndarray: Preprocessed image data.
        """
        image = read_image(image_path)
        return image.numpy()

    def perform_inference(self, input_data):
        """
        Perform inference using the model.

        Args:
            input_data (numpy.ndarray): Preprocessed image data.

        Returns:
            list: Outputs of the model.
        """
        return self.session.run(None, {self.input_name: input_data})

    def process_output(self, outputs):
        """
        Process the output of the model.

        Args:
            outputs (list): Outputs of the model.

        Returns:
            numpy.ndarray: Decoded predictions.
        """
        output_name = self.session.get_outputs()[0].name
        output_data = outputs[0]
        return decode_predictions(output_data, self.class_dict)

    def predict(self, img_path):
        """
        Perform the prediction using the input image path.

        Args:
            img_path (str): The path to the image for prediction.

        Returns:
            list: Predicted output from the model.
        """
        input_data = self.prepare_input(img_path)
        outputs = self.perform_inference(input_data)
        predictions = self.process_output(outputs)
        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence.".format(
                predictions[0], 100 * predictions[1]
            )
        )
        return predictions


if __name__ == "__main__":
    # Model 01
    # onnx_model_path = "onnx_model/efficientnetb3-eye-diseases-01.onnx"
    # class_labels = {0: "cataract", 1: "diabetic_retinopathy", 2: "normal"}
    # img_path = "images/cataract.jpg"

    # Model 02
    onnx_model_path = "onnx_model/efficientnetb3-eye-diseases-02.onnx"
    class_labels = {0: "ARMD", 1: "glaucoma", 2: "normal"}
    img_path = "images/ARMD.png"

    model = InferenceModel(onnx_model_path, class_labels)
    result = model.predict(img_path=img_path)
    print(result)
