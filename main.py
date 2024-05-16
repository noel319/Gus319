import cv2
import numpy as np
from onnx_inference import InferenceModel


def annotate_image(img_path, label):
    # Load the image
    img = cv2.imread(img_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter to smooth the image
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # Perform edge detection using Canny
    # edges = cv2.Canny(gray, 100, 200)

    # Threshold the grayscale image
    _, thresh = cv2.threshold(
        gray, np.mean(gray), 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Find contours in the thresholded image
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    # Get the largest contour
    cnt = sorted(contours, key=cv2.contourArea)[-1]

    # Create a mask for the contour
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    # Draw the contour filled with red color on the mask
    maskedRed = cv2.drawContours(mask, [cnt], -1, (0, 0, 255), -1)

    # Draw the contour filled with white color on the mask
    maskedFinal = cv2.drawContours(mask, [cnt], -1, (255, 255, 255), -1)

    # Apply the mask to the original image
    finalImage = cv2.bitwise_and(img, img, mask=maskedFinal)

    # Get the bounding box coordinates of the contour
    x, y, w, h = cv2.boundingRect(cnt)

    # Draw the bounding box rectangle on the original image
    img_with_bbox = cv2.rectangle(finalImage, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Add label text above the bounding box
    cv2.putText(
        img_with_bbox,
        label,
        (x + 5, y + 13),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
    )

    # Concatenate the original image and segmented eye image horizontally
    result = cv2.hconcat([img, img_with_bbox])

    # Display the original image, grayscale image, mask with red contour, edges, and final segmented image
    # cv2.imshow("Original Image", img)
    # cv2.imshow('Gray', gray)
    # cv2.imshow('Mask with Red Contour', maskedRed)
    # cv2.imshow('Edges', edges)
    # cv2.imshow('Thresholded Final Image', finalImage)
    # cv2.imshow("Image with Bounding Box", img_with_bbox)

    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return result


if __name__ == "__main__":
    # Model 01
    # onnx_model_path = "onnx_model/efficientnetb3-eye-diseases-01.onnx"
    # class_labels = {0: "cataract", 1: "diabetic_retinopathy", 2: "normal"}
    # img_paths = ["images/cataract.jpg", "images/diabetic_retinopathy.jpeg", "images/normal.jpg"]

    # Model 02
    onnx_model_path = "onnx_model/efficientnetb3-eye-diseases-02.onnx"
    class_labels = {0: "ARMD", 1: "glaucoma", 2: "normal"}
    img_paths = ["images/ARMD.png", "images/glaucoma.jpg", "images/normal.jpg"]

    # Intstantiate the model
    model = InferenceModel(onnx_model_path, class_labels)

    i = 2
    label, conf = model.predict(img_path=img_paths[i])
    result = annotate_image(img_paths[i], label)
        
