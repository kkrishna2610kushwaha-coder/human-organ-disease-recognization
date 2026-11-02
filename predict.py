
import tensorflow as tf
import numpy as np
import json
import argparse

IMG_SIZE = (224, 224)

# Load trained model
model = tf.keras.models.load_model(
    "organ_disease_model.keras"
)

# Load class names
with open("class_names.json", "r") as file:
    class_names = json.load(file)


def predict_disease(image_path):

    # Load and resize image
    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMG_SIZE,
        color_mode="rgb"
    )

    # Convert image into array
    image_array = tf.keras.utils.img_to_array(image)

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Model prediction
    predictions = model.predict(
        image_array,
        verbose=0
    )

    # Get predicted class
    predicted_index = np.argmax(predictions[0])

    disease_name = class_names[predicted_index]

    confidence = predictions[0][predicted_index] * 100

    print("\n========== PREDICTION RESULT ==========")

    print("Predicted Category:", disease_name)

    print("Confidence Score: {:.2f}%".format(confidence))

    print("=======================================")

    if confidence < 60:
        print("Warning: Low-confidence prediction.")

    print(
        "Educational prediction only; "
        "not a medical diagnosis."
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Predict organ disease from an image"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path of input medical image"
    )

    args = parser.parse_args()

    predict_disease(args.image)
