from PIL import Image
import numpy as np
import tensorflow as tf
from utils import clean_image, get_prediction, make_results
from flask import Flask, jsonify, request


app = Flask(__name__)


def load_model(path):
    # Xception Model
    xception_model = tf.keras.models.Sequential(
        [
            tf.keras.applications.xception.Xception(
                include_top=False, weights="imagenet", input_shape=(512, 512, 3)
            ),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(4, activation="softmax"),
        ]
    )

    # DenseNet Model
    densenet_model = tf.keras.models.Sequential(
        [
            tf.keras.applications.densenet.DenseNet121(
                include_top=False, weights="imagenet", input_shape=(512, 512, 3)
            ),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(4, activation="softmax"),
        ]
    )

    # Ensembling the Models
    inputs = tf.keras.Input(shape=(512, 512, 3))

    xception_output = xception_model(inputs)
    densenet_output = densenet_model(inputs)

    outputs = tf.keras.layers.average([densenet_output, xception_output])

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Loading the Weights of Model
    model.load_weights(path)

    return model


#Loading the Model
model = load_model("model.h5")


@app.route("/")
def home():
    return "App is running"


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file found"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Save the file to disk or perform any other desired processing
    prediction = predict(file)

    # file.save("uploads/" + file.filename)
    return jsonify(prediction),200


def predict(imageFile):
    # Reading the uploaded image
    image = Image.open(imageFile)
    np.array(Image.fromarray(np.array(image)).resize((700, 400), Image.ANTIALIAS))

    # Cleaning the image
    image = clean_image(image)

    # Making the predictions
    predictions, predictions_arr = get_prediction(model, image)

    # Making the results
    result = make_results(predictions, predictions_arr)

    return {
        "status": result["status"],
        "prediction": result["prediction"],
    }


if __name__ == "__main__":
    app.run()
