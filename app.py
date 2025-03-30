from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Load the trained model
model = tf.keras.models.load_model("deepfake_detection_model.h5")

# Define image size
IMG_SIZE = (128, 128)

app = Flask(__name__)

# Preprocessing function
def preprocess_image(image):
    try:
        image = image.convert("RGB")  # Ensure 3 channels (some images might be RGBA or grayscale)
        image = image.resize(IMG_SIZE)  # Resize to match model input
        image = np.array(image) / 255.0  # Normalize pixel values
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        print(f"Processed Image Shape: {image.shape}")  # Debugging
        print(f"Pixel Range: Min={image.min()}, Max={image.max()}")  # Check normalization
        
        return image
    except Exception as e:
        print("Error in preprocessing:", str(e))
        return None

    except Exception as e:
        print("Error in preprocessing:", str(e))
        return None


@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")  # New home page

@app.route("/detection", methods=["GET"])
def detection():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read()))

    # Preprocess Image
    processed_image = preprocess_image(image)
    if processed_image is None:
        return jsonify({"error": "Error processing the image"}), 500

    try:
        prediction = model.predict(processed_image)[0][0]  # Get single prediction
        print("Raw Prediction Score:", prediction)  # Debugging

        label = "Real" if prediction > 0.5 else "Fake"
        confidence = float(prediction if prediction > 0.5 else 1 - prediction)

        print(f"Prediction: {label} | Confidence: {confidence}")

        return jsonify({"prediction": label, "confidence": confidence})

    except Exception as e:
        print("Error:", str(e))  # Debugging
        return jsonify({"error": "Error analyzing the image. Try again."}), 500

if __name__ == "__main__":
    app.run(debug=True)