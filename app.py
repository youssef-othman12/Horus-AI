from flask import Flask, request, render_template, jsonify
import os

# Import class_names from the class_labels.py file
from class_labels import class_names

# Import LLM utility function
from llm_utils import generate_chat_response # Added import for llm_utils

# Import necessary libraries for model loading and image processing
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__) 

# Load the Keras model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "last_model_bgd.keras")
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    model = None
    print(f"Warning: Model file not found at {MODEL_PATH}. Classification will be mocked.")

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def classify_image(image_bytes):
    if model is None:
        print("Model not loaded, using mock classification.")
        return class_names[0] if class_names else "Mocked Artifact", "This is a mocked English description as the model is not available."
    
    if not image_bytes:
        return "Error: No image data", ""

    try:
        preprocessed_image = preprocess_image(image_bytes)
        predictions = model.predict(preprocessed_image)
        class_idx = int(np.argmax(predictions[0]))
        
        if 0 <= class_idx < len(class_names):
            predicted_class_name = class_names[class_idx]
        else:
            predicted_class_name = "Unknown Artifact"
            print(f"Warning: Predicted class index {class_idx} is out of bounds for class_names.")

        description = f"This is a magnificent {predicted_class_name}, a true masterpiece of ancient Egyptian art, reflecting the rich history and culture of the civilization."
        return predicted_class_name, description
    except Exception as e:
        print(f"Error during image classification: {e}")
        return "Error during classification", str(e)

def get_recommendations_mock(location, interests, liked_places, duration):
    loc_display = location.capitalize() if location else "popular areas in Egypt"
    int_display = interests if interests else "general sightseeing and history"
    liked_display = liked_places if liked_places else "iconic landmarks"
    dur_display = duration if duration else "a few"

    recommendations = [
        f"Day 1: Begin your exploration of {loc_display}. With an interest in {int_display}, consider visiting a renowned museum or a significant historical site.",
        f"Day 2: Discover more of {loc_display}. If you enjoyed places like {liked_display}, you might find similar experiences captivating today.",
        f"Day {dur_display if isinstance(dur_display, (int, str)) and str(dur_display).isdigit() else 'X'}: Conclude your {dur_display}-day journey with a memorable activity. Perhaps a cultural show or a scenic tour around the area."
    ]
    if not location and not interests and not liked_places:
        recommendations = [
            "Day 1: Start your Egyptian adventure by visiting the world-famous Giza Pyramids and the Sphinx. Prepare to be amazed!",
            "Day 2: Explore the Egyptian Museum in Cairo to see an incredible collection of ancient artifacts, including treasures from Tutankhamun's tomb.",
            f"Day {dur_display if isinstance(dur_display, (int, str)) and str(dur_display).isdigit() else 'X'}: Consider a relaxing Nile River cruise or explore the historic Khan el-Khalili bazaar for unique souvenirs."
        ]
    return "\n".join(recommendations)

@app.route("/")
def index():
    return render_template("horos1.html")

@app.route("/about_us")
def about_us():
    return render_template("about_us.html")

@app.route("/page2_image_result")
def result_page():
    return render_template("page2_image_result.html")

@app.route("/page3_recommendation_result")
def recommendation_display_page():
    return render_template("page3_recommendation_result.html")

@app.route("/upload_image", methods=["POST"])
def upload_image_route():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No image selected for uploading"}), 400

    try:
        image_bytes = file.read()
        class_name, description = classify_image(image_bytes)
        
        if "Error" in class_name:
             return jsonify({"error": description or class_name}), 500

        return jsonify({"class_name": class_name, "description": description})
    except Exception as e:
        print(f"Error in /upload_image route: {e}")
        return jsonify({"error": "An unexpected error occurred during image processing."}), 500

@app.route("/get_recommendations", methods=["POST"])
def get_recommendations_route():
    data = request.get_json()
    location = data.get("location") if data.get("location") else "Egypt"
    interests = data.get("interests") if data.get("interests") else "history and culture"
    liked_places = data.get("liked_places") if data.get("liked_places") else "famous landmarks"
    duration = data.get("duration") if data.get("duration") else "3"

    try:
        duration_val = int(duration) if isinstance(duration, str) and duration.isdigit() else duration
    except ValueError:
        duration_val = "3"

    try:
        recommendations = get_recommendations_mock(location, interests, liked_places, duration_val)
        return jsonify({"recommendations": recommendations})
    except Exception as e:
        print(f"Error in /get_recommendations route: {e}")
        return jsonify({"error": "An unexpected error occurred while generating recommendations."}), 500

# New route for handling chat messages
@app.route("/chat_with_horus", methods=["POST"])
def chat_with_horus_route():
    data = request.get_json()
    user_message = data.get("user_message")
    artifact_name = data.get("artifact_name")
    artifact_description = data.get("artifact_description")

    if not user_message or not artifact_name or not artifact_description:
        return jsonify({"error": "Missing required fields for chat (user_message, artifact_name, artifact_description)"}), 400

    try:
        bot_response = generate_chat_response(user_message, artifact_name, artifact_description)
        return jsonify({"bot_response": bot_response})
    except Exception as e:
        print(f"Error in /chat_with_horus route: {e}")
        return jsonify({"error": "An unexpected error occurred while generating chat response."}), 500

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print("IMPORTANT: The Keras model file 'last_model_bgd.keras' was not found.")
        print("The application will run with MOCKED image classification results.")
    app.run(debug=True, port=5000)

