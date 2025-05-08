from flask import Flask, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import requests

from class_labels import class_names

app = Flask(__name__, static_folder="static")


model = load_model("last_model_bgd.keras")  # Place your model in the same folder

GEMINI_API_KEY = "AIzaSyBLh8Z1bLLNR1xLLQmaiZwiZ9JJYDK8aAw"


def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def classify_image(image_bytes):
    img_array = preprocess_image(image_bytes)
    predictions = model.predict(img_array)
    class_idx = int(np.argmax(predictions[0]))
    return class_names[class_idx]


def ask_gemini(message):
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"text": message}]}]}
    response = requests.post(
        endpoint, headers={"Content-Type": "application/json"}, json=payload
    )
    result = response.json()
    return result["candidates"][0]["content"]["parts"][0]["text"]


@app.route("/interact", methods=["POST"])
def interact():
    print("Request received")
    image = request.files.get("image")
    question = request.form.get("question", "").strip()
    print("Image received:", bool(image))
    print("Question received:", question)

    response_text = ""

    if image:
        try:
            image_bytes = image.read()
            class_name = classify_image(image_bytes)
            print("Classified as:", class_name)
            response_text += f"The photo seems to be: {class_name}.\n"

            if not question:
                description_prompt = (
                    f"Describe this Egyptian artifact, person, or place: {class_name}. "
                    "Explain it as if you're a friendly tour guide talking to a tourist in a simple and engaging way. "
                    "Please keep your description concise, no more than 100 words."
                )
                g_response = ask_gemini(description_prompt)
                print("Gemini response:", g_response[:100])
                response_text += g_response
        except Exception as e:
            print("Error during classification or Gemini call:", e)
            return (
                jsonify({"error": "Failed to classify or generate description."}),
                500,
            )

    if question:
        try:
            final_prompt = question
            if image:
                final_prompt = f"The user is asking about the image: {class_name}. Question: {question}"
            g_response = ask_gemini(final_prompt)
            print("Gemini Q response:", g_response[:100])
            response_text += "\n" + g_response
        except Exception as e:
            print("Error during Gemini Q:", e)
            return jsonify({"error": "Failed to process question."}), 500

    if not image and not question:
        print("No input provided.")
        return jsonify({"error": "Please upload an image or ask a question."}), 400

    return jsonify({"reply": response_text.strip()})


@app.route("/")
def serve_index():
    return send_from_directory("static", "index.html")


if __name__ == "__main__":
    app.run(debug=True)
