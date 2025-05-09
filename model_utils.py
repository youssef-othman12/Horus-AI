from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

# This list would typically be loaded from a separate file (e.g., class_labels.py)
# For this self-contained example, we define it directly.
class_names = [
    "Ancient Egyptian Writing Sample",
    "Anubis",
    "Eye of Horus",
    "Horus",
    "Isis",
    "Mummy",
    "Pyramids",
    "Scarab Beetle",
    "Sphinx",
    "Tutankhamun"
]

# Load the Keras model (replace 'last_model_bgd.keras' with the actual path if different)
# For the purpose of this example, we'll assume the model is in the same directory as this script.
# In a real Flask app, you'd place this in your app's initialization or a dedicated utility module.
# model = load_model('last_model_bgd.keras') # This line would be used in a real scenario

def preprocess_image(image_bytes):
    """Converts image bytes to a NumPy array suitable for model input."""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))  # Assuming the model expects 224x224 images
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def classify_image_mock(image_bytes):
    """Simulates image classification and returns a class name and description."""
    # In a real application, this function would use the loaded Keras model to predict.
    # For now, we'll just return a fixed result for demonstration.
    # This simulates the model predicting the first class from our list.
    if not image_bytes:
        return "Error: No image data", ""
    
    # Simulate some processing if needed, but for now, keep it simple.
    # In a real scenario, you would call: 
    # preprocessed_image = preprocess_image(image_bytes)
    # predictions = model.predict(preprocessed_image)
    # class_idx = np.argmax(predictions[0])
    # class_name = class_names[class_idx]
    
    # For demonstration, let's assume the image is always classified as the first class.
    class_name = class_names[0] # Example: "Ancient Egyptian Writing Sample"
    description = f"This is a fascinating example of {class_name}. These writings provide invaluable insights into the daily lives, religious beliefs, and governance of ancient Egyptian civilization. They are often found on temple walls, tombs, and papyrus scrolls."
    
    return class_name, description

# Example usage (not part of the Flask app, just for testing this module):
if __name__ == '__main__':
    # This part would not run when imported by another script.
    # You would need a sample image file (e.g., 'sample_image.jpg') in the same directory to test this.
    try:
        with open("sample_image.jpg", "rb") as f:
            image_bytes_content = f.read()
        
        predicted_class, description_text = classify_image_mock(image_bytes_content)
        print(f"Predicted Class: {predicted_class}")
        print(f"Description: {description_text}")

    except FileNotFoundError:
        print("Error: sample_image.jpg not found. Please provide a sample image for testing.")
    except Exception as e:
        print(f"An error occurred: {e}")

