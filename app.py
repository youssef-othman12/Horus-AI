from flask import Flask, request, render_template, jsonify
import os
import io

# --- Original Imports for Image Classification and Chat ---
try:
    from class_labels import class_names
except ImportError:
    print("Warning: class_labels.py not found. Using default class_names for image classification.")
    class_names = ["Default Artifact"] # Placeholder

try:
    from llm_utils import generate_chat_response
except ImportError:
    print("Warning: llm_utils.py not found. Chat functionality will be a placeholder.")
    def generate_chat_response(user_message, artifact_name, artifact_description):
        return "Chat response generation is currently unavailable due to missing llm_utils."

from keras.saving import load_model
import numpy as np
from PIL import Image

# --- New Imports for Recommendation Logic ---
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# --- New Recommendation Logic Setup ---
ATTRACTIONS_DATA = [
    {
        "name": "Egyptian Museum",
        "city": "Cairo",
        "description": "Home to the world's largest collection of Pharaonic antiquities, including treasures from Tutankhamun's tomb.",
        "type": "Pharaonic",
        "popularity": 9,
        "key_artifacts": ["Tutankhamun's Death Mask", "Royal Mummies Collection", "Narmer Palette", "Statue of Khufu"],
        "highlights": "The museum houses over 120,000 artifacts, with the star attraction being King Tutankhamun's golden mask. Visitors can also explore the Royal Mummies Hall featuring perfectly preserved remains of Egypt's most powerful pharaohs.",
        "visiting_tips": "Visit early in the morning to avoid crowds. Plan at least 3 hours to see the main highlights. Photography is allowed in most areas but requires a special ticket.",
        "historical_significance": "Founded in 1902, the museum preserves Egypt's ancient heritage and provides invaluable insights into one of the world's earliest civilizations."
    },
    {
        "name": "Khan el-Khalili",
        "city": "Cairo",
        "description": "Historic souk and bazaar dating to the 14th century, famous for traditional crafts, spices, and Egyptian souvenirs.",
        "type": "Islamic",
        "popularity": 8,
        "notable_features": ["El-Fishawi Café", "Gold District", "Spice Market", "El-Hussein Mosque"],
        "highlights": "This bustling medieval-style marketplace offers a sensory journey through narrow alleyways filled with shops selling everything from hand-crafted jewelry and copper goods to textiles, spices, and perfumes.",
        "visiting_tips": "Best experienced in late afternoon and evening. Bargaining is expected. Visit El-Fishawi café, Cairo's oldest café, for traditional Egyptian tea.",
        "historical_significance": "Established in 1382 as a caravanserai for traveling merchants, it remains the commercial heart of historic Cairo."
    },
    {
        "name": "Al-Azhar Park",
        "city": "Cairo",
        "description": "A beautiful Islamic garden offering panoramic views of historic Cairo, featuring Islamic architectural elements.",
        "type": "Islamic",
        "popularity": 7,
        "notable_features": ["Lakeside Café", "Citadel View Restaurant", "Islamic-Style Gardens", "Historic Views"],
        "highlights": "This 30-hectare urban oasis provides a peaceful escape from Cairo's bustle with formal gardens, fountains, and stunning views of the Citadel and historic Cairo skyline.",
        "visiting_tips": "Visit in late afternoon to enjoy sunset views over the city. The park has excellent restaurants offering both Egyptian and international cuisine.",
        "historical_significance": "Built on what was once a 500-year-old garbage dump, this transformation project was funded by the Aga Khan Trust for Culture and has revitalized the surrounding historic district."
    },
    {
        "name": "Ibn Tulun Mosque",
        "city": "Cairo",
        "description": "One of the oldest and largest mosques in Egypt with a unique spiral minaret and vast courtyard.",
        "type": "Islamic",
        "popularity": 7,
        "architectural_features": ["Spiral Minaret", "Vast Courtyard", "Stucco Decorations", "Gypsum Windows"],
        "highlights": "This 9th-century architectural masterpiece features a unique spiral minaret and an expansive open courtyard surrounded by elegant arcades with distinctive pointed arches.",
        "visiting_tips": "Visit in the morning light for the best photography. Dress modestly and remove shoes before entering the prayer hall. Climb the minaret for panoramic views of Cairo.",
        "historical_significance": "Built in 879 AD, it's the oldest mosque in Egypt that preserves its original form and one of the largest mosques in the world by land area."
    },
    {
        "name": "Karnak Temple",
        "city": "Luxor",
        "description": "A vast temple complex dedicated to the Theban triad of Amun, Mut, and Khonsu, featuring massive columns and obelisks.",
        "type": "Pharaonic",
        "popularity": 9,
        "period": "New Kingdom to Ptolemaic",
        "notable_features": ["Great Hypostyle Hall", "Sacred Lake", "Avenue of Sphinxes", "Obelisks of Hatshepsut"],
        "highlights": "The temple's Great Hypostyle Hall contains 134 massive columns arranged in 16 rows, creating a forest of stone pillars that once supported a now-vanished roof. Many columns are over 10 meters tall and covered with intricate hieroglyphic carvings.",
        "visiting_tips": "Visit early morning or late afternoon to avoid the midday heat. Hire a knowledgeable guide to understand the complex's rich history. The Sound and Light show in the evening offers a different perspective.",
        "historical_significance": "Built over 2,000 years by successive pharaohs, it's the largest religious building ever constructed and was ancient Egypt's most important place of worship."
    },
    {
        "name": "Valley of the Kings",
        "city": "Luxor",
        "description": "Royal burial ground containing tombs of pharaohs from the New Kingdom, including Tutankhamun.",
        "type": "Pharaonic",
        "popularity": 9,
        "period": "New Kingdom",
        "notable_tombs": ["KV62 (Tutankhamun)", "KV17 (Seti I)", "KV7 (Ramses II)", "KV5 (Sons of Ramses II)"],
        "highlights": "This desert valley contains 63 magnificent royal tombs carved deep into the rock, with walls covered in vivid paintings depicting Egyptian mythology and the pharaoh's journey to the afterlife.",
        "visiting_tips": "Standard tickets include access to three tombs of your choice. Special tickets are required for premium tombs like Tutankhamun's. No photography is allowed inside the tombs. Visit early in the morning when temperatures are cooler.",
        "historical_significance": "For nearly 500 years (16th to 11th century BC), this secluded valley served as the burial place for most of Egypt's New Kingdom rulers, marking a shift from the earlier pyramid tombs."
    },
    {
        "name": "Luxor Temple",
        "city": "Luxor",
        "description": "Ancient Egyptian temple complex located on the east bank of the Nile River, known for its colossal statues and beautiful colonnades.",
        "type": "Pharaonic",
        "popularity": 8,
        "period": "New Kingdom",
        "notable_pharaohs": ["Amenhotep III", "Ramses II"],
        "highlights": "Unlike other temples dedicated to gods, Luxor Temple was dedicated to the rejuvenation of kingship. It features a 25-meter tall pink granite obelisk (whose twin now stands in Paris), massive seated statues of Ramses II, and beautiful colonnaded courtyards.",
        "visiting_tips": "Visit at night when the temple is dramatically illuminated. The temple is centrally located in Luxor city and easily accessible on foot from many hotels.",
        "historical_significance": "Connected to Karnak Temple by the Avenue of Sphinxes, this temple was where many pharaohs were crowned, including potentially Alexander the Great."
    },
    {
        "name": "Temple of Hatshepsut",
        "city": "Luxor",
        "description": "Mortuary temple of the female pharaoh Hatshepsut, featuring terraced colonnades set against dramatic cliffs.",
        "type": "Pharaonic",
        "popularity": 8,
        "period": "New Kingdom",
        "dynasty": "18th Dynasty",
        "highlights": "This unique temple features three dramatic ascending terraces with colonnaded facades, set dramatically against the sheer cliffs of Deir el-Bahari. Relief sculptures depict the divine birth of Hatshepsut and her famous trading expedition to the land of Punt.",
        "visiting_tips": "Visit early morning for the best lighting and views. The site has limited shade, so bring sunscreen and water. A short electric train connects the parking area to the temple entrance.",
        "historical_significance": "Built for one of Egypt's few female pharaohs who ruled for 20 years as king. After her death, her successor Thutmose III attempted to erase her legacy by destroying her images."
    },
    {
        "name": "Abu Simbel",
        "city": "Aswan",
        "description": "Massive rock temples built by Ramses II, featuring colossal statues and intricate carvings.",
        "type": "Pharaonic",
        "popularity": 9,
        "period": "New Kingdom",
        "dynasty": "19th Dynasty",
        "highlights": "Two massive rock temples with four 20-meter high seated statues of Ramses II guarding the entrance. Twice a year (February 22 and October 22), the sun penetrates the main temple to illuminate the innermost sanctuary statues.",
        "visiting_tips": "Most visitors arrive on day trips from Aswan by plane or convoy. Visit early morning to avoid crowds and heat. The Sound and Light show in the evening is spectacular.",
        "historical_significance": "In the 1960s, both temples were completely dismantled and relocated 65 meters higher to save them from submersion when the Aswan High Dam created Lake Nasser - one of the greatest archaeological rescue operations in history."
    },
    {
        "name": "Philae Temple",
        "city": "Aswan",
        "description": "Island temple complex dedicated to the goddess Isis, rescued from the rising waters of Lake Nasser after the Aswan Dam.",
        "type": "Pharaonic",
        "popularity": 8,
        "period": "Ptolemaic to Roman",
        "highlights": "Set on a picturesque island, this beautiful temple complex combines Egyptian and Greco-Roman architectural elements. The main temple is dedicated to Isis, sister-wife of Osiris and mother of Horus.",
        "visiting_tips": "Accessible only by boat, which adds to the experience. The Sound and Light show is among Egypt's best. Morning visits offer better lighting for photography.",
        "historical_significance": "This was the last active temple of the ancient Egyptian religion, with hieroglyphics still being added in the 5th century AD. The temple was completely dismantled and relocated when the Aswan Dam was built."
    },
    {
        "name": "The Unfinished Obelisk",
        "city": "Aswan",
        "description": "Enormous obelisk abandoned in the quarry when cracks appeared, providing insights into ancient stoneworking techniques.",
        "type": "Pharaonic",
        "popularity": 7,
        "period": "New Kingdom",
        "highlights": "This massive unfinished obelisk would have been the largest ever erected at 42 meters tall and weighing 1,200 tons. Its partial carving offers unique insights into ancient Egyptian stone quarrying and carving techniques.",
        "visiting_tips": "Visit in the morning when temperatures are cooler. A knowledgeable guide can explain the ancient quarrying techniques visible at the site.",
        "historical_significance": "Likely commissioned by Queen Hatshepsut, it was abandoned when cracks appeared during carving. It demonstrates the incredible stone-working skills of ancient Egyptians without modern technology."
    },
    {
        "name": "Elephantine Island",
        "city": "Aswan",
        "description": "Island with ruins of the Temple of Khnum and a nilometer used to measure the Nile flood levels.",
        "type": "Pharaonic",
        "popularity": 6,
        "period": "Multiple periods",
        "highlights": "This peaceful island in the middle of the Nile features ancient temple ruins, a museum with artifacts spanning 5,000 years, and one of the oldest nilometers used to measure the critical Nile floods.",
        "visiting_tips": "Easily reached by local ferry or felucca. The Aswan Museum displays artifacts from the island. The Nubian villages on the southern end offer cultural experiences and colorful architecture.",
        "historical_significance": "Served as Egypt's southern frontier for much of its history, with strategic and economic importance as the gateway to Nubia and Africa. Archaeological evidence shows continuous settlement since the Predynastic period."
    },
    {
        "name": "Bibliotheca Alexandrina",
        "city": "Alexandria",
        "description": "Modern library and cultural center built to recapture the spirit of the ancient Library of Alexandria.",
        "type": "Modern",
        "popularity": 8,
        "highlights": "This striking modern architectural marvel houses multiple libraries, four museums, a planetarium, and numerous art galleries and exhibition spaces. The main reading room can accommodate 2,000 readers under its sloping glass roof.",
        "visiting_tips": "Join a guided tour to fully appreciate the architecture and facilities. The Antiquities Museum and Manuscript Museum inside are worth visiting. Check the website for cultural events and exhibitions.",
        "historical_significance": "Built as a memorial to the ancient Library of Alexandria, once the largest in the world and center of learning in the ancient world until its destruction in antiquity."
    },
    {
        "name": "Citadel of Qaitbay",
        "city": "Alexandria",
        "description": "15th-century defensive fortress built on the site of the ancient Lighthouse of Alexandria.",
        "type": "Islamic",
        "popularity": 8,
        "highlights": "This picturesque medieval fortress features thick walls, winding passages, and panoramic views of the Mediterranean. Built with stones from the collapsed Lighthouse of Alexandria, one of the Seven Wonders of the Ancient World.",
        "visiting_tips": "Visit late afternoon for beautiful sunset views over the Mediterranean. Wear comfortable shoes as there are many stairs to climb. The Naval Museum inside has modest displays but interesting artifacts.",
        "historical_significance": "Built in 1477 by Sultan Qaitbay on the exact site of the famous Lighthouse of Alexandria (Pharos), which had collapsed after an earthquake. It served as an important defensive stronghold against Ottoman attacks."
    },
    {
        "name": "Catacombs of Kom El Shoqafa",
        "city": "Alexandria",
        "description": "Vast Roman-era underground necropolis combining Egyptian, Greek, and Roman artistic elements.",
        "type": "Greco-Roman",
        "popularity": 7,
        "highlights": "These three-level underground tomb complexes feature a unique blend of Pharaonic, Greek and Roman artistic elements. The main tomb chamber has sculptures showing Egyptian gods in Roman dress, demonstrating the cultural fusion of the time.",
        "visiting_tips": "Bring a sweater as it can be cool underground. The site requires some stair climbing. Photography is permitted but without flash.",
        "historical_significance": "Dating from the 2nd century AD, these are considered one of the Seven Wonders of the Middle Ages. They demonstrate the multicultural nature of Roman Alexandria with their fusion of artistic styles."
    },
    {
        "name": "Montazah Palace Gardens",
        "city": "Alexandria",
        "description": "Extensive royal gardens surrounding the Montazah Palace with beaches, woods, and formal gardens.",
        "type": "Modern",
        "popularity": 7,
        "highlights": "This 150-acre royal park features beautiful landscaped gardens, palm-lined avenues, and the distinctive Montazah Palace with its blend of Turkish and Florentine architectural styles. The park includes private beaches and woods.",
        "visiting_tips": "A perfect escape from Alexandria's urban bustle. While the palace itself is not open to the public, the gardens and beaches are accessible with an entrance ticket. Bring a picnic and swimwear in summer.",
        "historical_significance": "Built by Khedive Abbas II, the last Muhammad Ali Dynasty ruler, in 1892 as a summer residence for the Egyptian royal family. After the 1952 revolution, it became a presidential palace."
    },
    {
        "name": "Great Pyramids of Giza",
        "city": "Giza",
        "description": "The last remaining wonder of the ancient world, massive structures built as tombs for the pharaohs.",
        "type": "Pharaonic",
        "popularity": 10,
        "period": "Old Kingdom",
        "dynasty": "4th Dynasty",
        "notable_pharaohs": ["Khufu", "Khafre", "Menkaure"],
        "highlights": "The Great Pyramid of Khufu stands 147 meters tall and contains over 2.3 million stone blocks weighing 2.5-15 tons each. The precision of construction is remarkable - the base is level to within 2.1 cm, and the sides are aligned to the cardinal directions with an accuracy of up to 0.05 degrees.",
        "visiting_tips": "Arrive early morning or late afternoon to avoid crowds and midday heat. Entrance tickets to the pyramid interiors are limited and sold separately. Camel and horse rides are negotiable but agree on price beforehand.",
        "historical_significance": "Built around 2560 BC, the Great Pyramid remained the tallest human-made structure in the world for nearly 4,000 years. The complex demonstrates the Egyptians' advanced knowledge of mathematics, astronomy, and engineering."
    },
    {
        "name": "Great Sphinx of Giza",
        "city": "Giza",
        "description": "Massive limestone statue with the body of a lion and the head of a human, thought to represent King Khafre.",
        "type": "Pharaonic",
        "popularity": 9,
        "period": "Old Kingdom",
        "dynasty": "4th Dynasty",
        "highlights": "This enigmatic monument stands 20 meters tall and 73 meters long, making it the largest monolithic statue in the world. Carved from a single ridge of limestone, it has captured human imagination for thousands of years.",
        "visiting_tips": "Visit early morning or close to sunset for dramatic lighting and photographs. The Sphinx is viewed from a viewing platform at its base - you cannot touch or climb it.",
        "historical_significance": "Shrouded in mystery regarding its exact purpose and construction date. Between its paws stands the Dream Stela, placed by Thutmose IV, telling how the Sphinx appeared in his dream promising kingship if he cleared the sand covering it."
    },
    {
        "name": "Pyramids of Giza Sound and Light Show",
        "city": "Giza",
        "description": "Nighttime spectacle that brings ancient history to life through dramatic narration, music, and illumination of the pyramids and Sphinx.",
        "type": "Pharaonic",
        "popularity": 8,
        "highlights": "This evening show uses dramatic lighting effects, music, and narration to tell the story of ancient Egypt. The pyramids and Sphinx are illuminated in changing colors while the voice of the Sphinx recounts 5,000 years of Egyptian history.",
        "visiting_tips": "Shows are presented in different languages on different nights - check the schedule. Booking in advance is recommended in high season. Bring a jacket as desert evenings can be cool.",
        "historical_significance": "Though a modern attraction, the show helps visitors connect with the ancient history and mythology surrounding these monuments, using advanced technology to tell ancient stories."
    },
    {
        "name": "Tomb of Meresankh III",
        "city": "Giza",
        "description": "Exceptionally well-preserved tomb of a queen from the 4th Dynasty with vivid colors and statues.",
        "type": "Pharaonic",
        "popularity": 7,
        "period": "Old Kingdom",
        "dynasty": "4th Dynasty",
        "highlights": "This hidden gem features remarkably preserved colorful reliefs and ten life-sized statues of women carved from the living rock. The burial chamber walls retain their vibrant original colors after more than 4,500 years.",
        "visiting_tips": "Located in the Eastern Cemetery near the Great Pyramid. Less visited than other attractions, offering a more intimate experience. A special ticket may be required as it's often opened on rotation with other tombs.",
        "historical_significance": "Meresankh III was the granddaughter of King Khufu and wife of King Khafre. Her tomb provides rare insights into the lives of royal women in the Old Kingdom and features some of the best-preserved Old Kingdom paintings."
    }
]
attractions_df = pd.DataFrame(ATTRACTIONS_DATA)

RECOMMENDATION_SYSTEM_MODEL_ST = None
ATTRACTION_EMBEDDINGS = None
RECOMMENDATION_SYSTEM_READY = False
print("Loading sentence transformer model for recommendations...")
try:
    RECOMMENDATION_SYSTEM_MODEL_ST = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    print("Recommendation sentence transformer model loaded successfully!")
    print("Generating embeddings for attractions...")
    ATTRACTION_EMBEDDINGS = RECOMMENDATION_SYSTEM_MODEL_ST.encode(attractions_df['description'].tolist())
    print(f"Generated {len(ATTRACTION_EMBEDDINGS)} embeddings with dimension {ATTRACTION_EMBEDDINGS.shape[1]}")
    RECOMMENDATION_SYSTEM_READY = True
except Exception as e:
    print(f"Error loading sentence transformer model or generating embeddings: {e}")
    print("Recommendation system will not be fully functional. Please ensure 'sentence-transformers', 'pandas', and 'scikit-learn' are installed.")

# --- Original Image Classification Model Setup ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "last_model_bgd.keras")
image_classification_model = None
if os.path.exists(MODEL_PATH):
    try:
        image_classification_model = load_model(MODEL_PATH)
        print(f"Image classification model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading Keras model from {MODEL_PATH}: {e}")
        image_classification_model = None
else:
    print(f"Warning: Image classification model file not found at {MODEL_PATH}. Classification will be mocked.")

# --- Original Image Classification Functions (Unchanged) ---
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def classify_image(image_bytes):
    if image_classification_model is None:
        print("Image classification model not loaded, using mock classification.")
        # Ensure class_names is not empty before accessing
        cn = class_names[0] if class_names and len(class_names) > 0 else "Mocked Artifact"
        return cn, f"This is a mocked English description for {cn} as the model is not available."
    
    if not image_bytes:
        return "Error: No image data", ""

    try:
        preprocessed_image = preprocess_image(image_bytes)
        predictions = image_classification_model.predict(preprocessed_image)
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

# --- New Recommendation Function (Replaces get_recommendations_mock) ---
def generate_text_recommendations(current_location, interests, liked_places=None, top_n=3):
    if not RECOMMENDATION_SYSTEM_READY or RECOMMENDATION_SYSTEM_MODEL_ST is None or ATTRACTION_EMBEDDINGS is None:
        return "Recommendation system is currently unavailable. Please check logs for model loading errors."
    
    if isinstance(interests, str):
        interests = [interest.strip() for interest in interests.split(',') if interest.strip()]
    if not interests: # Default interests if none provided
        interests = ["egyptian history", "culture"]
        
    interests = [interest.lower() for interest in interests]
    
    recommendations = attractions_df.copy()
    
    if current_location and current_location.lower() != 'all' and current_location.lower() != 'any':
        recommendations['location_score'] = (recommendations['city'].str.lower() == current_location.lower()).astype(int)
    else:
        recommendations['location_score'] = 1
    
    interests_text = " ".join(interests)
    interests_embedding = RECOMMENDATION_SYSTEM_MODEL_ST.encode([interests_text])[0]
    
    similarity_scores = cosine_similarity([interests_embedding], ATTRACTION_EMBEDDINGS)[0]
    recommendations['interest_score'] = similarity_scores
    
    if liked_places and len(liked_places) > 0:
        liked_indices = []
        for place in liked_places:
            indices = recommendations.index[recommendations['name'].str.lower() == place.lower()].tolist()
            liked_indices.extend(indices)
        
        if liked_indices:
            liked_embeddings_val = ATTRACTION_EMBEDDINGS[liked_indices]
            history_scores = cosine_similarity(ATTRACTION_EMBEDDINGS, liked_embeddings_val).mean(axis=1)
            recommendations['history_score'] = history_scores
        else:
            recommendations['history_score'] = 0
    else:
        recommendations['history_score'] = 0
    
    recommendations['final_score'] = (
        0.2 * recommendations['location_score'] +
        0.5 * recommendations['interest_score'] +
        0.2 * recommendations['history_score'] +
        0.1 * (recommendations['popularity'] / 10)
    )
    
    top_recommendations_df = recommendations.sort_values('final_score', ascending=False).head(top_n)
    
    if top_recommendations_df.empty:
        return "No specific recommendations found based on your preferences. Try broadening your search or checking your spelling!"

    results_text = "Top Recommended Egyptian Attractions for you:\n\n"
    for i, (idx, row) in enumerate(top_recommendations_df.iterrows(), 1):
        results_text += f"{i}. {row['name']} ({row['city']}) - {row['type']}\n"
        # Ensure final_score is present and calculate match_score
        match_score = round(row["final_score"] * 100, 1) if "final_score" in row and isinstance(row["final_score"], (int, float)) else "N/A"
        results_text += f"   Match Score: {match_score}%\n"
        results_text += f"   Description: {row['description']}\n\n"
    
    return results_text.strip()

# --- Original Routes (Unchanged, except /get_recommendations) ---
@app.route("/")
def index():
    return render_template("horos1.html") # Original index page

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

# --- Modified /get_recommendations Route ---
@app.route("/get_recommendations", methods=["POST"])
def get_recommendations_route():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided for recommendations"}), 400
        
    location = data.get("location")
    interests = data.get("interests") # Expected to be a string like "history, culture"
    liked_places_input = data.get("liked_places") # Expected to be a string like "Pyramids, Museum"
    # Duration is no longer used by the new recommendation logic directly for structuring itinerary
    # duration = data.get("duration")

    liked_places_list = []
    if isinstance(liked_places_input, str) and liked_places_input.strip():
        liked_places_list = [p.strip() for p in liked_places_input.split(',') if p.strip()]
    elif isinstance(liked_places_input, list):
        liked_places_list = [str(p).strip() for p in liked_places_input if str(p).strip()] # Ensure strings

    current_location_param = location if location else "All" # Default to 'All' if not provided
    interests_param = interests if interests else "history, culture" # Default interests

    try:
        recommendations_text_output = generate_text_recommendations(
            current_location_param, 
            interests_param, 
            liked_places_list,
            top_n=3 # Number of recommendations to return
        )
        return jsonify({"recommendations": recommendations_text_output})
    except Exception as e:
        print(f"Error in /get_recommendations route: {e}")
        fallback_message = "We encountered an issue generating recommendations. Please try again later."
        if not RECOMMENDATION_SYSTEM_READY:
            fallback_message = "Recommendation system is currently initializing or unavailable. Please try again shortly."
        # Return the error and a fallback message in the expected format
        return jsonify({"error": str(e), "recommendations": fallback_message}), 500

# --- Original Chat Route (Unchanged) ---
@app.route("/chat_with_horus", methods=["POST"])
def chat_with_horus_route():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided for chat"}),400
        
    user_message = data.get("user_message")
    artifact_name = data.get("artifact_name")
    artifact_description = data.get("artifact_description")

    if not user_message or not artifact_name or not artifact_description:
        return jsonify({"error": "Missing required fields for chat (user_message, artifact_name, artifact_description)"}), 400

    try:
        # Check if generate_chat_response is the placeholder or the actual one
        if 'generate_chat_response' in globals() and callable(generate_chat_response):
            bot_response = generate_chat_response(user_message, artifact_name, artifact_description)
        else: # Should not happen if placeholder is defined correctly
            bot_response = "Chat functionality is currently unavailable."
        return jsonify({"bot_response": bot_response})
    except Exception as e:
        print(f"Error in /chat_with_horus route: {e}")
        return jsonify({"error": "An unexpected error occurred while generating chat response."}), 500

# --- Combined __main__ Block ---
if __name__ == "__main__":
    if image_classification_model is None:
        print("IMPORTANT: The Keras model file 'last_model_bgd.keras' was not found or failed to load.")
        print("The application will run with MOCKED image classification results.")
    else:
        print("Image classification Keras model loaded.")
        
    if not RECOMMENDATION_SYSTEM_READY:
        print("IMPORTANT: The recommendation system may not be fully functional due to model loading issues.")
        print("Please check messages above for errors related to 'sentence-transformers'.")
        print("You might need to install necessary packages: pip install sentence-transformers pandas scikit-learn tensorflow Pillow")
    else:
        print("Recommendation system ready.")
        
    app.run(debug=True, port=5000)

