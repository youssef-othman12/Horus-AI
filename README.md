# 🧠 Horus AI: Guardian of Ancient Egyptian Civilization

![Horus](https://github.com/user-attachments/assets/f07eb3ad-9123-4b16-9e63-f11cbd3405ea)

> *"Let the wisdom of the ancients meet the power of artificial intelligence."*

---

## 📽️ Ad Video

🎥 [**Watch the Ad Video**](https://drive.google.com/file/d/1lmisRs3lUKR51qWwAwyi_4d-P7ebV8g0/view?usp=sharing)

> 🧠 *This video ad was created entirely using AI tools.*

---

## 🔍 Introduction

**Have You Ever Lived History?**
Horus AI is more than just software — it’s a digital bridge between the past and the present.
It reimagines how we interact with ancient Egyptian artifacts using the power of AI.

* **Egypt's Legacy:** One of humanity’s greatest civilizations
* **Beyond Sight:** Experience history, not just see it
* **Technology’s Role:** Live history through advanced tech

---

## 🤖 The Problem

Current AI models like ChatGPT, Claude, and Gemini often:

* Provide **general or vague responses**
* **Misclassify historical images**
* Lack **deep cultural understanding** of ancient artifacts

---

## 💡 Our Solution: Horus AI

**Horus AI** teaches artificial intelligence to truly understand history — not just recognize it.

### 🔍 Features

| Feature                 | Description                                             |
| ----------------------- | ------------------------------------------------------- |
| 📸 Image Classification | Identify ancient artifacts with CNN + transfer learning |
| 📝 Descriptions         | Generate accurate and engaging historical content       |
| 🗺️ Recommendations     | Get personalized site suggestions and travel tips       |
| 💬 Virtual Guide        | Ask questions via a smart Gemini-powered chat assistant |

---

## 🔧 The Power Behind the Eye

| Technology            | Role                                                          |
| --------------------- | ------------------------------------------------------------- |
| 🧠 Keras + TensorFlow | Image classification using transfer learning                  |
| 🔤 Google Gemini API  | Generates context-aware responses and historical explanations |
| 🌐 Flask              | User-friendly web app to access Horus AI                      |
| ⚙️ Modular Codebase   | Efficient and maintainable project structure                  |

---

## 🧪 Model Development

### 🔬 Data Collection & Augmentation

* Curated top-quality datasets of ancient Egyptian artifacts
* **Augmentation Techniques**:

  * Rotation, flipping, zooming, cropping
  * Brightness/contrast shifts

### ⚙️ Preprocessing

* Resize images
* Filter low-resolution data
* Balance underrepresented and overrepresented classes

### 🧠 Baseline Model

* **Model:** CNN using Keras
* **Initial Accuracy:** \~50%
* **Improvements:**

  * Hyperparameter tuning
  * Class merging and relabeling
  * Manual data verification
  * Final Accuracy: **\~80%**

---

## 📊 Error Analysis Highlights

* **Issue:** Classes like *Ramessum* and *Ramesseum* were split unnecessarily
* **Fix:** Merged confusing or duplicate classes
* **Result:** Reduced misclassification (e.g., Sphinx misclassified under *Giza\_Pyramid\_Complex*)

---

## 🎯 Future Goals: 90% Accuracy

* 🧪 Advanced augmentations
* 🔁 More real-world data collection
* 💻 Better pretrained models
* 🧹 Clean, labeled, and balanced datasets

---

## 💬 Gemini-Powered Chatbot

**Horus AI Assistant** is built using **Google Gemini Pro**:

* 💡 Provides cultural, accurate explanations
* 🗣️ Responds to historical queries interactively
* 📌 Integrated feedback loop for better personalization

---

## 🌍 Web Interface Highlights

| Component            | Description                                           |
| -------------------- | ----------------------------------------------------- |
| 🧠 Model Integration | `model_utils.py` handles classification logic         |
| 🔤 NLP Utilities     | `llm_utils.py` manages Gemini API interactions        |
| 🌐 Frontend          | Flask + HTML/CSS for uploading, viewing, and chatting |

---

## 🧭 How The System Works

1. **Upload Image**: Submit an artifact image
2. **Classification**: AI identifies the artifact
3. **Description**: Get historical context
4. **Recommendations**: Travel site suggestions
5. **Live Q\&A**: Chat with Horus AI

---

## 🧩 Project Structure

```
flask_project/
├── app.py                  # Main Flask app
├── class_labels.py         # Artifact labels
├── last_model.keras        # Trained CNN model
├── llm_utils.py            # Gemini API logic
├── model_utils.py          # Image processing
├── requirements.txt
├── static/
│   └── images/
└── templates/
    └── index.html, etc.
```

---

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/Nadercr7/Horus-AI-Depi
cd flask_project

# Set up environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Configure API key
echo "GEMINI_API_KEY=your_api_key_here" > .env

# Run the app
flask run
```

Visit: `http://127.0.0.1:5000`

---

## 🙌 Final Thoughts

### 🔁 Progress Recap

From raw data to a polished web app with 80% classification accuracy — we combined:

* Deep learning (CNN)
* Transfer learning
* NLP (Gemini)
* Error analysis and human feedback

### 🚀 Vision

To create **intelligent, accessible archaeology tools** where:

* AI becomes a **historical companion**
* Learning about civilizations is **immersive and personalized**
