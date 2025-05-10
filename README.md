# Horus AI: Ancient Egypt Artifact Explorer and Guide



## Project Overview

Horus AI is a Flask-based web application designed to explore and provide information about ancient Egyptian artifacts. It allows users to upload images of artifacts for classification, receive AI-generated descriptions, get personalized travel recommendations for historical sites in Egypt, and engage in a chat with an AI assistant (Horus AI) to learn more about the identified artifacts. The application integrates a Keras model for image classification and the Google Gemini API for generating chat responses and potentially enhancing descriptions and recommendations.



## Features

*   **Artifact Image Classification:** Upload an image of an ancient Egyptian artifact, and the application will classify it using a pre-trained Keras model. If the model is unavailable, it provides a mocked classification.
*   **AI-Generated Descriptions:** Get a descriptive text about the classified artifact.
*   **Personalized Travel Recommendations:** Based on optional user inputs like location, interests, liked places, and duration of stay, the application provides mock travel recommendations for historical sites in Egypt.
*   **Interactive Chat with Horus AI:** Engage in a conversation with "Horus AI" (powered by Google Gemini Pro) to ask questions and learn more details about the identified artifact. The chat is context-aware, using the artifact's name and description.
*   **User-Friendly Web Interface:** The application provides a simple and intuitive web interface built with Flask and HTML/CSS for easy interaction.
*   **Modular Design:** The project is structured with separate modules for application logic (`app.py`), utility functions (`llm_utils.py`, `model_utils.py`), and class labels (`class_labels.py`).



## Setup and Installation

To set up and run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Nadercr7/Horus-AI-Depi]
    cd flask_project
    ```
2.  **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set up environment variables:**
    Create a `.env` file in the project root directory and add the following:
    ```env
    GEMINI_API_KEY=your_gemini_api_key
    ```
    Replace `your_gemini_api_key` with your actual Gemini API key.

5.  **Run the application:**
    ```bash
    flask run
    ```
    The application will be accessible at `http://127.0.0.1:5000` by default.

## Usage

1.  Navigate to the application URL in your web browser.
2.  Upload an image of an ancient Egyptian artifact.
3.  The application will classify the artifact and display its name and a brief description.
4.  You can then interact with Horus AI to ask further questions about the artifact or request travel recommendations.

## Project Structure

```
flask_project/
├── app.py                  # Main Flask application
├── class_labels.py         # Python list of artifact class names
├── last_model.keras        # Pre-trained Keras model for image classification
├── requirements.txt        # Python dependencies
├── static/
│   └── ...                 # Static files (CSS, images, etc.)
└── templates/
    └── ...                 # HTML templates
```

## Contributing

Contributions are welcome! Please follow these steps:

1.  Fork the repository.
2.  Create a new branch: `git checkout -b feature-branch`.
3.  Make your changes and commit them: `git commit -m 'Add some feature'`.
4.  Push to the branch: `git push origin feature-branch`.
5.  Submit a pull request.

## License

This project is licensed under the Horus AI team 

