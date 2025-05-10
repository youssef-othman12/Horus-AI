import google.generativeai as genai
import os

# Configure the Gemini API key
GEMINI_API_KEY = "AIzaSyBLh8Z1bLLNR1xLLQmaiZwiZ9JJYDK8aAw"
genai.configure(api_key=GEMINI_API_KEY)

def generate_chat_response(user_message, artifact_name, artifact_description):
    """
    Generates a response from the Gemini LLM based on user message and artifact context.
    """
    try:
        
        model = genai.GenerativeModel('models/gemini-1.5-flash')

        # Construct prompt
        prompt = (
            f"You are Horus AI, an expert on ancient Egyptian artifacts. "
            f"A user is asking about an artifact identified as '{artifact_name}'. "
            f"Description: '{artifact_description}'.\n\n"
            f"User: {user_message}\n"
            f"Horus AI:"
        )

        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return "I apologize, I encountered an issue trying to respond. Please try again later."
