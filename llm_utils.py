import google.generativeai as genai
import os

# Configure the Gemini API key
# In a production environment, use environment variables or a config file
GEMINI_API_KEY = "AIzaSyBLh8Z1bLLNR1xLLQmaiZwiZ9JJYDK8aAw" # Provided by user
genai.configure(api_key=GEMINI_API_KEY)

# Initialize the generative model
# model = genai.GenerativeModel('gemini-pro') # Or other suitable model

# For chat, it's better to use a chat session
# chat_model = genai.GenerativeModel('gemini-pro')

def generate_chat_response(user_message, artifact_name, artifact_description):
    """
    Generates a response from the Gemini LLM based on user message and artifact context.
    """
    try:
        model = genai.GenerativeModel('gemini-pro')
        # Construct a prompt that includes context about the artifact
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

# Example usage (for testing this module independently):
# if __name__ == '__main__':
#     test_artifact_name = "Pharaoh's Mask"
#     test_artifact_description = "A golden funerary mask from the 18th dynasty."
#     test_user_message = "Tell me more about its significance."
#     bot_response = generate_chat_response(test_user_message, test_artifact_name, test_artifact_description)
#     print(f"User: {test_user_message}")
#     print(f"Horus AI: {bot_response}")

