import openai
import os
import base64
import pandas as pd

'''
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
'''

"You are a professional traffic scenario assistant. Interpret top-down traffic maps accurately."

class mapInterpretation:
    def __init__(self, api_key=None):
        """
        Initialize the OpenAI client with the provided API key,
        essential traffic rules, and map comprehension guidelines.
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("API key must be provided either directly or through the 'OPENAI_API_KEY' environment variable.")
        
        self.client = openai.OpenAI(api_key=self.api_key)

    def analyze_map_image(self, image_url: str, prompt: str, max_tokens: int = 300):
        """
        Use OpenAI's VLM model to analyze a top-down traffic map image.

        :param image_url: URL to the top-down traffic map image
        :param prompt: Instructional text to analyze the image
        :param max_tokens: Maximum number of tokens in the response
        :return: LLM-generated description
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a professional art critic."},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]}
                ],
                max_tokens=max_tokens
            )
            print("Prompt tokens:", response.usage.prompt_tokens)
            print("Completion tokens:", response.usage.completion_tokens)
            print("Total tokens used:", response.usage.total_tokens)
            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Error interpreting map image: {e}")
            return None
        
# Example usage
if __name__ == "__main__":
    interpreter = mapInterpretation()
    image_url = "https://raw.githubusercontent.com/lemonwilliam/LLM-TSG/refs/heads/main/screen_shot_00290.jpg?token=GHSAT0AAAAAADCP7FVNDDBNZYC54A3P2VV22AQ3WVA"
    description = interpreter.analyze_map_image(image_url, prompt="Describe the image given.")
    if description:
        print("\nMap Description:\n", description)
    else:
        print("Failed to interpret the map image.")

