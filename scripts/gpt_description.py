import openai
import os
import json
import pandas as pd

'''
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
'''

class gptDescription:
    def __init__(self, api_key=None, task_description=None, action_definitions=None, trigger_definitions=None):
        """
        Initialize the OpenAI client with the provided API key,
        essential task description, and comprehension guidelines.
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("API key must be provided either directly or through the 'OPENAI_API_KEY' environment variable.")
        
        self.client = openai.OpenAI(api_key=self.api_key)

        '''
        Store task description and guidelines for contextual learning
        '''
        self.task_description = task_description if task_description else None
        self.action_definitions = action_definitions if action_definitions else None
        self.trigger_definitions = trigger_definitions if trigger_definitions else None
        

    def generate_response(self, metadata=None, map_content=None, trajectory_content=None, temperature=0.5):
        """
        Generate a response using the GPT-4o model with essential knowledge, map, and trajectory context.
        """
        try:
            messages = [{"role": "system", "content": "You are a highly accurate and logical traffic scenario analyzer. Your tasks involve understanding vehicle behavior from trajectory and map data then generating a structured description of the scenario suitable for code-based conversion to the ASAM OpenSCENARIO format."}]
            
            if self.task_description:
                messages.append({"role": "user", "content": f"Task Description:\n{self.task_description}"})

            if self.action_definitions:
                messages.append({"role": "user", "content": f"Action Comprehension Guidelines:\n{self.action_definitions}"})

            if self.trigger_definitions:
                messages.append({"role": "user", "content": f"Trigger Condition Comprehension Guidelines:\n{self.trigger_definitions}"})
            
            if metadata:
                messages.append({"role": "user", "content": f"Meta Data:\n{metadata}"})

            if map_content:
                messages.append({"role": "user", "content": f"Map Data:\n{map_content}"})
            
            if trajectory_content:
                messages.append({"role": "user", "content": f"Trajectory Data:\n{trajectory_content}"})

            response = self.client.chat.completions.create(
                model="o3-mini",
                messages=messages,
            )
            print("Prompt tokens:", response.usage.prompt_tokens)
            print("Completion tokens:", response.usage.completion_tokens)
            print("Total tokens used:", response.usage.total_tokens)
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

