import os
import openai
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain


class MapInterpretation:
    def __init__(self, api_key=None):
        """
        Initialize the OpenAI client and LangChain memory pipeline.
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("API key must be provided either directly or through the 'OPENAI_API_KEY' environment variable.")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # LangChain setup for memory-based follow-up analysis
        self.llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=self.api_key)
        self.memory = ConversationBufferMemory(memory_key="history", return_messages=True)
        self.chain = ConversationChain(llm=self.llm, memory=self.memory)

        '''
        self.task_description = task_description if task_description else None
        self.condition_definitions = condition_definitions if condition_definitions else None
        '''


    def analyze_map_image(self, image_url: str, max_tokens: int = 500):
        """
        Use OpenAI Vision API to analyze the top-down traffic map image.
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a traffic scenario assistant. Interpret top-down traffic maps accurately."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Please describe the structure of this intersection map."},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]
                    }
                ],
                max_tokens=max_tokens
            )
            content = response.choices[0].message.content.strip()
            print("Image analysis complete.")
            print("Prompt tokens:", response.usage.prompt_tokens)
            print("Completion tokens:", response.usage.completion_tokens)
            print("Total tokens used:", response.usage.total_tokens)
            return content
        except Exception as e:
            print(f"Error during image analysis: {e}")
            return None

    def feed_map_understanding_to_memory(self, map_description: str):
        """
        Save the result of Task 1 (map analysis) into LangChain memory.
        """
        self.chain.run(f"This is the result of the map analysis:\n{map_description}")

    def analyze_agent_behaviors(self, behavior_log: str):
        """
        Task 2: Analyze agent behaviors using previously stored map understanding.
        """
        prompt = f"""
Now that you understand the underlying map, analyze the following agent behaviors that occur on this map.

Agent Behavior Log:
{behavior_log}

For each agent, explain the likely reason for its action:
- due to road structure
- interaction with another agent
- or no specific reason
"""
        return self.chain.run(prompt)
