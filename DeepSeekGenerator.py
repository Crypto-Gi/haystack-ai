from haystack import component
from openai import OpenAI

@component
class DeepSeekGenerator:
    def __init__(self, api_key: str, model: str = "deepseek-chat", temperature: float = 0.7, max_tokens: int = 1024):
        """
        Initialize the DeepSeek generator.
        
        :param api_key: Your DeepSeek API key.
        :param model: The model to use (default is "deepseek-chat").
        :param temperature: Sampling temperature (default is 0.7).
        :param max_tokens: Maximum number of tokens to generate (default is 1024).
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize the OpenAI client with DeepSeek's base URL
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")

    @component.output_types(response=str)
    def run(self, query: str, system_role: str = "You are a helpful assistant"):
        """
        Generate a response using DeepSeek Chat.
        
        :param query: The user's input query as a string.
        :param system_role: The role and content for the system message (default is "You are a helpful assistant").
        
        :return: Generated response as a string.
        """
        # Construct the messages list dynamically
        messages = [
            {"role": "system", "content": system_role},
            {"role": "user", "content": query}
        ]
        
        # Call the DeepSeek API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=False
        )
        
        # Extract and return the generated content
        return {"response": response.choices[0].message.content}
