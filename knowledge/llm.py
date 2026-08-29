import os
from google import genai

class LLM:
    def generate(self, prompt):
        client = genai.Client()
        response = client.models.generate_content(
            model = "gemini-2.5-flash",
            contents = prompt
        )
        return response.text