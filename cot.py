import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.responses.create(
    model="gpt-4o-mini",
    input=[
        {
            "role": "system",
            "content": "You are an expert math tutor."
        },
        {
            "role": "user",
            "content": """
Solve this problem carefully.

Question:
A train travels 60 km in 45 minutes.
What is its average speed in km/h?

Think through the solution internally and provide only:
1. Short explanation
2. Final answer
"""
        }
    ]
)

print(response.output_text)