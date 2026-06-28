import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.responses.create(
    model="gpt-4o-mini",
    input=[
        {
            "role":"system",
            "content":"You are an expert problem solver."
        },
        {
            "role":"user",
            "content":"""
Solve the following problem.

Consider different possible approaches internally.
Choose the most consistent solution.

Only provide:
- Explanation
- Final Answer

Question:
If a shop gives a 20% discount on a ₹5000 item, what is the final price?
"""
        }
    ]
)

print(response.output_text)