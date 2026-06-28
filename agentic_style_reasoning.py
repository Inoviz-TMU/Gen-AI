import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

system_prompt = """
You are an AI analyst.

For every task:

1. Understand the problem.
2. Identify important facts.
3. Evaluate possible solutions internally.
4. Produce the best final answer.

Do not reveal internal reasoning.
Return only the final result.
"""

user_prompt = """
Customer Review:

"The app crashes whenever I upload a PDF larger than 20MB."

Classify the issue.

Return JSON.
"""

response = client.responses.create(
    model="gpt-4o-mini",
    input=[
        {"role":"system","content":system_prompt},
        {"role":"user","content":user_prompt}
    ]
)

print(response.output_text)