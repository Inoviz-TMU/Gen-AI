import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

system_prompt = """
You are an expert network engineer.

Analyze each log carefully before answering.explain me briefly how to fix it step by step .

Return ONLY JSON.
"""

user_prompt = """
Example 1

Input:
CPU utilization crossed 95%.

recommendations_logs_to_check :5 apps running , and batch job running that consumes 90% of ram , and google chrome is opened with multipe tabs .

Output:
{
 "event":"CPU_SPIKE",
 "severity":"HIGH",
 "source":"SERVER"
}

Example 2

Input:
Memory usage reached 92%.
Explaination : 24 chrome tabs opned and a heavy applicaiton agents.py runnign in background .
Output:
{
 "event":"MEMORY_SPIKE",
 "severity":"HIGH",
 "source":"SERVER"
}

Now classify:

Input:
Disk latency exceeded 500ms.give me 2-3 recommnedation steps .
"""

response = client.responses.create(
    model="gpt-5-mini",
    input=[
        {"role":"system","content":system_prompt},
        {"role":"user","content":user_prompt}
    ]
)

print(response.output_text) 