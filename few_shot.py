import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

system_prompt = (
    "You are a log classification parser. "
    "Output only the JSON object — no preamble, no explanation."
)

few_shot_prompt = """
Input: Firewall blocked 450 repeated requests from 192.168.1.50.
Output: {"event": "DOS_BLOCK", "severity": "HIGH", "source": "FIREWALL"}

Input: Broker node disk utilization crossed 89% threshold.
Output: {"event": "DISK_SURGE", "severity": "MEDIUM", "source": "OS_BROKER"}

Input: Gateway logged a routine SSL handshake renegotiation.
Output:
"""

response = client.responses.create(
    model="gpt-4o-mini",
    input=[
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": few_shot_prompt,
        },
    ],
)

print(response.output_text)