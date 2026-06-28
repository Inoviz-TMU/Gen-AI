import os
from dotenv import load_dotenv
from openai import OpenAI

#load dotenv variables
load_dotenv()

api_key=os.getenv("OPENAI_API_KEY")

#openai client
client=OpenAI(api_key=api_key)

response=client.responses.create(
    model="gpt-4o-mini",
    instructions="you are a helpful assistant , Explain things simply",
    input="give me a code to add 2 numbers and put a condition that it shuld be exceed 100"
)
print(response.output_text)
#response.OutputText

