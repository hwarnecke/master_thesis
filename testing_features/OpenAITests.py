# testing different features of the OpenAI API

from dotenv import load_dotenv
import os
from openai import OpenAI


# load api key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

chate_response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant."
        }
    ],
)

print(chate_response.choices[0].message.content)
