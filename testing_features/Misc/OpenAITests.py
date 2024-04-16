# testing different features of the OpenAI API

from dotenv import load_dotenv
import os
from openai import OpenAI


# load api key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

while True:
    print("You:")
    question = input()
    chat_response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "user",
                "content": question
            }
        ],
    )
    print("GPT:" + chat_response.choices[0].message.content)

