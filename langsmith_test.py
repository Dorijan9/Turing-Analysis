from dotenv import load_dotenv
import os

load_dotenv() # Load variables from .env file

from langchain_openai import ChatOpenAI

# Initialize the LLM
llm = ChatOpenAI()

# Invoke the LLM with a simple input
response = llm.invoke("Hello, world!")

# Print the response
print(response)