import os
from datetime import datetime
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
from llama_index.core import Settings
from agent import agent
from data_reader import query_engine
from llama_index.core.agent import ReActAgent

load_dotenv()

# model init
Settings.llm = OpenAI(model = 'gpt-4', temperature = 0.2)

response = agent.chat(f'Get details of 1 wardboy (you can find data in er roster), he/she should be available around 15:00. Return the data in key value format. For Eg. {{"name": "John Doe", "mobile": "1234567890", etc...}}')

print(response)