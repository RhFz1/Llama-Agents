import os
from datetime import datetime
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
from llama_index.core import Settings
from agent import agent
from data_reader import query_engine, roster_query_tool
from llama_index.core.agent import ReActAgent

load_dotenv()

# model init
Settings.llm = OpenAI(model = 'gpt-4', temperature = 0.2)

response = query_engine.query("Fetch the checklist for the nurse, and show it in step by step format.")
print(response)