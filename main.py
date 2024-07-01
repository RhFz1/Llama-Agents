import os
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage, Settings
load_dotenv()

# model init

Settings.llm = OpenAI(temperature=0.2, model='gpt-4')

# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir=os.environ.get('store_path'))

# load index from storage
index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine()
response = query_engine.query("Which male MBBS doctor is free around 16:00")
print(response)