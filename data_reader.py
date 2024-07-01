import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core import VectorStoreIndex
load_dotenv()

# Here this api basically reads the data from the general roster data and stores it in document formats.
documents = SimpleDirectoryReader(os.path.join(os.environ.get('data_path'),"general_roster_data")).load_data()

# Here we define a pipeline for the ingestion of the data.
pipeline = IngestionPipeline(transformations=[TokenTextSplitter()])

# Each document is converted into a node, here after applying a transformation.
# Node here can be considered as a bite sized patches of data that can be fed into the LLM.
nodes = pipeline.run(documents=documents)

# Here we create a vector store index, which maps the nodes to a embedding space.
index = VectorStoreIndex(nodes=nodes)

# Here we save the index stores to a persistent storage.
index.storage_context.persist(os.path.join(os.environ.get('data_path'),"index_store"))