import os
from dotenv import load_dotenv
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, SimpleDirectoryReader
from llama_index.core.tools import QueryEngineTool
load_dotenv()

# Here this api basically reads the data from the general roster data and stores it in document formats.
documents = None

# Adjusting the read piple
for direc in os.listdir(os.path.join(os.environ.get('data_path'))):
    print(direc)
    if direc == "index_store":
        continue
    if documents is None:
        documents = SimpleDirectoryReader(os.path.join(os.environ.get('data_path'),direc)).load_data()
    else:
        documents += SimpleDirectoryReader(os.path.join(os.environ.get('data_path'),direc)).load_data()

# Here we define a pipeline for the ingestion of the data.
pipeline = IngestionPipeline(transformations=[TokenTextSplitter()])

# Each document is converted into a node, here after applying a transformation.
# Node here can be considered as a bite sized patches of data that can be fed into the LLM.
nodes = pipeline.run(documents=documents)

# Here we create a vector store index, which maps the nodes to a embedding space.
index = VectorStoreIndex(nodes=nodes)

# # Here we save the index stores to a persistent storage.
# index.storage_context.persist(os.path.join(os.environ.get('data_path'),"index_store"))

# # rebuild storage context
# storage_context = StorageContext.from_defaults(persist_dir=os.environ.get('store_path'))

# # load index from storage
# index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine()

roster_query_tool = QueryEngineTool.from_defaults(query_engine=query_engine,
                                                  name="Query Tool",
                                                  description="This is a tool which can be used when you want to fetch data from the roster.")