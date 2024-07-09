from datetime import datetime
from dotenv import load_dotenv
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from data_reader import roster_query_tool
from llama_index.core import Settings

load_dotenv()

def get_current_time():
    """Returns the current time in HH:MM:SS format."""
    return datetime.now().strftime("%H:%M:%S")

def compare_time(time1, time2):
    """Returns if time1 is greater than time2."""
    time1 = datetime.strptime(time1, "%H:%M:%S")
    time2 = datetime.strptime(time2, "%H:%M:%S")
    return time1 > time2

compare_time_tool = FunctionTool.from_defaults(fn=compare_time)
get_time_tool = FunctionTool.from_defaults(fn=get_current_time)

agent = ReActAgent.from_tools(tools=[get_time_tool,compare_time_tool ,roster_query_tool], verbose=True)