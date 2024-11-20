import os
import traceback
from dotenv import load_dotenv
import pandas as pd
from langchain_anthropic import ChatAnthropic

# Load environment variables
load_dotenv()

# Retrieve the API key
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

if not anthropic_api_key:
    raise ValueError("Please set the ANTHROPIC_API_KEY environment variable.")

# Initialize the model with ChatAnthropic
model = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    temperature=0,
    max_tokens=1000,
    api_key=anthropic_api_key,
)

# read csv file, if empty space found fill with 0
df = pd.read_csv('./csv-data/online_store_products.csv').fillna(value=0) 

print(df.head())

