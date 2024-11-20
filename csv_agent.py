import os
import traceback
from dotenv import load_dotenv
import pandas as pd
from langchain_anthropic import ChatAnthropic
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# Load environment variables
load_dotenv()

# Retrieve the API key
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
if not anthropic_api_key:
    raise ValueError("Please set the ANTHROPIC_API_KEY environment variable.")

# Initialize the model with ChatAnthropic
try:
    model = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        temperature=0,
        max_tokens=1000,
        api_key=anthropic_api_key,
    )
except Exception as e:
    print("Error initializing ChatAnthropic model:")
    traceback.print_exc()
    raise

# read csv file, if empty space found fill with 0
try:
    df = pd.read_csv('./csv-data/online_store_products.csv').fillna(value=0) 
    # print("CSV loaded successfully.")
except FileNotFoundError:
    print("Error: The file './data/salaries_2023.csv' was not found.")
    print("Current working directory:", os.getcwd())
    raise
except pd.errors.EmptyDataError:
    raise ValueError("The file './data/salaries_2023.csv' is empty.")
except pd.errors.ParserError:
    raise ValueError("The file './data/salaries_2023.csv' contains invalid data.")

# used a "agent" variable to create the pandas_dataframe_agent
try:
    agent = create_pandas_dataframe_agent(
        llm=model,
        df=df,
        verbose=True,
        allow_dangerous_code=True# Enable execution of arbitrary code
    )
except Exception as e:
    print("Error creating pandas dataframe agent:")
    traceback.print_exc()
    raise

#call the agent with our query question.
try:
    res = agent.invoke("What is the total sales revenue for each product category")
except Exception as e:
    print("Error running the agent:")
    traceback.print_exc()
    raise
    # print(res)