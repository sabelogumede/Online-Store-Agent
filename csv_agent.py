import os
import traceback
from dotenv import load_dotenv
import pandas as pd
from langchain_anthropic import ChatAnthropic
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# Load environment variables
load_dotenv()

# add pre and sufix prompt
CSV_PROMPT_PREFIX = """ 
1. Begin by setting the pandas display options to show all columns clearly.
2. Retrieve and list the column names from the DataFrame.
3. Analyze the data relevant to the question provided by the user.
4. Formulate a response based on your analysis.
"""

CSV_PROMPT_SUFFIX = """
- **IMPORTANT**: Before providing your Final Answer, explore at least two distinct methods to approach the question.
- Reflect on the results obtained from each method. Ask yourself if both methods adequately address the original question.
- If there is uncertainty, try a different method or approach until you reach consistent results from at least two methods.
- Ensure that all numerical results are formatted with four or more decimal places, using commas for thousands (e.g., 1,234.56).
- If the methods yield different results after multiple attempts, acknowledge that you are unsure of the answer.
- If you are confident in your final answer, present it in a well-structured Markdown format that is clear and visually appealing.
- **CRUCIAL**: Do not fabricate answers or rely on prior knowledge; base your response solely on the results of your calculations.
- **FINAL ANSWER**: Include a section titled "\n\nExplanation:\n" where you detail how you arrived at your conclusion. In this explanation, specify which column names were used in your calculations and any relevant insights derived from them.
"""

QUESTION = "Which products are at risk of running out of stock soon based on sales velocity and stock quantity?"

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
    # res = agent.invoke("What is the total sales revenue for each product category")
    res = agent.invoke({"input": CSV_PROMPT_PREFIX + QUESTION + CSV_PROMPT_SUFFIX})
except Exception as e:
    print("Error running the agent:")
    traceback.print_exc()
    raise

print(f"Final result: {res["output"]}")