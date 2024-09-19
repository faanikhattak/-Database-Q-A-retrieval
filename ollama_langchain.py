

from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import FewShotPromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX
from langchain.prompts.prompt import PromptTemplate
from few_shots import few_shots
from langchain.llms import Ollama
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

def get_few_shot_db_chain():
    db_user = "irfan"
    db_password = "1233"
    db_host = "localhost"
    db_name = "database_name"

    # Connect to MySQL database using SQLDatabase from LangChain
    db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",
                              sample_rows_in_table_info=3)
    
    # Load the Mistral model via Ollama
    model = "mistral:latest"
    llm = Ollama(model=model)

    # Initialize HuggingFace embeddings for few-shot prompting
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L12-v2')

    # Prepare texts for vectorization from the few-shot examples
    to_vectorize = [" ".join(example.values()) for example in few_shots]

    # Create a vectorstore using Chroma
    vectorstore = Chroma.from_texts(
        texts=to_vectorize,
        embedding=embeddings,
        metadatas=few_shots
    )

    # Example selector for few-shot examples using semantic similarity
    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=vectorstore,
        k=3  # Adjust the number of examples to retrieve
    )

    # Custom prompt for generating SQL queries with focus on SQL, followed by a natural language response
    mysql_prompt = """You are a MySQL expert. Given an input question, first create a syntactically correct MySQL query to run. After obtaining the SQL result, return the result of the query along with a natural language response.

    Use the following format:

    Question: {input}
    SQLQuery: Query to run with no preamble
    SQLResult: The result of the SQLQuery
    Answer: Provide the answer here based on the SQLResult.

    Do not include the natural language answer directly in the SQL query.
    """

    # Example prompt template
    example_prompt = PromptTemplate(
        input_variables=["Question", "SQLQuery", "SQLResult", "Answer"],
        template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",
    )

    # Few-shot prompt template
    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=mysql_prompt,
        suffix=PROMPT_SUFFIX,
        input_variables=["input", "table_info", "top_k"],
    )

    # Create SQLDatabaseChain with few-shot prompt
    chain = SQLDatabaseChain.from_llm(
        llm, db,
        verbose=True,
        prompt=few_shot_prompt
    )
    return chain


# Function to execute SQL query and generate natural language response
def execute_query_and_generate_answer(query, db_connection):
    try:
        # Execute the SQL query
        result = db_connection.execute(query)
        result_data = result.fetchall()

        # Generate the answer based on the result
        if result_data:
            sql_result = result_data[0][0]  # Assuming a single value
            return f"The result is {sql_result}."
        else:
            return "No result found."

    except Exception as e:
        print(f"Error executing query: {e}")
        return None


# Few-shot examples
few_shots = [
    {'Question': "How many t-shirts do we have left for Nike in XS size and white color?",
     'SQLQuery': "SELECT sum(stock_quantity) FROM t_shirts WHERE brand = 'Nike' AND color = 'White' AND size = 'XS'",
     'SQLResult': "91",
     'Answer': "The number of white-color Nike shirts you have in XS size is 91."},
    {'Question': "How much is the total price of the inventory for all S-size t-shirts?",
     'SQLQuery': "SELECT SUM(price*stock_quantity) FROM t_shirts WHERE size = 'S'",
     'SQLResult': "22292",
     'Answer': "The total price of the inventory for all S-size t-shirts is $22,292."},
    {'Question': "If we have to sell all the Levi’s T-shirts today with discounts applied. How much revenue will our store generate (post discounts)?",
     'SQLQuery': """SELECT sum(a.total_amount * ((100-COALESCE(discounts.pct_discount,0))/100)) as total_revenue
                    FROM (SELECT sum(price*stock_quantity) as total_amount, t_shirt_id 
                          FROM t_shirts WHERE brand = 'Levi' GROUP BY t_shirt_id) a
                    LEFT JOIN discounts ON a.t_shirt_id = discounts.t_shirt_id""",
     'SQLResult': "16725.4",
     'Answer': "The revenue generated post-discounts from selling all Levi's T-shirts today is $16,725.40."},
    {'Question': "How many white color Levi's shirts do I have?",
     'SQLQuery': "SELECT sum(stock_quantity) FROM t_shirts WHERE brand = 'Levi' AND color = 'White'",
     'SQLResult': "290",
     'Answer': "The number of white-color Levi's shirts you currently have is 290."},
    {'Question': "How much sales revenue will be generated if we sell all large size Nike t-shirts today after discounts?",
     'SQLQuery': """SELECT sum(a.total_amount * ((100-COALESCE(discounts.pct_discount,0))/100)) as total_revenue
                    FROM (SELECT sum(price*stock_quantity) as total_amount, t_shirt_id 
                          FROM t_shirts WHERE brand = 'Nike' AND size='L' GROUP BY t_shirt_id) a
                    LEFT JOIN discounts ON a.t_shirt_id = discounts.t_shirt_id""",
     'SQLResult': "290",
     'Answer': "The total revenue from selling all large size Nike t-shirts today after discounts is $290."}
]










