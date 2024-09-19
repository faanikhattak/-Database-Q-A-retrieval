import streamlit as st
from ollama_langchain import get_few_shot_db_chain

st.title("ollama_mistral T Shirts: Database Q&A 👕")
st.header("rebuilt by irfan ")

question = st.text_input("Question: ")

if question:
    chain = get_few_shot_db_chain()
    if chain:
        response = chain.run(question)
        st.header("Answer")
        st.write(response)
    # else:
    #     st.error("Failed to initialize the chain. Please check the configuration.")






















# import streamlit as st
# from langchain_helper import get_few_shot_db_chain


# st.title("AtliQ T Shirts: Database Q&A 👕")

# question = st.text_input("Question: ")

# if question:
#     chain = get_few_shot_db_chain()
#     response = chain.run(question)

#     st.header("Answer")
#     st.write(response)