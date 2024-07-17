import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
task = "text-generation"

st.set_page_config(page_title="Arth Mitra.AI", page_icon="💰")
st.title("Arth Mitra.AI 💸")

# Define the template for the financial advisor
financial_template = """
You are an expert financial advisor. Here are the financial details provided by the user:

1. Monthly Income: {income} INR
2. Monthly Expenses: {expenses} INR
3. Current Savings: {savings} INR
4. Current Investments: {investments} INR
5. Risk Tolerance: {risk_tolerance}
6. Financial Goals and Question: {financial_goals}

Provide a detailed analysis and advice based on the provided information. Consider investment options like Mutual Funds, Stocks, Real Estate, PPF, NPS, Sukanya Samriddhi Yojana, and others.
"""

prompt = ChatPromptTemplate.from_template(financial_template)

# Function to get a response from the model
def get_response(financial_data):
    # Initialize the Hugging Face Endpoint
    llm = HuggingFaceEndpoint(
        huggingfacehub_api_token=api_token,
        repo_id=repo_id,
        task=task
    )

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({
        "income": financial_data['income'],
        "expenses": financial_data['expenses'],
        "savings": financial_data['savings'],
        "investments": financial_data['investments'],
        "risk_tolerance": financial_data['risk_tolerance'],
        "financial_goals": financial_data['financial_goals'],
    })

    return response

# Budget Planner Interface
st.header("Budget Planner")

income = st.number_input("Monthly Income (INR):", min_value=0, step=1000, format='%d')
expenses = st.number_input("Monthly Expenses (INR):", min_value=0, step=1000, format='%d')
savings = st.number_input("Current Savings (INR):", min_value=0, step=1000, format='%d')
investments = st.number_input("Current Investments (INR):", min_value=0, step=1000, format='%d')
risk_tolerance = st.selectbox("Risk Tolerance:", ["Low", "Medium", "High"])
financial_goals = st.text_area("Financial Goals and Question:")

financial_data = {
    "income": income,
    "expenses": expenses,
    "savings": savings,
    "investments": investments,
    "risk_tolerance": risk_tolerance,
    "financial_goals": financial_goals
}

if st.button("Submit Financial Data"):
    if all(financial_data.values()):
        st.success("Financial data submitted successfully. Analyzing your financial goals...")

        # Get response based on financial data
        response = get_response(financial_data)

        # Display the financial analysis
        st.header("Financial Analysis")
        st.write(response)

    else:
        st.error("Please fill out all fields with valid values.")
