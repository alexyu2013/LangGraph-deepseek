!pip install openai langchain yfinance pandas numpy streamlit langgraph langchain-community -i 

import os
import re
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI  # 使用 OpenAI 的 Chat 模型
from langgraph.graph import StateGraph, END
from typing import Dict, TypedDict
from langchain.schema.runnable import RunnableLambda

# Define state
class AgentState(TypedDict):
    question: str
    facts: list
    conclusion: str

# 直接在代码中定义 OpenAI API 密钥和 API 地址
OPENAI_API_KEY = "sk-YbovpUeEi3FsJaNs990d77D515D445278a2187Ff5232160b"  # 替换为你的 OpenAI API 密钥
OPENAI_API_BASE = "https://api.laoyulaoyu.top/v1"  # 替换为你的自定义 API 地址

# 初始化 OpenAI 模型
def get_llm():
    return ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model_name='deepseek-v3',  # 使用 OpenAI 模型
        openai_api_base=OPENAI_API_BASE  # 指定 API 地址
    )

llm = get_llm()

def extract_ticker(question: str) -> str:
    match = re.search(r'\b[A-Z]{1,5}\b', question)
    return match.group(0) if match else None

def compute_technical_indicators(stock_data: pd.DataFrame) -> Dict[str, str]:
    if stock_data.empty:
        return {"Error": "No price data available"}
    
    # 计算 50 日和 200 日移动平均线
    stock_data['MA_50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['MA_200'] = stock_data['Close'].rolling(window=200).mean()
    
    # 计算每日收益率
    stock_data['Daily Return'] = stock_data['Close'].pct_change()
    
    # 计算波动率（30 日标准差）
    stock_data['Volatility'] = stock_data['Daily Return'].rolling(window=30).std()
    
    # 计算 RSI（相对强弱指数）
    delta = stock_data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    stock_data['RSI'] = 100 - (100 / (1 + rs))
    
    return {
        "50-day MA": f"${stock_data['MA_50'].iloc[-1]:.2f}",
        "200-day MA": f"${stock_data['MA_200'].iloc[-1]:.2f}",
        "Volatility": f"{stock_data['Volatility'].iloc[-1]:.4f}",
        "RSI": f"{stock_data['RSI'].iloc[-1]:.2f}"
    }

def gather_facts(state: AgentState) -> AgentState:
    ticker_symbol = extract_ticker(state["question"])
    if not ticker_symbol:
        state["facts"].append("Error: Could not extract ticker from question.")
        return state
    
    stock = yf.Ticker(ticker_symbol)
    price_data = stock.history(period="1y")
    fundamentals = stock.info or {}
    
    facts = [
        f"Latest Closing Price: ${price_data['Close'].iloc[-1]:.2f}",
        f"P/E Ratio: {fundamentals.get('trailingPE', 'N/A')}",
        f"Market Cap: {fundamentals.get('marketCap', 'N/A')}",
        f"Revenue: {fundamentals.get('totalRevenue', 'N/A')}",
        f"Debt-to-Equity Ratio: {fundamentals.get('debtToEquity', 'N/A')}",
    ]
    facts += [f"{key}: {value}" for key, value in compute_technical_indicators(price_data).items()]
    
    state["facts"].append("\n".join(facts))
    return state

def draw_conclusion(state: AgentState) -> AgentState:
    facts = "\n".join(state["facts"])
    prompt = PromptTemplate(template="""Based on these financial facts:
    {facts}
    Provide a structured analysis of the stock's potential movement:
    - **Trend Analysis**
    - **Market Position**
    - **Potential Risks**
    """, input_variables=["facts"])
    
    chain = prompt | llm
    try:
        response = chain.invoke({"facts": facts})
        # Extract only the content from the response
        if hasattr(response, 'content'):
            state["conclusion"] = response.content
        else:
            state["conclusion"] = str(response)
    except Exception as e:
        state["conclusion"] = f"Error in conclusion generation: {str(e)}"
    
    return state

workflow = StateGraph(AgentState)
workflow.add_node("gather_facts", gather_facts)
workflow.add_node("draw_conclusion", draw_conclusion)
workflow.add_edge("gather_facts", "draw_conclusion")
workflow.add_edge("draw_conclusion", END)
workflow.set_entry_point("gather_facts")
graph = workflow.compile()

# Streamlit UI
st.title("Stock Market Analysis")
question = st.text_input("Enter a stock symbol:")
if st.button("Analyze"):
    initial_state = {"question": question, "facts": [], "conclusion": ""}
    result = graph.invoke(initial_state)
    st.subheader("Gathered Facts:")
    for fact in result["facts"]:
        st.text(fact)
    st.subheader("Final Conclusion:")
    st.markdown(result["conclusion"])
