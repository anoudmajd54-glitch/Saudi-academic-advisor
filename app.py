import os
import streamlit as st
from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_classic.prompts import PromptTemplate
from langchain_classic.output_parsers import PydanticOutputParser

st.set_page_config(page_title="My Saudi Agent", page_icon="🎓")
st.title("🎓 My Saudi Academic Advisor")
st.write("An academic advisor that recommends 3 study tips for a Saudi university student.")

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_BASE_URL"] = "https://api.groq.com/openai/v1"
os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"

llm = ChatOpenAI(model="openai/gpt-oss-120b", temperature=0)

system_prompt = PromptTemplate(
    input_variables=["request"],
    template="""# Role: You are a helpful academic advisor for Saudi university students.
Audience: Saudi university students.
Task: Provide 3 clear study tips based on the student's request.
Think step-by-step, then formulate the response.
Constraints: Do not invent fake universities; keep recommendations practical.
Return ONLY JSON like: {{"summary": "...", "items": ["tip 1", "tip 2", "tip 3"]}}

Request: {request}""",
)

class Result(BaseModel):
    summary: str = Field(description="short summary")
    items: List[str] = Field(description="list of recommendations")

parser = PydanticOutputParser(pydantic_object=Result)
chain = system_prompt | llm | parser

request = st.text_area("Your request", placeholder="e.g., Give me 3 study tips for statistics exams")

if st.button("Get Study Tips", type="primary"):
    if not request.strip():
        st.warning("Please type a request first.")
    else:
        with st.spinner("Thinking..."):
            try:
                res = chain.invoke({"request": request})
                st.success(res.summary)
                for i, tip in enumerate(res.items, 1):
                    st.markdown(f"**{i}.** {tip}")
            except Exception as e:
                st.error(f"Error: {e}")
