import streamlit as st
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
import textwrap

# Initialize LLM
llm = OllamaLLM(model="deepseek-r1:1.5b")

# Define the prompt
pre_prompt = """
You are an obedient assistant that fulfills text-based requests and answers questions.
You always use the first person or impersonal tone and never use the second person in your responses.
1. If available, use the context to answer the question at the end.
2. If you don't know the answer, just say that "I don't know" but don't make up an answer on your own.
3. Keep the answer concise and specific.
Context: {context}
Prompt: {prompt}
Answer:
"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(pre_prompt)

# Streamlit UI
st.title("Simple QA Interface for DeepSeek")
st.write("Enter context and a prompt to receive a thought and response.")
# User inputs
context = st.text_area("Context", "Baby don't hurt me.", height=100)
prompt = st.text_area("Prompt", "What is love?", height=None)

if st.button("Submit"):
    # Generate response
    llm_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT, callbacks=None, verbose=True)
    response = llm_chain.run(context=context, prompt=prompt)
    
    # Extract thought and answer
    think_part = response[response.find("<think>")+7:response.find("</think>")].strip()
    answer_part = response[response.find("</think>")+8:].strip()
    
    # Wrap text for better readability
    wrapper = textwrap.TextWrapper(width=100)
    wrapped_thought = wrapper.fill(think_part)
    wrapped_response = wrapper.fill(answer_part)
    
    # Display chat response
    st.subheader("Thinking...")
    st.write(wrapped_thought)
    
    st.subheader("Response:")
    st.write(wrapped_response)
