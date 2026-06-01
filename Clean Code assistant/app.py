import streamlit as st
import transformers
import os
import json
import faiss
import sentence_transformers
import assemble

@st.cache_resource
def load_models():
    index = faiss.read_index("Clean Code assistant\\docs\\clean_code_index.faiss")
    metadata = ""
    with open("Clean Code assistant\\docs\\chunks_metadata.json",'r',encoding='utf-8') as meta:
        metadata = json.load(meta)
        meta.close()
    model_em = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')
    tokenizer = transformers.AutoTokenizer.from_pretrained("google/flan-t5-small")
    model_lm = transformers.AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    return index, metadata, model_em, tokenizer, model_lm

index, metadata, model_em, tokenizer, model_lm = load_models()
st.set_page_config(page_title="Clean Code Assistant",layout = "centered")
st.title("Clean Code Assistant")
st.text_input("Question",key="question_input")
if st.button("Ask"):
    if len(st.session_state["question_input"])>3:
        result = assemble.ask(st.session_state["question_input"],2)
        st.write("Answer:")
        st.write(result)
    else:
        st.write("Please enter a valid question")    