import transformers
import os
import json
import faiss
import sentence_transformers
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

index = faiss.read_index("Clean Code assistant\\docs\\clean_code_index.faiss")
metadata = ""
with open("Clean Code assistant\\docs\\chunks_metadata.json",'r',encoding='utf-8') as meta:
    metadata = json.load(meta)
    meta.close()
    
model_em = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')
tokenizer = transformers.AutoTokenizer.from_pretrained("google/flan-t5-small")
model_lm = transformers.AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

# LM dry run test
# input_text = "What is code?"
# input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# outputs = model_lm.generate(input_ids,max_new_tokens = 150)
# print(tokenizer.decode(outputs[0]))

def ask(question,k=3):
    q_embedding = model_em.encode([question])
    search_result = index.search(q_embedding,k)
    indicies = list(search_result[1][0])
    found_chunks  = []
    for indx in indicies:
        text = metadata[indx]['text']
        found_chunks.append(text)
    context = "\n".join(found_chunks)
    prompt = f"Answer based only on context below. If the answer is not in the context, say 'I don't know'.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model_lm.generate(inputs.input_ids, max_new_tokens=150, temperature=0.2, do_sample=False)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

print(ask("What does Bjarne Stroustrup say about clean code?",2))