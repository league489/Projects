import json
import faiss
import sentence_transformers
index = faiss.read_index("Clean Code assistant\\docs\\clean_code_index.faiss")
metadata = ""
with open("Clean Code assistant\\docs\\chunks_metadata.json",'r',encoding='utf-8') as meta:
    metadata = json.load(meta)
    meta.close()

model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')
question = "What are the characteristics of a clean function according to Robert Martin?"
question_embedding = model.encode([question])

search_result = index.search(question_embedding,3)


indicies = list(search_result[1][0])
found_chunks  = []
for indx in indicies:
    text = metadata[indx]['text']
    found_chunks.append(text)

for c in found_chunks:
    print(c)
    print("\n\n")