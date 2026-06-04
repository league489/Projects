import sentence_transformers
import faiss
import numpy
import json

with open("Clean Code assistant\\docs\\chunk_file.json",'r',encoding='utf-8') as chunks_file:
    chunks = json.load(chunks_file)
    chunks_file.close()
print(len(chunks))

model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')

embeddings = []
for chunk in chunks:
    embedding = model.encode(chunk["text"])
    embeddings.append(embedding)
print(len(embeddings))

dim = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dim)
embeddings_array = numpy.array(embeddings).astype('float32')
index.add(embeddings_array)
faiss.write_index(index,"Clean Code assistant\\docs\\clean_code_index.faiss")

with open("Clean Code assistant\\docs\\chunks_metadata.json",'w',encoding='utf-8') as chunk_meta:
    json.dump(chunks,chunk_meta)
    chunk_meta.close()
