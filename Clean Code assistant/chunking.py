import sys
import json
sys.stdout.reconfigure(encoding = 'utf-8')

file = ""
with open("Clean Code assistant\\docs\\clean_code_text.txt",'r',encoding='utf-8') as clean:
    file = clean.read()
    clean.close()
 
chunk_s = 1000
overlap_s = 100
step_s = chunk_s - overlap_s

    
def chunk_text(text_file,chunk_size,overlap):
   chunks = []
   step = chunk_size - overlap
   for i in range(0,len(text_file),step):
        chunk = text_file[i:i+chunk_size]
        chunks.append({"id":len(chunks), "text":chunk})
   return chunks    
    
chunkss = chunk_text(file,chunk_s,overlap_s)
print(len(chunkss))
with open("Clean Code assistant\\docs\\chunk_file.json",'w',encoding='utf-8') as chunk_file:
    json.dump(chunkss,chunk_file)
    chunk_file.close()