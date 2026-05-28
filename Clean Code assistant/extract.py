import pdfplumber
import sys
sys.stdout.reconfigure(encoding = 'utf-8')

pages = []
with pdfplumber.open("Clean Code assistant\\docs\\clean_code.pdf") as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        pages.append(text)
    pdf.close()
    
with open("Clean Code assistant\\docs\\clean_code_text.txt",'w',encoding='utf-8') as file:
    for p in pages:
        file.write(p)
        file.write("\n")
    file.close()
    