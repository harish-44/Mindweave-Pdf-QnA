from pypdf import PdfReader 
  
# creating a pdf reader object 
reader = PdfReader('example.pdf') 
  
# extract text from all pages
text = ''
for i in range(len(reader.pages)):
    text += reader.pages[i].extract_text()

print(text)