from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from pypdf import PdfReader 
from langchain_community.llms import Ollama

from markdown_pdf import MarkdownPdf, Section

pdf = MarkdownPdf(toc_level=2)


  

pdf_files = (
    "qb for preparatory.pdf",
    "updatedQB_FD.pdf",
    "FULL STACK DEVELOPMENT Test-1 Question Bank.pdf",
    "FSD-IA1-QB-sunil.pdf",
    "example.pdf",
    "1st_Inernals_QB.pdf",
    "SEPM- Question Bank 2nd IA.pdf",
)
# creating a pdf reader object 
pdf_file_chosen = pdf_files[1]
reader = PdfReader(pdf_file_chosen)

pdf.add_section(Section(f"# Question Bank Answers\n\nanswered by Llama 3\n\n> {pdf_file_chosen}"))
  
# extract text from all pages
text = ''
for i in range(len(reader.pages)):
    text += reader.pages[i].extract_text()

# text = text[1000:2000]  # Print only the first 1000 characters for brevity

# escape the curly braces
text = text.replace("{", "{{").replace("}", "}}")

# escape JSON special characters
text = text.replace("\\", "\\\\").replace('"', '\\"')

# escape brackets (, )
text = text.replace("(", "").replace(")", "")

# replace ‘ with ' and ’ with '
text = text.replace("‘", "'").replace("’", "'")

# replace = with is
text = text.replace("=", "is")

# replace - with to
text = text.replace("CO-2", "")
text = text.replace("CO-1", "")



print("Extracted text from the document:")
print(text)

# Schema for structured response
class Question(BaseModel):
    question: str = Field(title="Question", description="Question to ask", example="What is your name?")

class Questions(BaseModel):
    questions: list[Question] = Field(title="Questions", description="List of questions", example=[{"question": "What is your name?"}])


# Prompt template
prompt = PromptTemplate.from_template(text +
    """
Human: {question}
AI: """
)

# Chain
llm = OllamaFunctions(model="llama3", format="json", base_url="http://100.99.62.103:11434")
structured_llm = llm.with_structured_output(Questions)
chain = prompt | structured_llm

alex = chain.invoke("Return the questions from the document.")
# print(alex)

## Pretty print the response

i = 1
# for q in alex.questions:
#     print(f"Q {i}: {q.question}")
#     print()
#     i += 1

# generate the answers for the questions
llm_ans = Ollama(model="llama3", base_url="http://100.99.62.103:11434")

i = 1
for q in alex.questions:
    response = llm_ans.invoke(q.question)
    print(f"Q: {q.question}")
    print(f"A: {response}")
    print()

    text = f"Q{i}: **{q.question}**\n\nA: {response}\n\n"

    pdf.add_section(Section(text, toc=False))
    i += 1

pdf.save("answers4.pdf")
