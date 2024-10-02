from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from pypdf import PdfReader 
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.utils import filter_complex_metadata
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

from markdown_pdf import MarkdownPdf, Section

pdf = MarkdownPdf(toc_level=2)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)


def ingest_pdf(pdf_file_path):
    global vector_store, retriever
    docs = PyPDFLoader(file_path=pdf_file_path).load()
    chunks = text_splitter.split_documents(docs)
    chunks = filter_complex_metadata(chunks)

    vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 3,
            "score_threshold": 0.1,
        },
    )

refer_to_pdf = "textbook.pdf"
ingest_pdf(refer_to_pdf)
  

pdf_files = (
    "qb for preparatory.pdf",
    "updatedQB_FD.pdf",
    "FULL STACK DEVELOPMENT Test-1 Question Bank.pdf",
    "FSD-IA1-QB-sunil.pdf",
    "example.pdf",
    "1st_Inernals_QB.pdf",
    "FULL STACK DEVELOPMENT Test-2 Question Bank.pdf"
)
# creating a pdf reader object 
pdf_file_chosen = pdf_files[1]
reader = PdfReader(pdf_file_chosen)

pdf.add_section(Section(f"# Question Bank Answers\n\nanswered by Llama 3\n\n> {pdf_file_chosen}\n\nUsing RAG"))
  
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

prompt = PromptTemplate.from_template(
    """
    You are given a question and you have to answer it as a responsible AI. Use the information from the context to answer the question.
    Question: {question}
    Context: {context}
    Answer: """
)

answer_chain = ({"context": retriever, "question": RunnablePassthrough()}| prompt | llm_ans | StrOutputParser())

i = 1
for q in alex.questions:
    answer = answer_chain.invoke({"question": q.question})
    text = f"Q {i}: {q.question}\n\nA: {answer}\n\n"
    print(text)

    pdf.add_section(Section(text, toc=False))
    i += 1

pdf.save("answers_fd_ia_2_qb.pdf")
