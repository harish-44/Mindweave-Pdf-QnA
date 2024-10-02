from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from pypdf import PdfReader
  
# creating a pdf reader object 
reader = PdfReader('DSV-IA--QB1_M1-M2.pdf')
  
# extract text from all pages
text = ''
for i in range(len(reader.pages)):
    page_text = reader.pages[i].extract_text()
    if page_text:
        text += page_text + "\n"

print("Extracted text from the document:")
print(text[:1000])  # Print only the first 1000 characters for brevity

# Schema for structured response
class Question(BaseModel):
    question: str = Field(title="Question", description="Question to ask", example="What is your name?")

class Questions(BaseModel):
    questions: list[Question] = Field(title="Questions", description="List of questions", example=[{"question": "What is your name?"}])

# Prompt template
template_text = text + """
Human: {question}
AI: """

prompt = PromptTemplate.from_template(template_text)

# Chain
llm = OllamaFunctions(model="llama3", format="json", temperature=0, base_url="http://100.99.62.103:11434")
structured_llm = llm.with_structured_output(Questions)
chain = prompt | structured_llm

question_to_ask = "Return the questions from the document."

try:
    response = chain.invoke({"question": question_to_ask})
    if response and response.questions:
        for i, q in enumerate(response.questions, 1):
            print(f"Q {i}: {q.question}")
    else:
        print("No questions found in the response.")
except ValueError as e:
    print(f"ValueError: {e}")
    print("Ensure the AI's response format is correct and includes 'tool_calls'.")
except Exception as e:
    print(f"Unexpected error: {e}")
    