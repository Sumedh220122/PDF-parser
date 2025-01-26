from langchain_mistralai.chat_models import ChatMistralAI
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from schema import SchemaReviews


loader = PyPDFLoader("Sumedh_cloud.pdf")
pages = loader.load_and_split()

text = " ".join(list(map(lambda page: page.page_content, pages)))

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key = 'AIzaSyCaQBZEg1ho_NqVezEMWHlKK9Uc-dXmFI4'
)

structured_llm = llm.with_structured_output(SchemaReviews)

response = structured_llm.invoke(text)

print(response.content)
