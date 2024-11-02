from flask import render_template , Flask , jsonify , request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
os.environ['GROQ_API_KEY'] = GROQ_API_KEY

embeddings = download_hugging_face_embeddings()
index_name = "medibot"
docserach = PineconeVectorStore.from_existing_index(
    index_name = index_name,
    embedding = embeddings
)
retriever = docserach.as_retriever(search_type = "similarity" , search_kwargs = {"k" : 7})
llm = ChatGroq( model_name = "Llama3-8b-8192",
                temperature = 0.4)

human_prompt = "{input}"

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", human_prompt)
])

question_answer_chain = create_stuff_documents_chain(llm , prompt)
rag_chain = create_retrieval_chain(retriever , question_answer_chain)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get" , methods = ["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response: " + response['answer'])
    return str(response['answer'])

if __name__ == "__main__":
    app.run(host = "0.0.0.0" , port = "8080" , debug = True)