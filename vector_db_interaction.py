import os
from dotenv import find_dotenv, load_dotenv # for api key
from langchain.llms import HuggingFaceHub
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
import textwrap

print("load our api keys")
load_dotenv(find_dotenv())

print("load hong kong university nlp instructor embeddings... 'cuda' for gpu was changed to 'cpu'")
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", model_kwargs={"device": "cpu"}) 

print("let's load the Chroma vector DB that was already created from disk.")
db3 = Chroma(persist_directory="./db", embedding_function=instructor_embeddings)
docs = db3.similarity_search("What's up with Digital Physics?")
print("this is the page_content of the vector most closely matching 'what's up with digital physics?': ", docs[0].page_content)

print("Parameterize the vecotr retreiver, the vector db nearest neighbor algorithm should retrieve top 5.")
retriever = db3.as_retriever(search_kwargs={"k": 5})

print("We're going to use a google opensource llm model from huggingface hub. ChatGPT or another one we can pay for requests from via API keys might be worth it.")
repo_id="google/flan-T5-large"
# llm = TogetherLLM(model= "togethercomputer/llama-2-70b-chat", temperature = 0.1, max_tokens = 1024)
hub_llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.8, "max_length": 128})

print("Now we'll create the chain to answer questions. Params include the LLM, the retriever, we're going to stuff it into the context window length, the retriever, and source documents too.")
qa_chain = RetrievalQA.from_chain_type(llm=hub_llm, #llm,
                                  chain_type="stuff",
                                  retriever=retriever,
                                  return_source_documents=True)

## Cite sources
def wrap_text_preserve_newlines(text, width=110):
    print("wrap text & preserve newline function called...")
    print("Split the input text into lines based on newline characters")
    lines = text.split('\n')

    print("Wrap each line individually")
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    print("Join the wrapped lines back together using newline characters")
    wrapped_text = '\n'.join(wrapped_lines)

    print("return wrapped text")
    return wrapped_text

def process_llm_response(llm_response):
    print("process the llm response called...")
    print(wrap_text_preserve_newlines(llm_response['result']))
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])

print("let's do some tests...")
query = "What is the meaning of Digital Physics?"
llm_response = qa_chain(query)
process_llm_response(llm_response)

# break it down
query = "Who is Marcel?"
llm_response = qa_chain(query)
process_llm_response(llm_response)

print("qa_chain retreiver search type & vector store: ")
print(qa_chain.retriever.search_type , qa_chain.retriever.vectorstore)
print("qa_chain chain template: ")
print(qa_chain.combine_documents_chain.llm_chain.prompt.template)