#import statements
import os
import tempfile
import whisper
from yt_dlp import YoutubeDL
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
from operator import itemgetter
import pandas as pd
import pytorch
def generateRAG(url, user_question):
    #loading api keys and urls
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    YOUTUBE_VIDEO = url

    #laoding model and intializing parser
    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
    parser = StrOutputParser()

    #template and prompt engineering -> Q&A format
    template = """
    Answer the question based on the context below. If you can't answer the question, reply "I don't know".
    Context: {context} 
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    #downloading transcripts from URLs
    if not os.path.exists("transcription.txt"):
        whisper_model = whisper.load_model("base")

        with tempfile.TemporaryDirectory() as tmpdir:
            options = {
                "format": "bestaudio/best",
                "outtmpl": f"{tmpdir}/%(title)s.%(ext)s",
            }
            with YoutubeDL(options) as ydl:
                info = ydl.extract_info(YOUTUBE_VIDEO, download=True)
                audio_file = ydl.prepare_filename(info)

            transcription = whisper_model.transcribe(audio_file, fp16=False)["text"].strip()

            with open("transcription.txt", "w") as file:
                file.write(transcription)

    #chekcing transcript file
    with open("transcription.txt") as file:
        transcription = file.read()
    transcription[:100]

    #loading transcripts into documents
    loader = TextLoader("transcription.txt")
    text_documents = loader.load()

    #intializing embeddings
    embeddings = OpenAIEmbeddings()

    #document splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    documents = text_splitter.split_documents(text_documents)

    #pincone intialization
    index_name = "projectrag"
    pinecone = PineconeVectorStore.from_documents(
        documents, embeddings, index_name=index_name
    )

    #chain creation
    chain = (
        {"context": pinecone.as_retriever(), "question": RunnablePassthrough()} | prompt | model | parser
    )

    #testing Q&A
    result = chain.invoke(user_question)
    print(result)


generateRAG("https://www.youtube.com/watch?v=DxREm3s1scA", "What type of programming platforms does musk use?")
