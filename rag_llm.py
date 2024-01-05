from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import HuggingFaceHub, OpenAI

class AskYoutubeVideo:
    def __init__(
        self,
        chunk_size=1000, 
        chunk_overlap=100, 
        embedding_model="hkunlp/instructor-large", 
        llm_repo_id="google/flan-t5-large",
        normalize_embeddings=True,
        device='cuda',
        max_length=512,
        temperature=0.05
        ):

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        model_kwargs = {'device': device}
        encode_kwargs = {'normalize_embeddings': normalize_embeddings}
        self.embeddings = HuggingFaceInstructEmbeddings(
            model_name=embedding_model,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        self.db = None
        llm = HuggingFaceHub(
            repo_id=llm_repo_id, 
            model_kwargs={"temperature": temperature, "max_length": max_length}
            )
        prompt = PromptTemplate(
            input_variables=["question", "docs"],
            template="""
            You are a helpful Youtube assistanct that can answer questions about videos based on the video's transcript
                    
            Answer the following question: {question}
            By searching the following video transcript: {docs}

            Only use the factual information from the transcipt to answer the question.

            If you feel like you do not have enough information to answer the question, say "I don't know".

            Your answers should be detailed.
            """)
        self.chain = LLMChain(llm=llm, prompt=prompt)
        
    def get_docs(self, video_url):
        loader = YoutubeLoader.from_youtube_url(video_url)
        transcript = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
            )
        docs = text_splitter.split_documents(transcript)
        return docs

    def embed_docs(self, docs):
        self.db = FAISS.from_documents(docs, self.embeddings)
    
    def query_docs(self, question, k):
        if self.db is None:
            raise "Please embed_docs before ask a question"
        docs = self.db.similarity_search(question, k=k)
        docs_page_content = " ".join([d.page_content for d in docs])
        return docs_page_content

    def generate_answer(self, docs_page_content, question):
        response = self.chain.run(question=question, docs=docs_page_content)
        answer = response.replace("\n", "")
        return answer
    
    def ask(self, question, k=5):
        docs = self.query_docs(question, k)
        return self.generate_answer(question, docs)
    
# https://www.promptingguide.ai/techniques/rag
yt = AskYoutubeVideo()
docs = yt.get_docs("https://www.youtube.com/watch?v=Jr8gLJr9WKQ&ab_channel=EnglishSkillsMastery")
yt.embed_docs(docs)

question = "What is teacher's name?"
answer = yt.ask(question=question, k=5)
print(f"Question: {question}, Answer: {answer}")

