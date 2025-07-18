## **LLM 준비**
### **필요한 라이브러리 다운로드 받기**
"""

!pip install chromadb
!pip install langchain-community
!pip -q install langchain pypdf chromadb sentence-transformers faiss-gpu
!pip install faiss-gpu
!pip install langchain
!pip install openai
!pip install tiktoken
!pip install langchain_openai
!pip install -U langchain langchain_experimental -q

# 필요한 라이브러리 설치
from langchain.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceEndpoint
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain_community.document_loaders import CSVLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv

import os
import warnings
import langchain_openai

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_openai import OpenAIEmbeddings

warnings.filterwarnings("ignore")

"""### **데이터 준비**"""

# RAG 활용을 위한 데이터 전처리

# 소수점 이하 2자리 제거
ETF = ETF.round(2)

# tabular 데이터 형태를 자연어 텍스트 변환
def prepare_data_for_rag(df):
    df['text'] = (
        "ETF 이름은 " + df['ETF'].astype(str) + "입니다, " +
        df['ETF'].astype(str) + " 종목 종가는 " + df['종목종가'].astype(str) + "입니다. " +
        df['ETF'].astype(str) + " 단기 수익은 " + df['단기수익'].astype(str) + "입니다. " +
        df['ETF'].astype(str) + " 장기 수익은 " + df['장기수익'].astype(str) + "입니다. " +
        df['ETF'].astype(str) + " 변동성은 " + df['변동성'].astype(str) + "입니다. " +
        df['ETF'].astype(str) + " 유동성은 " + df['유동성'].astype(str) + "입니다. " +
        df['ETF'].astype(str) + " 관심도은 " + df['관심도'].astype(str) + "입니다. " +
        df['ETF'].astype(str) + " 1개월 수익률은 " + df['1개월_수익율'].astype(str) + "입니다. " +
        df['ETF'].astype(str) + " 3개월 수익률은 " + df['3개월_수익율'].astype(str) + "입니다. " +
        df['ETF'].astype(str) +  "MACD은 " + df['MACD'].astype(str) + "입니다. " +
        df['ETF'].astype(str) + " 볼린저 갭은 " + df['볼린저갭'].astype(str) + "입니다. " +
        df['ETF'].astype(str) + " 거래 변동성은 " + df['거래_변동성'].astype(str) + "입니다. " +
        df['ETF'].astype(str) + " 총 거래 금액은 " + df['총거래금액'].astype(str) + "입니다. " +
        df['ETF'].astype(str) + " 매수매도 차이 비율은 " + df['매수매도_차이비율'].astype(str) + "입니다. " +
        df['ETF'].astype(str) + " 유입 금액 분산은 " + df['유입금액_분산'].astype(str) + "입니다. " +
        df['ETF'].astype(str) + " 유출 금액 분산은 " + df['유출금액_분산'].astype(str) + "입니다. " +
        df['ETF'].astype(str) + " 종목 조회수는 " + df['종목조회수'].astype(str) + "입니다. " +
        df['ETF'].astype(str) + " 관심 종목 등록 수는 " + df['관심종목등록수'].astype(str) + "입니다. " +
        df['ETF'].astype(str) + " 계좌 수 증감률은 " + df['계좌수증감률'].astype(str) + "입니다. " +
        df['ETF'].astype(str) + " 관심 등록 대비 매수 계좌는 " + df['관심등록대비_매수계좌'].astype(str) + "입니다."
    )

    rag_data = df[['ETF', 'text']].copy()

    return rag_data

# ETF명 컬럼과 해당 ETF에 관련한 정보를 담고 있는 text 컬럼
rag_ready_data = prepare_data_for_rag(ETF)
rag_ready_data.head()

"""### **CSVLoader**
- langchain_community 라이브러리의 document_loaders 모듈의 CSVLoader 클래스를 사용하여 CSV 파일에서 데이터 로드
- CSV 파일의 각 행을 추출하여 서로 다른 Document 객체로 변환
- 문서 객체로 이루어진 리스트 형태로 반환
"""

# csv 형태로 저장
rag_ready_data.to_csv('DACON_ETF/rag_ready_data.csv', index=False)

"""### **GPT API 불러오기**"""

# GPT API key
api_key = "access_key"
os.environ['OPENAI_API_KEY'] = api_key

# CSV 파일 로드
loader = CSVLoader(file_path="DACON_ETF/rag_ready_data.csv")
docs = loader.load()
print(f"문서의 수: {len(docs)}")

# 10번째 페이지의 내용 출력
print(f"\n[페이지내용]\n{docs[20].page_content[:650]}")
print(f"\n[metadata]\n{docs[20].metadata}\n")

"""### **Text split**
- RecursiveCharacterTextSplitter 클래스는 텍스트를 재귀적으로 분할하여 의미적으로 관련 있는 텍스트 조각들이 같이 있도록함.
- 분할된 청크들이 설정된 chunk_size보다 작아질 때까지 분할 과정을 반복
- 여기서 chunk_overlap은 분할된 텍스트 조각들 사이에서 중복으로 포함될 문자 수를 의미
"""

# 문서 분할 단계로 각 청크의 최대 길이를 설정해주고 인접한 청크 간에 겹치는 문자의 수를 지정
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=650,
    chunk_overlap=50,
    add_start_index=True
)
splits = text_splitter.split_documents(docs)

# 분할된 텍스트 조각들의 총 수
len(splits)

splits[24]

"""### **Embedding & Vectorstore**
- 임베딩(Embedding)은 텍스트 데이터를 숫자로 이루어진 벡터로 변환하는 과정
- OpenAI사의 임베딩모델을 사용
- 벡터 저장소(Vector Store)는 벡터 형태로 표현된 데이터, 즉 임베딩 벡터들을 효율적으로 저장하고 검색할 수 있는 시스템이나 데이터베이스를 의미
- 대규모 벡터 데이터셋에서 빠른 속도로 가장 유사한 항목을 찾아내는 것이 목표
"""

# 임베딩 벡터를 데이터베이스에 저장하기 위한 오픈소스 소프트웨어로 Chroma 라이브러리 사용
vectorstore = Chroma.from_documents(
    documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_type="similarity")

"""## **RAG 기반 LLM 모델 생성**
- LLM은 RAG에서 검색된 정보를 바탕으로 고품질의 자연어 응답을 생성하는 핵심 역할을 수행
- 사용자에게 정확하고 유용한 정보를 제공
- 사용 모델은 *ChaptGPT의 gpt-4-turbo-preview*
    - temperature 하이퍼파라미터는 모델의 출력의 다양성과 창의성을 조절하는 역할로 0~1사이의 값으로 조절
    - 숫자가 높을수록 창의적인 답변 생성
"""

# ChaptGPT의 gpt-4-turbo-preview 모델을 사용한 LLM 모델 생성
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
llm = ChatOpenAI(temperature=0.3, model="gpt-4-turbo-preview")

"""### **QA서비스**
- 일반적인 rag chain을 구성하는 것을 넘어서 질문자의 이전 질문을 이해하는 QA 시스템 구축을 위해 채팅 기록을 인식하는 검색기인 create_history_aware_retriever를 사용
- 생성된 RAG 체인에 사용자의 질문과 이전의 대화 기록을 입력값을 넣게 되면 그에 대한 답변을 LLM이 생성
"""

# 사용자 질문과 채팅 기록을 기반으로 독립적인 질문을 형성하는 시스템 프롬프트
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

# 채팅 기록과 사용자의 최신 질문을 바탕으로 질문을 변경하는 프롬프트 템플릿 생성
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),  # 시스템 메시지
        MessagesPlaceholder("chat_history"),         # 채팅 기록 자리 표시자
        ("human", "{input}"),                        # 사용자 질문 자리 표시자
    ]
)

# 사용자의 입력을 바탕으로 쿼리를 생성하고, 인덱싱된 데이터에서 가장 관련성 높은 정보를 검색하여 출력
history_aware_retriever = create_history_aware_retriever(
    llm,  # 대형 언어 모델
    retriever,  # 기본 검색기
    contextualize_q_prompt  # 질문 변경 프롬프트
)

# 질문 응답 시스템 프롬프트
qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\
Please answer in Korean.\

{context}"""

# 질문 응답을 위한 프롬프트 템플릿 생성
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),                 # 시스템 메시지
        MessagesPlaceholder("chat_history"),          # 채팅 기록 자리 표시자
        ("human", "{input}"),                         # 사용자 질문 자리 표시자
    ]
)

# 질문 응답 체인 생성
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# RAG 체인 검색기와 질문 응답 체인을 결합
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# ETF 종목에 대한 QA 시스템 구축

# 질문자의 질문 기록을 담아두는 리스트 형성
chat_history = []

# ETF 이름 입력 받기
etf_name = input("알고 싶은 ETF의 이름을 입력하세요: ")

# ETF에 대한 질문 생성
question = f"{etf_name}에 대한 정보를 알려주세요."

# AI의 응답 요청
ai_msg = rag_chain.invoke({"input": question, "chat_history": chat_history})

# 대화 이력에 질문과 응답 추가
chat_history.extend([HumanMessage(content=question), ai_msg["answer"]])

# AI의 응답 출력
print(f"답변: {ai_msg['answer']}")

# 추가 질문을 위한 반복 구조
while True:
    follow_up_question = input("추가 질문이 있으면 입력하세요 (종료하려면 '종료' 입력): ")

    if follow_up_question.lower() == '종료':
        print("질문을 종료합니다.")
        break

    # AI의 응답 요청
    ai_msg_follow_up = rag_chain.invoke({"input": follow_up_question, "chat_history": chat_history})

    # 대화 이력에 후속 질문과 응답 추가
    chat_history.extend([HumanMessage(content=follow_up_question), ai_msg_follow_up["answer"]])

    # AI의 응답 출력
    print(f"답변: {ai_msg_follow_up['answer']}")

