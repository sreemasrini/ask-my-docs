import os
import logging
from typing import List
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pymongo
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title=os.getenv("APP_NAME", "Ask My Docs"))

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB Connection
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client = pymongo.MongoClient(MONGO_URI)
DATABASE_NAME = os.getenv("DATABASE_NAME", "ask_my_docs")
db = client[DATABASE_NAME]
documents_collection = db["documents"]
embeddings_collection = db["embeddings"]

# Initialize OpenAI client
openai_client = OpenAI()

# ======================= Models =========================
class QueryRequest(BaseModel):
    query: str

class DocumentResponse(BaseModel):
    document_id: str
    content: str
    score: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[DocumentResponse]

# ======================= Helper Functions =========================
def get_embedding(text: str, model="text-embedding-3-small"):
    """Generate embedding for a given text using OpenAI."""
    try:
        return openai_client.embeddings.create(
            model=model,
            input=text
        ).data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def process_document(file_path: str):
    """Process document and generate embeddings."""
    # Determine loader based on file extension
    if file_path.lower().endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.lower().endswith('.txt'):
        loader = TextLoader(file_path)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    # Split document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    docs = loader.load_and_split(text_splitter)

    # Generate and store embeddings
    document_metadata = {
        "filename": os.path.basename(file_path),
        "chunks": []
    }

    for doc in docs:
        try:
            chunk_embedding = get_embedding(doc.page_content)
            chunk_doc = {
                "content": doc.page_content,
                "embedding": chunk_embedding
            }
            document_metadata["chunks"].append(chunk_doc)
            
            # Store embedding in MongoDB
            embeddings_collection.insert_one({
                "document_filename": os.path.basename(file_path),
                "content": doc.page_content,
                "embedding": chunk_embedding
            })
        except Exception as e:
            logger.error(f"Error processing document chunk: {e}")
            # Continue processing other chunks even if one fails

    return document_metadata

# ======================= API Routes =========================
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document."""
    logger.info(f"Received file upload: {file.filename}")
    
    # Save uploaded file
    upload_dir = "/tmp/ask_my_docs"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Process document
        document_metadata = process_document(file_path)
        logger.info(f"Document processed successfully: {file.filename}")
        return {"message": "Document uploaded and processed successfully", "filename": file.filename}
    except Exception as e:
        logger.error(f"Error processing document {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query documents using RAG approach."""
    try:
        # Generate query embedding
        query_embedding = get_embedding(request.query)

        # Retrieve top-k most similar document chunks
        k = 10  # Number of top chunks to retrieve
        similar_chunks = list(embeddings_collection.aggregate([
            {
                "$addFields": {
                    "similarity": {
                        "$function": {
                            "body": """function(embedding, queryEmbedding) {
                                const dotProduct = embedding.reduce((sum, a, i) => sum + a * queryEmbedding[i], 0);
                                const embeddingMagnitude = Math.sqrt(embedding.reduce((sum, a) => sum + a * a, 0));
                                const queryMagnitude = Math.sqrt(queryEmbedding.reduce((sum, a) => sum + a * a, 0));
                                return dotProduct / (embeddingMagnitude * queryMagnitude);
                            }""",
                            "args": ["$embedding", query_embedding],
                            "lang": "js"
                        }
                    }
                }
            },
            { "$sort": { "similarity": -1 } },
            { "$limit": k }
        ]))

        # Prepare context for LLM
        context = "\n\n".join([chunk['content'] for chunk in similar_chunks])
        
        # Generate answer using OpenAI
        prompt = f"""You are an AI assistant. Answer based ONLY on the documents.
If the answer is not present, reply 'I don't know.'
Context: {context}
Question: {request.query}
Answer:"""

        llm_response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )

        # Prepare sources for response
        sources = [
            DocumentResponse(
                document_id=str(chunk['_id']), 
                content=chunk['content'], 
                score=chunk.get('similarity', 0)
            ) for chunk in similar_chunks
        ]

        return QueryResponse(
            answer=llm_response.choices[0].message.content or "I don't know.",
            sources=sources
        )
    except Exception as e:
        logger.error(f"Error querying documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ======================= Run Server =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
