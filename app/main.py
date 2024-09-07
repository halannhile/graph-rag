from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from graphrag.document_processor import process_document
from graphrag.graph_creator import GraphCreator
from graphrag.query_processor import QueryProcessor
from graphrag.llm_interface import LLMInterface
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

llm_interface = LLMInterface()
graph_creator = GraphCreator(llm_interface)
query_processor = QueryProcessor(graph_creator)

class Query(BaseModel):
    query: str

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        logger.info(f"File read: {file.filename}")
        
        chunks = process_document(content, file.filename)
        logger.info(f"Document processed into {len(chunks)} chunks")
        
        graph_creator.create_graph(chunks)
        logger.info("Graph created successfully")
        
        return {"filename": file.filename, "status": "processed"}
    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.get("/graph")
async def get_graph():
    return graph_creator.get_graph_data()

@app.post("/query")
async def query(query: Query):
    try:
        logger.info(f"Received query: {query.query}")
        result = query_processor.process_query(query.query)
        logger.info("Query processed successfully")
        return {"result": result}
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)