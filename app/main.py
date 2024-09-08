from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from graphrag.document_processor import process_document
from graphrag.graph_creator import GraphCreator
from graphrag.query_processor import QueryProcessor
from graphrag.llm_interface import LLMInterface
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

llm_interface = LLMInterface()
graph_creator = GraphCreator(llm_interface)
query_processor = QueryProcessor(graph_creator)

graph_status = {"status": "Not started", "progress": 0, "message": ""}

class Query(BaseModel):
    query: str

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        chunks = process_document(content, file.filename)
        graph_creator.add_document_to_graph(chunks)
        return {"filename": file.filename, "status": "processed"}
    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": str(e)})

def finalize_graph_task():
    global graph_status
    try:
        graph_status = {"status": "In progress", "progress": 25, "message": "Creating element summaries"}
        graph_creator.create_element_summaries()
        
        graph_status = {"status": "In progress", "progress": 50, "message": "Creating communities"}
        graph_creator.create_communities()
        
        graph_status = {"status": "In progress", "progress": 75, "message": "Creating community summaries"}
        graph_creator.create_community_summaries()
        
        graph_status = {"status": "Completed", "progress": 100, "message": "Graph finalization completed"}
    except Exception as e:
        logger.error(f"Error finalizing graph: {str(e)}")
        graph_status = {"status": "Error", "progress": -1, "message": f"Error: {str(e)}"}

@app.post("/finalize_graph")
async def finalize_graph(background_tasks: BackgroundTasks):
    background_tasks.add_task(finalize_graph_task)
    return {"status": "Graph finalization started"}

@app.get("/graph_status")
async def get_graph_status():
    return graph_status

@app.get("/graph")
async def get_graph():
    data = graph_creator.get_graph_data()
    logger.info(f"Graph data: {data}")
    if not data['nodes'] and not data['edges']:
        logger.warning("Graph is empty")
    return data

@app.post("/query")
async def query(query: Query):
    try:
        result = query_processor.process_query(query.query)
        return {"result": result}
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)