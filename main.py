import os
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from model import load_model
from feature_extraction import feature_extractor

os.makedirs("files", exist_ok=True)

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

@app.get("/")
async def read_root():
    logger.info("Root endpoint accessed")
    return {"message": "Welcome to the Audio Classification API"}

@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    try:
        file_location = f"files/{file.filename}"
        
        with open(file_location, "wb") as audio_file:
            audio_file.write(file.file.read())
        
        logger.info(f"File uploaded successfully: {file.filename}")
        
        the_test = feature_extractor(file_location)
        model = load_model('model.pkl')
        
        prediction = model.predict(the_test)
        
        logger.info(f"Prediction made for file: {file.filename}")
        
        return JSONResponse(content={"filename": file.filename, "filepath": file_location, "prediction": prediction.tolist()})
    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

