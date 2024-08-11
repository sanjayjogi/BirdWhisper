from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from model import load_model
from feature_extraction import feature_extractor
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Audio Classification API"}

@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    file_location = f"files/{file.filename}"
    
    with open(file_location, "wb") as audio_file:
        audio_file.write(file.file.read())
    
    # Extract features and load the model
    the_test = feature_extractor(file_location)
    model = load_model('KNeighbors.pkl')
    
    # Predict
    prediction = model.predict(the_test)
    
    return JSONResponse(content={"filename": file.filename, "filepath": file_location, "prediction": prediction.tolist()})
