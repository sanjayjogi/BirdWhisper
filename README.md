# BirdWhisper

Welcome to the BirdWhisper project! This API allows you to upload audio files, extract features, and classify them using a pre-trained machine learning model. It is designed specifically for bird audio classification.

## Features

- Upload audio files for classification
- Extract audio features using `librosa`
- Noise reduction and high-pass filtering using `noisereduce` and `scipy`
- Classification using a machine learning model
- CORS support for cross-origin requests

## Project Structure

├── main.py                # Main FastAPI application

├── model.py               # Model loading utility

├── feature_extraction.py  # Feature extraction from audio files

├── audio_processing.py    # Noise reduction and high-pass filter

├── requirements.txt       # Python dependencies

└── README.md              # Project documentation



## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/sanjayjogi/BirdWhisper.git
    cd birdwhisper
    ```

2. Create a virtual environment and activate it:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Download the pre-trained model file and place it in the root directory of the project.

## Running the Application

Start the FastAPI application:

```bash
uvicorn main:app --reload
```

### Upload Audio

- **URL**: `/upload-audio/`
- **Method**: `POST`
- **Description**: Upload an audio file for classification.

#### Request

- **File**: `file` (multipart/form-data)

#### Response

json
```bash
{
  "filename": "example.mp3",
  "filepath": "files/example.mp3",
  "prediction": [1]
}
```
