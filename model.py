import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(file_path):
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
            logger.info(f"Model loaded successfully from {file_path}")
        return model
    except FileNotFoundError:
        logger.error(f"Model file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading model from {file_path}: {e}")
        raise
