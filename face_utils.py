from deepface import DeepFace
import numpy as np
import cv2
import base64
import io
from PIL import Image
import gc

def base64_to_image(base64_string):
    # Remove the data URL prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    # Decode base64 string
    image_data = base64.b64decode(base64_string)
    
    # Convert to numpy array
    image = Image.open(io.BytesIO(image_data))
    image = np.array(image)
    
    # Convert to RGB if needed
    if len(image.shape) == 2:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    return image

def generate_embedding(image):
    try:
        # Use a lighter model (VGG-Face is lighter than Facenet)
        embedding = DeepFace.represent(
            img_path=image,
            model_name="VGG-Face",
            detector_backend="opencv",
            enforce_detection=False
        )
        
        # Clean up memory
        gc.collect()
        
        return embedding[0]["embedding"]
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        raise

def compare_embeddings(embedding1, embedding2):
    try:
        # Convert to numpy arrays if they aren't already
        embedding1 = np.array(embedding1)
        embedding2 = np.array(embedding2)
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        
        # Clean up memory
        gc.collect()
        
        # Consider verified if similarity is above threshold
        is_verified = similarity > 0.6  # Adjusted threshold for VGG-Face
        
        return is_verified, similarity
    except Exception as e:
        print(f"Error comparing embeddings: {str(e)}")
        raise 