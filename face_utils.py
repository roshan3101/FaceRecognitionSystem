from deepface import DeepFace
import numpy as np
from PIL import Image
import base64
import io

def base64_to_image(base64_string):
    """Convert base64 string to PIL Image"""
    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        raise ValueError(f"Invalid base64 image data: {str(e)}")

def generate_embedding(image):
    """Generate face embedding using DeepFace with FaceNet model"""
    try:
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Generate embedding using FaceNet model
        embedding = DeepFace.represent(
            img_path=image,
            model_name='Facenet',
            detector_backend='opencv',
            enforce_detection=True
        )
        print("Generated embedding successfully!!")
        
        return embedding[0]['embedding']
    except Exception as e:
        raise ValueError(f"Face detection or embedding generation failed: {str(e)}")

def compare_embeddings(embedding1, embedding2, threshold=0.7):
    """Compare two embeddings using cosine similarity"""
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)
    
    # Calculate cosine similarity
    similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    
    print("Similarity calculated successfully!!",similarity)
    # Return True if similarity is above threshold
    return similarity > threshold, float(similarity) 