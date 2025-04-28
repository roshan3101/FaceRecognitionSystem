from deepface import DeepFace
import numpy as np
import cv2
import base64
import io
from PIL import Image
import gc
import os

# Set environment variable to limit TensorFlow memory usage
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

def base64_to_image(base64_string):
    try:
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
        
        # Resize image to reduce memory usage
        image = cv2.resize(image, (96, 96))  # OpenFace expects 96x96 images
        
        return image
    except Exception as e:
        print(f"Error converting base64 to image: {str(e)}")
        raise

def generate_embedding(image):
    try:
        # Use OpenFace model (much lighter than VGG-Face)
        embedding = DeepFace.represent(
            img_path=image,
            model_name="OpenFace",
            detector_backend="opencv",
            enforce_detection=False,
            align=False  # Skip alignment to save memory
        )
        
        # Clean up memory aggressively
        gc.collect()
        if 'tensorflow' in globals():
            import tensorflow as tf
            tf.keras.backend.clear_session()
        
        return embedding[0]["embedding"]
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        raise

def compare_embeddings(embedding1, embedding2):
    try:
        # Convert to numpy arrays if they aren't already
        embedding1 = np.array(embedding1, dtype=np.float32)  # Use float32 to save memory
        embedding2 = np.array(embedding2, dtype=np.float32)
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        
        # Clean up memory aggressively
        del embedding1
        del embedding2
        gc.collect()
        
        # Consider verified if similarity is above threshold
        is_verified = similarity > 0.5  # Adjusted threshold for OpenFace
        
        return is_verified, float(similarity)  # Convert to Python float
    except Exception as e:
        print(f"Error comparing embeddings: {str(e)}")
        raise 