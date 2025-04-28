from flask import Flask, request, jsonify, render_template
from db import get_user_embeddings_collection
from face_utils import base64_to_image, generate_embedding, compare_embeddings
import os
from flask_cors import CORS
from bson import ObjectId
import logging
import gc
import tensorflow as tf

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configure TensorFlow memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        logger.warning(f"GPU memory growth configuration failed: {e}")

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["*"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Accept"],
        "supports_credentials": True
    }
})

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Accept')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.json
        
        if not data or 'user_id' not in data or 'face_images' not in data:
            return jsonify({'error': 'Missing required fields: user_id and face_images'}), 400
        
        user_id = data['user_id']
        face_images_base64 = data['face_images']
        
        if not isinstance(face_images_base64, list):
            return jsonify({'error': 'face_images must be an array of base64 strings'}), 400
            
        if len(face_images_base64) > 3:  # Further reduced to 3 images
            return jsonify({'error': 'Maximum 3 images allowed per user'}), 400
        
        # Generate embeddings for all images
        embeddings = []
        embeddings_count = 0
        for base64_image in face_images_base64:
            try:
                image = base64_to_image(base64_image)
                embedding = generate_embedding(image)
                embeddings.append(embedding)
                embeddings_count += 1
                
                # Clean up memory aggressively
                del image
                del embedding
                gc.collect()
                tf.keras.backend.clear_session()
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                return jsonify({'error': f'Error processing image: {str(e)}'}), 500
        
        # Convert string ID to ObjectId
        try:
            user_object_id = ObjectId(user_id)
        except:
            return jsonify({'error': 'Invalid user ID format'}), 400
        
        # Find user by ObjectId and update embeddings
        collection = get_user_embeddings_collection()
        result = collection.update_one(
            {'_id': user_object_id},
            {'$set': {'embeddings': embeddings}},
            upsert=False
        )
        
        if result.matched_count == 0:
            return jsonify({'error': 'User not found'}), 404
        
        if(result.matched_count > 0):
            try:
                collection.update_one(
                    {'_id': user_object_id},
                    {'$set': {'isFaceRegistered': True}},
                    upsert=False
                )

            except Exception as e:
                logger.error(f"Error updating user: {str(e)}")
                return jsonify({'error': f'Error updating user: {str(e)}'}), 500
        
        # Clean up memory
        del embeddings
        gc.collect()
        tf.keras.backend.clear_session()
        
        return jsonify({
            'message': 'Faces registered successfully',
            'count': embeddings_count
        }), 200
        
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return jsonify({'error': f'Registration failed: {str(e)}'}), 500

@app.route('/verify', methods=['POST'])
def verify():
    try:
        data = request.json
        
        if not data or 'user_id' not in data or 'face_image' not in data:
            return jsonify({'error': 'Missing required fields: user_id and face_image'}), 400
        
        user_id = ObjectId(data['user_id'])
        face_image_base64 = data['face_image']
        
        # Get stored embeddings
        collection = get_user_embeddings_collection()
        stored_user = collection.find_one({'_id': user_id})
        
        if not stored_user:
            return jsonify({'error': 'User not found'}), 404
        
        if(stored_user['isFaceRegistered'] == False):
            return jsonify({'error': 'User face not registered'}), 404
            
        if 'embeddings' not in stored_user or not stored_user['embeddings']:
            return jsonify({'error': 'No face embeddings found for this user'}), 404
        
        try:
            # Convert base64 to image
            image = base64_to_image(face_image_base64)
            
            # Generate new embedding
            new_embedding = generate_embedding(image)
            
            # Compare with all stored embeddings and get the best match
            best_similarity = 0
            for stored_embedding in stored_user['embeddings']:
                is_verified, similarity = compare_embeddings(
                    stored_embedding,
                    new_embedding
                )
                if similarity > best_similarity:
                    best_similarity = similarity
            
            # Clean up memory
            del image
            del new_embedding
            gc.collect()
            tf.keras.backend.clear_session()
            
            # Consider verified if any of the stored embeddings matches above threshold
            is_verified = best_similarity > 0.5  # Adjusted threshold for OpenFace
            
            return jsonify({
                'verified': is_verified,
                'similarity': float(best_similarity)
            }), 200
            
        except Exception as e:
            logger.error(f"Verification processing error: {str(e)}")
            return jsonify({'error': f'Error processing verification: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"Verification error: {str(e)}")
        return jsonify({'error': f'Verification failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 