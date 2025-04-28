from flask import Flask, request, jsonify, render_template
from db import get_user_embeddings_collection
from face_utils import base64_to_image, generate_embedding, compare_embeddings
import os
from flask_cors import CORS
from bson import ObjectId
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
            
        if len(face_images_base64) > 15:
            return jsonify({'error': 'Maximum 15 images allowed per user'}), 400
        
        # Generate embeddings for all images
        embeddings = []
        for base64_image in face_images_base64:
            # Remove the data URL prefix if present
            if ',' in base64_image:
                base64_image = base64_image.split(',')[1]
            
            image = base64_to_image(base64_image)
            embedding = generate_embedding(image)
            embeddings.append(embedding)
        
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
            upsert=False  # Don't create new document if not found
        )
        
        if result.matched_count == 0:
            return jsonify({'error': 'User not found'}), 404
        
        return jsonify({
            'message': 'Faces registered successfully',
            'count': len(embeddings)
        }), 200
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Registration failed: {str(e)}'}), 500

@app.route('/verify', methods=['POST'])
def verify():
    try:
        data = request.json
        
        if not data or 'user_id' not in data or 'face_image' not in data:
            return jsonify({'error': 'Missing required fields: user_id and face_image'}), 400
        
        user_id = ObjectId(data['user_id'])
        face_image_base64 = data['face_image']
        
        # Remove the data URL prefix if present
        if ',' in face_image_base64:
            face_image_base64 = face_image_base64.split(',')[1]
        
        # Get stored embeddings
        collection = get_user_embeddings_collection()
        stored_user = collection.find_one({'_id': user_id})
        
        if not stored_user:
            return jsonify({'error': 'User not found'}), 404
        
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
        
        # Consider verified if any of the stored embeddings matches above threshold
        is_verified = best_similarity > 0.7
        
        return jsonify({
            'verified': is_verified,
            'similarity': best_similarity
        }), 200
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Verification failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 