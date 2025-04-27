# Face Recognition API

A Flask-based API for face registration and verification using DeepFace and MongoDB.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up MongoDB:
- Install MongoDB locally or use a cloud service
- Create a `.env` file with your MongoDB connection string:
```
MONGODB_URI=mongodb://localhost:27017/
```

3. Run the application:
```bash
python app.py
```

## API Endpoints

### Register Faces
- **URL**: `/register`
- **Method**: `POST`
- **Body**:
```json
{
    "user_id": "unique_user_id",
    "face_images": [
        "base64_encoded_image1",
        "base64_encoded_image2",
        ...
    ]
}
```
- **Notes**:
  - Maximum 15 images per user
  - All images must be base64 encoded strings
  - Images will be stored as face embeddings in MongoDB

### Verify Face
- **URL**: `/verify`
- **Method**: `POST`
- **Body**:
```json
{
    "user_id": "unique_user_id",
    "face_image": "base64_encoded_image"
}
```

## Response Format

### Register
- Success: `{"message": "Faces registered successfully", "count": 5}`
- Error: `{"error": "error message"}`

### Verify
- Success: `{"verified": true/false, "similarity": 0.85}`
- Error: `{"error": "error message"}`

## Notes
- The API uses FaceNet model for face embedding generation
- Default similarity threshold is set to 0.7
- Images should be sent as base64 encoded strings
- Make sure MongoDB is running before starting the application
- Each user can have up to 15 face images stored
- Verification compares against all stored embeddings and returns the best match 