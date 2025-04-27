from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

# MongoDB connection string - you should set this in your environment variables
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
DB_NAME = os.getenv('DB_NAME', 'test')

client = MongoClient(MONGODB_URI)
db = client[DB_NAME]
user_embeddings = db['users']

def get_db():
    print("DB connected successfully!!")
    return db

def get_user_embeddings_collection():
    print("User embeddings collection connected successfully!!")
    return user_embeddings 