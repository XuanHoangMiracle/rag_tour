from pymongo import MongoClient
from django.conf import settings
from urllib.parse import urlparse

class MongoDBClient:
    _instance = None
    _client = None
    _database = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._client = MongoClient(settings.MONGODB_URI)
            
            parsed_uri = urlparse(settings.MONGODB_URI)
            db_name = parsed_uri.path.strip('/').split('?')[0]
            
            if not db_name:
                raise ValueError("Database name not found in MONGODB_URI")
            
            cls._database = cls._client[db_name]
            print(f"✅ Connected to MongoDB database: {db_name}")
            
        return cls._instance
    
    def get_database(self):
        return self._database
    
    def get_collection(self, collection_name):
        return self._database[collection_name]

# Singleton instance
mongodb_client = MongoDBClient()
