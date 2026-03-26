# mongodb_handler.py
import os
from datetime import datetime
import pymongo
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parent / '.env')



class MongoDBHandler:
    def __init__(self, connection_string=None):
        """Initialize MongoDB connection."""
        self.client = None
        self.db = None
        # Use your connection string
        self.connection_string = connection_string or self._get_default_connection()
        self.connected = False
        
        # Print connection info
        print(f"\n🔗 MongoDB Connection Configuration:")
        print(f"   Connection String: {self._mask_connection_string(self.connection_string)}")
        
    def _mask_connection_string(self, conn_str):
        """Mask sensitive info in connection string."""
        if '@' in conn_str:
            parts = conn_str.split('@')
            masked = f"*****@{parts[1]}" if '://' in parts[0] else f"*****@{parts[1]}"
            return masked
        return conn_str
    
    def _convert_numpy_types(self, obj):
        """Convert NumPy types to Python native types for MongoDB compatibility."""
        if isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def _get_default_connection(self):
        database_url = os.getenv("MONGO_URI")
        if not database_url:
            raise ValueError("❌ MONGO_URI not found in environment variables")
        return database_url 
    
    def connect(self):
        """Connect to MongoDB."""
        try:
            print("\n🔌 Connecting to MongoDB Atlas...")
            print("   Attempting connection to MongoDB cluster...")
            
            self.client = MongoClient(self.connection_string, serverSelectionTimeoutMS=5000)
            
            # Test connection
            self.client.admin.command('ping')
            self.connected = True
            print("✅ Connected to MongoDB Atlas successfully!")
            
            # Get server info
            server_info = self.client.server_info()
            print(f"   MongoDB Version: {server_info.get('version', 'Unknown')}")
            print(f"   Cluster: {server_info.get('gitVersion', 'Unknown')}")
            
            # Use or create database
            self.db = self.client['traffic_detection_db']
            print(f"   Database: traffic_detection_db")
            
            # List existing collections
            collections = self.db.list_collection_names()
            if collections:
                print(f"   Existing Collections: {', '.join(collections)}")
            else:
                print(f"   No existing collections - will create when data is saved")
            
            # Create collections if they don't exist
            self._create_collections()
            
            return True
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            print(f"❌ Failed to connect to MongoDB Atlas: {e}")
            print("\n💡 Troubleshooting Tips:")
            print("   1. Check your internet connection")
            print("   2. Verify your MongoDB credentials")
            print("   3. Make sure your IP is whitelisted in MongoDB Atlas")
            print("   4. Go to https://cloud.mongodb.com and check Network Access")
            print("   5. Add your current IP to the whitelist: 0.0.0.0/0 for testing")
            self.connected = False
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            self.connected = False
            return False
    
    def _create_collections(self):
        """Create collections with indexes."""
        try:
            # Analysis results collection
            if 'analysis_results' not in self.db.list_collection_names():
                self.db.create_collection('analysis_results')
                self.db.analysis_results.create_index('timestamp')
                self.db.analysis_results.create_index('traffic_level')
                print("   ✓ Created analysis_results collection")
            
            # Model metrics collection
            if 'model_metrics' not in self.db.list_collection_names():
                self.db.create_collection('model_metrics')
                self.db.model_metrics.create_index('training_date')
                print("   ✓ Created model_metrics collection")
            
            # Traffic data collection
            if 'traffic_data' not in self.db.list_collection_names():
                self.db.create_collection('traffic_data')
                self.db.traffic_data.create_index('timestamp')
                self.db.traffic_data.create_index('location')
                print("   ✓ Created traffic_data collection")
            
            print("✅ Collections ready with indexes")
        except Exception as e:
            print(f"⚠️ Warning: Could not create collections: {e}")
    
    def save_analysis_result(self, image_path, traffic_level, confidence, probability, metadata=None):
        """Save image analysis result to MongoDB with type conversion."""
        if not self.connected:
            print("⚠️ Not connected to MongoDB - data not saved")
            return None
        
        # Convert NumPy types to Python native types
        confidence = float(confidence)
        probability = float(probability)
        
        document = {
            'image_path': str(image_path),
            'traffic_level': str(traffic_level),
            'confidence': confidence,
            'probability': probability,
            'timestamp': datetime.now(),
            'metadata': metadata or {}
        }
        
        # Convert any NumPy types in metadata
        document = self._convert_numpy_types(document)
        
        try:
            result = self.db.analysis_results.insert_one(document)
            print(f"   💾 Saved to MongoDB: {traffic_level} ({confidence:.1%})")
            return result.inserted_id
        except Exception as e:
            print(f"❌ Failed to save analysis: {e}")
            return None
    
    def save_model_metrics(self, model_name, accuracy, f1_score, roc_auc, training_samples, epochs):
        """Save model training metrics to MongoDB with type conversion."""
        if not self.connected:
            print("⚠️ Not connected to MongoDB - metrics not saved")
            return None
        
        # Convert NumPy types to Python native types
        accuracy = float(accuracy) if accuracy else 0.0
        f1_score = float(f1_score) if f1_score else 0.0
        roc_auc = float(roc_auc) if roc_auc else 0.0
        training_samples = int(training_samples)
        epochs = int(epochs)
        
        document = {
            'model_name': str(model_name),
            'accuracy': accuracy,
            'f1_score': f1_score,
            'roc_auc': roc_auc,
            'training_samples': training_samples,
            'epochs': epochs,
            'training_date': datetime.now(),
            'model_version': f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
        # Convert any NumPy types
        document = self._convert_numpy_types(document)
        
        try:
            result = self.db.model_metrics.insert_one(document)
            print(f"   💾 Saved model metrics to MongoDB")
            return result.inserted_id
        except Exception as e:
            print(f"❌ Failed to save model metrics: {e}")
            return None
    
    def save_traffic_data(self, traffic_level, confidence, location=None, additional_data=None):
        """Save real-time traffic data to MongoDB with type conversion."""
        if not self.connected:
            print("⚠️ Not connected to MongoDB - traffic data not saved")
            return None
        
        # Convert NumPy types to Python native types
        confidence = float(confidence)
        
        document = {
            'traffic_level': str(traffic_level),
            'confidence': confidence,
            'location': str(location) if location else 'unknown',
            'timestamp': datetime.now(),
            'additional_data': additional_data or {}
        }
        
        # Convert any NumPy types in additional_data
        document = self._convert_numpy_types(document)
        
        try:
            result = self.db.traffic_data.insert_one(document)
            print(f"   💾 Saved traffic data to MongoDB")
            return result.inserted_id
        except Exception as e:
            print(f"❌ Failed to save traffic data: {e}")
            return None
    
    def get_recent_analyses(self, limit=10):
        """Get recent analysis results."""
        if not self.connected:
            return []
        
        try:
            results = self.db.analysis_results.find().sort('timestamp', -1).limit(limit)
            return list(results)
        except Exception as e:
            print(f"❌ Failed to get analyses: {e}")
            return []
    
    def get_traffic_statistics(self):
        """Get traffic statistics from database."""
        if not self.connected:
            return {}
        
        try:
            # Count by traffic level
            heavy_count = self.db.analysis_results.count_documents({'traffic_level': 'HEAVY TRAFFIC'})
            low_count = self.db.analysis_results.count_documents({'traffic_level': 'LOW TRAFFIC'})
            
            # Average confidence
            pipeline = [
                {'$match': {'confidence': {'$exists': True}}},
                {'$group': {
                    '_id': None,
                    'avg_confidence': {'$avg': '$confidence'},
                    'total_analyses': {'$sum': 1}
                }}
            ]
            stats = list(self.db.analysis_results.aggregate(pipeline))
            
            return {
                'heavy_count': heavy_count,
                'low_count': low_count,
                'total_analyses': heavy_count + low_count,
                'avg_confidence': stats[0]['avg_confidence'] if stats else 0,
                'heavy_percentage': (heavy_count / (heavy_count + low_count) * 100) if (heavy_count + low_count) > 0 else 0
            }
        except Exception as e:
            print(f"❌ Failed to get statistics: {e}")
            return {}
    
    def export_to_csv(self, collection_name, filename=None):
        """Export collection to CSV with type conversion."""
        if not self.connected:
            return False
        
        if filename is None:
            filename = f"{collection_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        try:
            collection = self.db[collection_name]
            data = list(collection.find())
            
            if data:
                # Convert ObjectId and datetime to strings
                for doc in data:
                    doc['_id'] = str(doc['_id'])
                    if 'timestamp' in doc:
                        doc['timestamp'] = doc['timestamp'].isoformat()
                
                df = pd.DataFrame(data)
                df.to_csv(filename, index=False)
                print(f"✅ Exported {len(data)} records to {filename}")
                return True
            else:
                print(f"⚠️ No data in {collection_name}")
                return False
        except Exception as e:
            print(f"❌ Failed to export: {e}")
            return False
    
    def close(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            print("🔌 MongoDB Atlas connection closed")
            self.connected = False


# Quick test function
def test_mongodb_connection():
    """Test MongoDB connection."""
    print("="*50)
    print("Testing MongoDB Atlas Connection")
    print("="*50)
    
    handler = MongoDBHandler()
    
    if handler.connect():
        # Test saving data
        print("\n📝 Testing data save...")
        handler.save_traffic_data("TEST CONNECTION", 0.99, "connection_test")
        
        # Get statistics
        stats = handler.get_traffic_statistics()
        print(f"\n📊 Database Statistics: {stats}")
        
        handler.close()
        return True
    else:
        return False


if __name__ == "__main__":
    test_mongodb_connection()