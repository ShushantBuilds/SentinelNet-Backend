from django.apps import AppConfig
import joblib
import os

class ApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'
    rf_model = None

    def ready(self):
        # 1. Get the directory where THIS file (apps.py) lives
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 2. Look for the model in this exact same directory
        model_path = os.path.join(current_dir, 'random_forest_model.pkl')
        
        print("\n--- Model Loading Check ---")
        print(f"Looking for model at: {model_path}")
        
        # 3. Load the model into memory if it exists
        if os.path.exists(model_path):
            ApiConfig.rf_model = joblib.load(model_path)
            print("✅ Status: SUCCESS - Random Forest Model loaded into Django!")
        else:
            print("❌ Status: FAILED - Model file not found at the path above.")
            print("---------------------------\n")