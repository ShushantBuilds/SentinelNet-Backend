from django.apps import AppConfig
import joblib
import os

class ApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'
    rf_model = None

    def ready(self):
        # Build the path dynamically
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        model_path = os.path.join(base_dir, 'notebooks', 'saved_models', 'random_forest_model.pkl')
        
        print("\n--- Model Loading Check ---")
        print(f"Looking for model at: {model_path}")
        
        # Load the model into memory if it exists
        if os.path.exists(model_path):
            # THE FIX: Assign it to the CLASS variable (ApiConfig), not the instance (self)
            ApiConfig.rf_model = joblib.load(model_path)
            print("✅ Status: SUCCESS - Random Forest Model loaded into Django!")
        else:
            print("❌ Status: FAILED - Model file not found at the path above.")
            print("---------------------------\n")