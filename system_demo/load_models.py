import joblib
import os

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

def load_facial_model():
    model = joblib.load(os.path.join(MODEL_PATH, "facial_model.pkl"))
    label_encoder = joblib.load(os.path.join(MODEL_PATH, "facial_label_encoder.pkl"))
    return model, label_encoder

def load_voice_model():
    model = joblib.load(os.path.join(MODEL_PATH, "voice_model.pkl"))
    label_encoder = joblib.load(os.path.join(MODEL_PATH, "voice_label_encoder.pkl"))
    return model, label_encoder

def load_product_model():
    model = joblib.load(os.path.join(MODEL_PATH, "product_model.pkl"))
    label_encoder = joblib.load(os.path.join(MODEL_PATH, "product_label_encoders.pkl"))
    return model, label_encoder

def load_encoders():
    """Load all label encoders for the recommendation system"""
    return joblib.load(os.path.join(MODEL_PATH, "product_label_encoders.pkl"))