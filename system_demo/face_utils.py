import cv2
import numpy as np

def preprocess_image(image_path, size=(11, 11)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image at {image_path} not found.")
    
    img = cv2.resize(img, size)
    flattened = img.flatten()
    
    mean_val = np.mean(flattened)
    std_val = np.std(flattened)
    
    features = np.concatenate([flattened, [mean_val, std_val]])
    return features.reshape(1, -1)

def verify_face(image_path, facial_model, label_encoder, expected_user=None):
    try:
        features = preprocess_image(image_path)
        prediction = facial_model.predict(features)[0]
        predicted_user = label_encoder.inverse_transform([prediction])[0]

        print(f"[INFO] Predicted Face: {predicted_user}")

        if expected_user:
            return predicted_user.lower() == expected_user.lower()
        else:
            return predicted_user.lower() not in ['unknown', 'unauthorized']

    except Exception as e:
        print(f"[ERROR] Facial recognition failed: {e}")
        return False