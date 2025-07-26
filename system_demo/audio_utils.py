import librosa
import numpy as np

def extract_features(audio_path, sr=22050, duration=3):
    """Loads audio and extracts MFCC features."""
    try:
        audio, _ = librosa.load(audio_path, sr=sr, duration=duration)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        
        return mfcc_mean[:3].reshape(1, -1)
        
    except Exception as e:
        raise ValueError(f"Error loading audio {audio_path}: {e}")

def verify_voice(audio_path, voice_model, label_encoder, expected_user=None):
    """Verifies if the voice matches expected_user or checks if authorized."""
    try:
        features = extract_features(audio_path)
        prediction = voice_model.predict(features)[0]
        predicted_user = label_encoder.inverse_transform([prediction])[0]

        print(f"[INFO] Predicted Voice: {predicted_user}")

        if expected_user:
            return predicted_user.lower() == expected_user.lower()
        else:
            return predicted_user.lower() not in ['unknown', 'unauthorized']

    except Exception as e:
        print(f"[ERROR] Voice verification failed: {e}")
        return False