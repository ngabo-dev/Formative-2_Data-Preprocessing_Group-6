import os
import joblib
from face_utils import verify_face
from audio_utils import verify_voice
from recommend_utils import get_user_vector, recommend_products

user_name = "KING"
face_img_path = f"../images/{user_name}/surprised.jpg"

audio_files = {
    "KING": "../audio/raw/omar_confirm.wav",
    "Deolinda": "../audio/raw/deolinda_confirm.wav",
    "Jean_Paul": "../audio/raw/Jean_Paul_confirm.wav",
}

voice_path = audio_files.get(user_name)
if not voice_path or not os.path.exists(voice_path):
    print(f"❌ Voice audio file for user '{user_name}' not found!")
    exit()

facial_model = joblib.load('../models/facial_model.pkl')
facial_encoder = joblib.load('../models/facial_label_encoder.pkl')

voice_model = joblib.load('../models/voice_model.pkl')
voice_encoder = joblib.load('../models/voice_label_encoder.pkl')

product_model = joblib.load('../models/product_model.pkl')
product_encoders = joblib.load('../models/product_label_encoders.pkl')

print(f"\n🔍 Starting authentication for user: {user_name}")

face_verified = verify_face(face_img_path, facial_model, facial_encoder, expected_user=user_name)
print(f"Face verification: {'✅ Passed' if face_verified else '❌ Failed'}")

if not face_verified:
    print("Access denied due to face verification failure.")
    exit()

voice_verified = verify_voice(voice_path, voice_model, voice_encoder, expected_user=user_name)
print(f"Voice verification: {'✅ Passed' if voice_verified else '❌ Failed'}")

if not voice_verified:
    print("Access denied due to voice verification failure.")
    exit()

print("\n✅ Identity verified successfully!")

user_vector = get_user_vector(user_name, product_encoders)
predicted_product_code = recommend_products(user_vector, product_model)[0]
predicted_product_name = product_encoders['product'].inverse_transform([predicted_product_code])[0]

print(f"🎯 Recommended product for {user_name}: {predicted_product_name}")

print("\n🎉 Demo completed successfully.")
