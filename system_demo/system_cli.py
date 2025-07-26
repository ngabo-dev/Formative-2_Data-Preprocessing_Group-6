import os
import sys
import joblib
from face_utils import verify_face
from audio_utils import verify_voice
from recommend_utils import recommend_for_user

facial_model = joblib.load('../models/facial_model.pkl')
facial_encoder = joblib.load('../models/facial_label_encoder.pkl')

voice_model = joblib.load('../models/voice_model.pkl')
voice_encoder = joblib.load('../models/voice_label_encoder.pkl')

product_model = joblib.load('../models/product_model.pkl')
product_encoders = joblib.load('../models/product_label_encoders.pkl')

authorized_users = {
    "Omar": {"face": "../images/Omar/surprised.jpeg", "voice": "../audio/raw/omar_confirm.wav"},
    "Deolinda": {"face": "../images/Deolinda/surprised.jpeg", "voice": "../audio/raw/deolinda_confirm.wav"},
    "jean_Pierre": {"face": "../images/Jean_Pierre/surprised.jpeg", "voice": "../audio/raw/jean_confirm.wav"},
}

unauthorized_users = {
    "Unauthorized1": {"face": "../images/unauthorized/surprised.jpg", "voice": "../audio/raw/unauthorized_confirm.aac"},
}

def authenticate_and_recommend(user_name, face_path, voice_path):
    print(f"\n🔍 Starting authentication for user: {user_name}")

    if not os.path.exists(face_path):
        print(f"❌ Face image not found at {face_path}")
        return False
    if not os.path.exists(voice_path):
        print(f"❌ Voice audio not found at {voice_path}")
        return False

    face_verified = verify_face(face_path, facial_model, facial_encoder, expected_user=user_name)
    print(f"Face verification: {'✅ Passed' if face_verified else '❌ Failed'}")
    if not face_verified:
        print("Access denied due to face verification failure.")
        return False

    voice_verified = verify_voice(voice_path, voice_model, voice_encoder, expected_user=user_name)
    print(f"Voice verification: {'✅ Passed' if voice_verified else '❌ Failed'}")
    if not voice_verified:
        print("Access denied due to voice verification failure.")
        return False

    print("\n✅ Identity verified successfully!")

    predicted_product_name = recommend_for_user(user_name)

    print(f"🎯 Recommended product for {user_name}: {predicted_product_name}")
    return True

def main():
    print("=== Multimodal Authentication & Product Recommendation Demo ===")
    print("\nSelect user type:")
    print("1. Authorized User")
    print("2. Unauthorized User")
    print("0. Exit")

    choice = input("Enter choice: ").strip()

    if choice == "1":
        print("\nAvailable authorized users:")
        for idx, user in enumerate(authorized_users.keys(), start=1):
            print(f"{idx}. {user}")
        user_choice = input("Select user by number: ").strip()
        try:
            user_name = list(authorized_users.keys())[int(user_choice) - 1]
        except (IndexError, ValueError):
            print("Invalid selection.")
            return
        user_data = authorized_users[user_name]
        authenticate_and_recommend(user_name, user_data["face"], user_data["voice"])

    elif choice == "2":
        print("\nAvailable unauthorized users:")
        for idx, user in enumerate(unauthorized_users.keys(), start=1):
            print(f"{idx}. {user}")
        if not unauthorized_users:
            print("No unauthorized users configured. Add them in the script.")
            return
        user_choice = input("Select user by number: ").strip()
        try:
            user_name = list(unauthorized_users.keys())[int(user_choice) - 1]
        except (IndexError, ValueError):
            print("Invalid selection.")
            return
        user_data = unauthorized_users[user_name]
        authenticate_and_recommend(user_name, user_data["face"], user_data["voice"])

    elif choice == "0":
        print("Exiting.")
        sys.exit(0)

    else:
        print("Invalid choice.")

if __name__ == "__main__":
    while True:
        main()
        input("\nPress Enter to continue...\n")
