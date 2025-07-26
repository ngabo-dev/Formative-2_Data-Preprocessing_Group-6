import pickle
from load_models import load_product_model, load_encoders

def get_user_vector(user_name, encoders):
    """Get user feature vector based on user profile"""
    
    user_profiles = {
        'Omar': {
            'customer_id_new': 'A100',
            'social_media_platform': 'Instagram',
            'engagement_score': 85.5,
            'purchase_interest_score': 90.0,
            'review_sentiment': 'Positive',
            'customer_id_new_numeric': 100,
            'customer_id_legacy': 'CUST001',
            'transaction_id': 'T001',
            'purchase_amount': 250.0,
            'purchase_date': '2024-01-15',
            'customer_rating': 4.5,
            'purchase_month': 1,
            'purchase_day_of_week': 2,
            'total_purchase_amount': 500.0,
            'number_of_transactions': 3,
            'average_customer_rating': 4.2
        },
        'omar': {
            'customer_id_new': 'A100',
            'social_media_platform': 'Instagram',
            'engagement_score': 85.5,
            'purchase_interest_score': 90.0,
            'review_sentiment': 'Positive',
            'customer_id_new_numeric': 100,
            'customer_id_legacy': 'CUST001',
            'transaction_id': 'T001',
            'purchase_amount': 250.0,
            'purchase_date': '2024-01-15',
            'customer_rating': 4.5,
            'purchase_month': 1,
            'purchase_day_of_week': 2,
            'total_purchase_amount': 500.0,
            'number_of_transactions': 3,
            'average_customer_rating': 4.2
        },
        'Deolinda': {
            'customer_id_new': 'A101',
            'social_media_platform': 'TikTok',
            'engagement_score': 75.0,
            'purchase_interest_score': 80.0,
            'review_sentiment': 'Positive',
            'customer_id_new_numeric': 101,
            'customer_id_legacy': 'CUST002',
            'transaction_id': 'T002',
            'purchase_amount': 180.0,
            'purchase_date': '2024-02-10',
            'customer_rating': 4.0,
            'purchase_month': 2,
            'purchase_day_of_week': 5,
            'total_purchase_amount': 360.0,
            'number_of_transactions': 2,
            'average_customer_rating': 4.0
        },
        'jean_Pierre': {
            'customer_id_new': 'A102',
            'social_media_platform': 'Facebook',
            'engagement_score': 65.0,
            'purchase_interest_score': 70.0,
            'review_sentiment': 'Neutral',
            'customer_id_new_numeric': 102,
            'customer_id_legacy': 'CUST003',
            'transaction_id': 'T003',
            'purchase_amount': 320.0,
            'purchase_date': '2024-03-05',
            'customer_rating': 3.5,
            'purchase_month': 3,
            'purchase_day_of_week': 1,
            'total_purchase_amount': 640.0,
            'number_of_transactions': 4,
            'average_customer_rating': 3.8
        },
        'Jean_Pierre': {
            'customer_id_new': 'A102',
            'social_media_platform': 'Facebook',
            'engagement_score': 65.0,
            'purchase_interest_score': 70.0,
            'review_sentiment': 'Neutral',
            'customer_id_new_numeric': 102,
            'customer_id_legacy': 'CUST003',
            'transaction_id': 'T003',
            'purchase_amount': 320.0,
            'purchase_date': '2024-03-05',
            'customer_rating': 3.5,
            'purchase_month': 3,
            'purchase_day_of_week': 1,
            'total_purchase_amount': 640.0,
            'number_of_transactions': 4,
            'average_customer_rating': 3.8
        }
    }
    
    user_features = user_profiles.get(user_name, {
        'customer_id_new': 'A100',
        'social_media_platform': 'Instagram',
        'engagement_score': 50.0,
        'purchase_interest_score': 60.0,
        'review_sentiment': 'Positive',
        'customer_id_new_numeric': 100,
        'customer_id_legacy': 'CUST001',
        'transaction_id': 'T001',
        'purchase_amount': 100.0,
        'purchase_date': '2024-01-01',
        'customer_rating': 3.0,
        'purchase_month': 1,
        'purchase_day_of_week': 1,
        'total_purchase_amount': 200.0,
        'number_of_transactions': 1,
        'average_customer_rating': 3.0
    })
    
    try:
        feature_vector = [
            encoders['customer_id_new'].transform([user_features['customer_id_new']])[0] if 'customer_id_new' in encoders else 0,
            encoders['social_media_platform'].transform([user_features['social_media_platform']])[0] if 'social_media_platform' in encoders else 0,
            user_features['engagement_score'],
            user_features['purchase_interest_score'],
            encoders['review_sentiment'].transform([user_features['review_sentiment']])[0] if 'review_sentiment' in encoders else 0,
            user_features['customer_id_new_numeric'],
            hash(user_features['customer_id_legacy']) % 100,
            hash(user_features['transaction_id']) % 1000,
            user_features['purchase_amount'],
            encoders['purchase_date'].transform([user_features['purchase_date']])[0] if 'purchase_date' in encoders else 0,
            user_features['customer_rating'],
            user_features['purchase_month'],
            user_features['purchase_day_of_week'],
            user_features['total_purchase_amount'],
            user_features['number_of_transactions'],
            user_features['average_customer_rating']
        ]
        
        print(f"[DEBUG] Created {len(feature_vector)} features for {user_name}: {feature_vector}")
        return feature_vector
        
    except Exception as e:
        print(f"[ERROR] Error creating feature vector: {e}")
        return [0] * 16

def recommend_products(user_vector, product_model):
    """Predict product recommendation for user"""
    try:
        prediction = product_model.predict([user_vector])[0]
        return [prediction]
    except Exception as e:
        print(f"[ERROR] Product recommendation failed: {e}")
        return [0]

def recommend_for_user(user_name):
    """Complete recommendation pipeline for a verified user"""
    try:
        product_model, _ = load_product_model()
        encoders = load_encoders()
        
        user_vector = get_user_vector(user_name, encoders)
        
        recommended_products = recommend_products(user_vector, product_model)
        
        if 'product_category' in encoders and recommended_products[0] < len(encoders['product_category'].classes_):
            product_name = encoders['product_category'].inverse_transform([recommended_products[0]])[0]
            return f"{product_name} (Product Code: {recommended_products[0]})"
        else:
            return f"Product Code: {recommended_products[0]}"
            
    except Exception as e:
        print(f"[ERROR] Complete recommendation failed: {e}")
        return "Electronics (Default Product)"
