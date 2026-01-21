"""
Diabetes Prediction Web Application
Flask API for diabetes prediction using trained ML models
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import json
import os
from datetime import datetime

app = Flask(__name__)

# Load the trained model and scaler
try:
    model = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    
    # Load training results
    with open('models/training_results.json', 'r') as f:
        training_results = json.load(f)
    
    print("✓ Model and scaler loaded successfully")
    print(f"✓ Best model: {training_results['best_model']}")
except FileNotFoundError:
    print("⚠ Warning: Model files not found. Please run train_models.py first.")
    model = None
    scaler = None
    training_results = None

# Feature names and their descriptions
FEATURE_INFO = {
    'Pregnancies': {
        'description': 'Number of times pregnant',
        'range': '0-17',
        'unit': 'count'
    },
    'Glucose': {
        'description': 'Plasma glucose concentration (2 hours in oral glucose tolerance test)',
        'range': '0-200',
        'unit': 'mg/dL'
    },
    'BloodPressure': {
        'description': 'Diastolic blood pressure',
        'range': '0-122',
        'unit': 'mm Hg'
    },
    'SkinThickness': {
        'description': 'Triceps skin fold thickness',
        'range': '0-99',
        'unit': 'mm'
    },
    'Insulin': {
        'description': '2-Hour serum insulin',
        'range': '0-846',
        'unit': 'mu U/ml'
    },
    'BMI': {
        'description': 'Body mass index',
        'range': '0-67.1',
        'unit': 'weight in kg/(height in m)^2'
    },
    'DiabetesPedigreeFunction': {
        'description': 'Diabetes pedigree function (genetic influence)',
        'range': '0.078-2.42',
        'unit': 'score'
    },
    'Age': {
        'description': 'Age',
        'range': '21-81',
        'unit': 'years'
    }
}


@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html', 
                          feature_info=FEATURE_INFO,
                          training_results=training_results)


@app.route('/predict', methods=['POST'])
def predict():
    """Make a prediction based on input features"""
    try:
        if model is None or scaler is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please train the model first by running train_models.py'
            }), 500
        
        # Get data from request
        data = request.get_json()
        
        # Extract features in the correct order
        features = [
            float(data.get('pregnancies', 0)),
            float(data.get('glucose', 0)),
            float(data.get('bloodPressure', 0)),
            float(data.get('skinThickness', 0)),
            float(data.get('insulin', 0)),
            float(data.get('bmi', 0)),
            float(data.get('diabetesPedigreeFunction', 0)),
            float(data.get('age', 0))
        ]
        
        # Validate inputs
        if any(f < 0 for f in features):
            return jsonify({
                'success': False,
                'error': 'All values must be non-negative'
            }), 400
        
        # Scale features
        features_array = np.array([features])
        features_scaled = scaler.transform(features_array)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled)[0]
        
        # Get risk level
        diabetes_probability = prediction_proba[1] * 100
        
        if diabetes_probability < 30:
            risk_level = 'Low'
            risk_color = 'success'
        elif diabetes_probability < 70:
            risk_level = 'Moderate'
            risk_color = 'warning'
        else:
            risk_level = 'High'
            risk_color = 'danger'
        
        # Generate recommendations
        recommendations = generate_recommendations(features, diabetes_probability)
        
        # Prepare response
        response = {
            'success': True,
            'prediction': int(prediction),
            'probability': {
                'no_diabetes': round(prediction_proba[0] * 100, 2),
                'diabetes': round(prediction_proba[1] * 100, 2)
            },
            'risk_level': risk_level,
            'risk_color': risk_color,
            'recommendations': recommendations,
            'model_used': training_results['best_model'] if training_results else 'Unknown',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def generate_recommendations(features, diabetes_probability):
    """Generate personalized health recommendations"""
    recommendations = []
    
    # Unpack features
    pregnancies, glucose, bp, skin, insulin, bmi, dpf, age = features
    
    # Glucose recommendations
    if glucose > 140:
        recommendations.append({
            'category': 'Blood Glucose',
            'icon': '🍬',
            'message': 'Your glucose level is high. Consider reducing sugar intake and monitoring blood sugar regularly.',
            'priority': 'high'
        })
    elif glucose > 100:
        recommendations.append({
            'category': 'Blood Glucose',
            'icon': '🍬',
            'message': 'Your glucose level is slightly elevated. Maintain a balanced diet and exercise regularly.',
            'priority': 'medium'
        })
    
    # BMI recommendations
    if bmi > 30:
        recommendations.append({
            'category': 'Body Weight',
            'icon': '⚖️',
            'message': 'Your BMI indicates obesity. Consider a weight management program with diet and exercise.',
            'priority': 'high'
        })
    elif bmi > 25:
        recommendations.append({
            'category': 'Body Weight',
            'icon': '⚖️',
            'message': 'Your BMI is in the overweight range. Maintain a healthy diet and regular physical activity.',
            'priority': 'medium'
        })
    
    # Blood pressure recommendations
    if bp > 90:
        recommendations.append({
            'category': 'Blood Pressure',
            'icon': '💓',
            'message': 'Your blood pressure is elevated. Reduce salt intake and manage stress levels.',
            'priority': 'high'
        })
    
    # Age-related recommendations
    if age > 45:
        recommendations.append({
            'category': 'Age Factor',
            'icon': '👴',
            'message': 'Regular health checkups are recommended for your age group. Monitor diabetes markers annually.',
            'priority': 'medium'
        })
    
    # Overall risk recommendations
    if diabetes_probability > 70:
        recommendations.append({
            'category': 'Urgent Action',
            'icon': '🏥',
            'message': 'High diabetes risk detected. Please consult with a healthcare provider immediately.',
            'priority': 'high'
        })
    elif diabetes_probability > 30:
        recommendations.append({
            'category': 'Prevention',
            'icon': '🏃',
            'message': 'Moderate risk detected. Adopt a diabetes prevention program with diet and exercise.',
            'priority': 'medium'
        })
    else:
        recommendations.append({
            'category': 'Maintain Health',
            'icon': '✅',
            'message': 'Low risk detected. Continue maintaining a healthy lifestyle with regular checkups.',
            'priority': 'low'
        })
    
    # General recommendations
    recommendations.append({
        'category': 'General Health',
        'icon': '🥗',
        'message': 'Eat a balanced diet rich in vegetables, whole grains, and lean proteins.',
        'priority': 'low'
    })
    
    recommendations.append({
        'category': 'Exercise',
        'icon': '💪',
        'message': 'Aim for at least 150 minutes of moderate aerobic activity per week.',
        'priority': 'low'
    })
    
    return recommendations


@app.route('/api/metrics')
def get_metrics():
    """Get model performance metrics"""
    if training_results:
        return jsonify(training_results)
    else:
        return jsonify({
            'error': 'Training results not available'
        }), 404


@app.route('/about')
def about():
    """Render the about page"""
    return render_template('about.html', training_results=training_results)


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🏥 DIABETES PREDICTION WEB APPLICATION")
    print("="*60)
    
    if model and scaler:
        print("✓ Models loaded successfully")
        print(f"✓ Using: {training_results['best_model']}")
        print(f"✓ Trained on: {training_results['training_date']}")
    else:
        print("⚠ Models not found!")
        print("  Please run: python train_models.py")
    
    print("\n🌐 Starting server...")
    print("📍 Access the application at: http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
