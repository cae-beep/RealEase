# chatbot_debug.py - Debug version with crash protection
import sys
import traceback

def setup_crash_handler():
    """Setup global exception handler"""
    def handle_exception(exc_type, exc_value, exc_traceback):
        print("\n" + "="*80)
        print("💥 UNHANDLED EXCEPTION - FULL STACK TRACE")
        print("="*80)
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print("="*80 + "\n")
        
    sys.excepthook = handle_exception

setup_crash_handler()

# Now import everything else
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import re
import os
from datetime import datetime
import logging
import numpy as np
import random

print("🚀 STARTING DEBUG VERSION OF CHATBOT")
print("="*70)

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# CONFIGURATION
MODEL_PATH = 'models/nlu_model.pkl'

# Global variables - initialize as None
vectorizer = None
classifier = None
model_classes = []

print(f"📁 Model path: {MODEL_PATH}")
print(f"📁 Model exists: {os.path.exists(MODEL_PATH)}")

# Try to load model WITH PROTECTION
def safe_load_model():
    global vectorizer, classifier, model_classes
    
    try:
        print("🔄 Loading model...")
        
        if not os.path.exists(MODEL_PATH):
            print("❌ Model file not found!")
            return False
            
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
            print(f"✅ Model pickle loaded")
            
        vectorizer = model_data.get('vectorizer')
        classifier = model_data.get('classifier')
        
        if not vectorizer:
            print("❌ No vectorizer in model")
            return False
        if not classifier:
            print("❌ No classifier in model")
            return False
            
        if hasattr(classifier, 'classes_'):
            model_classes = classifier.classes_.tolist()
            print(f"✅ Model loaded with {len(model_classes)} intents")
            print(f"📊 Intents: {model_classes}")
            return True
        else:
            print("❌ Classifier has no classes_ attribute")
            return False
            
    except Exception as e:
        print(f"💥 ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

# Load model
model_loaded = safe_load_model()

# Simple intent fallback
def simple_intent_detection(query):
    query_lower = query.lower()
    
    if 'financing' in query_lower or 'loan' in query_lower:
        return 'financing'
    elif 'near' in query_lower:
        return 'find_near_landmark'
    elif 'under' in query_lower or 'below' in query_lower:
        if 'bed' in query_lower or 'bath' in query_lower:
            return 'find_property_with_criteria'
    elif 'ready' in query_lower or 'available now' in query_lower:
        return 'find_ready_property'
    elif 'find' in query_lower or 'show me' in query_lower:
        return 'find_property'
    elif 'about' in query_lower or 'tell me' in query_lower:
        return 'location_info'
    
    return 'unknown'

# Simple response
def simple_response(intent, query):
    responses = {
        'financing': f"🏦 **Financing Options**\n\nFor '{query}', I can help with bank loans, Pag-IBIG, and other financing options.",
        'find_near_landmark': f"📍 **Properties Near Landmarks**\n\n'{query}' - I'll help you find properties near schools, malls, hospitals in Batangas.",
        'find_property_with_criteria': f"🔍 **Properties with Criteria**\n\n'{query}' - Searching for properties with specific price and features.",
        'find_ready_property': f"🚚 **Ready Properties**\n\n'{query}' - Looking for immediate occupancy properties.",
        'find_property': f"🏠 **Property Search**\n\n'{query}' - Finding properties in Batangas.",
        'location_info': f"📍 **Location Info**\n\n'{query}' - Information about Batangas locations.",
        'unknown': f"🤔 **General Help**\n\n'{query}' - How can I assist you with property search?"
    }
    return responses.get(intent, responses['unknown'])

@app.route('/api/chat', methods=['POST'])
def chat():
    print("\n" + "="*70)
    print("📞 CHAT REQUEST RECEIVED")
    print("="*70)
    
    try:
        data = request.json
        print(f"📦 Request data: {data}")
        
        if not data:
            return jsonify({'error': 'No data'}), 400
            
        query = data.get('query', '').strip()
        print(f"💬 Query: '{query}'")
        
        if not query:
            return jsonify({'error': 'No query'}), 400
        
        # Step 1: Try model prediction
        intent = "unknown"
        confidence = 0.0
        
        if vectorizer and classifier:
            try:
                print("🧠 Attempting model prediction...")
                # Simple preprocessing
                processed = query.lower()
                processed = re.sub(r'[^\w\s\?\.]', ' ', processed)
                processed = re.sub(r'\s+', ' ', processed).strip()
                
                print(f"📝 Processed: '{processed}'")
                
                X = vectorizer.transform([processed])
                print(f"✅ Vectorized, shape: {X.shape}")
                
                intent = classifier.predict(X)[0]
                proba = classifier.predict_proba(X)[0]
                confidence = float(max(proba))
                
                print(f"🎯 Model prediction: {intent} ({confidence:.1%})")
                
            except Exception as e:
                print(f"❌ Model prediction failed: {e}")
                intent = simple_intent_detection(query)
                print(f"🔧 Fallback intent: {intent}")
        else:
            print("⚠️ No model, using fallback")
            intent = simple_intent_detection(query)
        
        # Generate response
        response_text = simple_response(intent, query)
        
        print(f"💬 Generated response (length: {len(response_text)})")
        
        return jsonify({
            'success': True,
            'query': query,
            'intent': intent,
            'confidence': confidence,
            'response': response_text,
            'properties_found': 0,
            'model_used': 'yes' if vectorizer else 'no'
        })
        
    except Exception as e:
        print(f"💥 ERROR in chat endpoint: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': str(e),
            'response': "Server error occurred"
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'service': 'Debug Chatbot',
        'model_loaded': model_loaded,
        'intents': model_classes if model_loaded else [],
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 DEBUG CHATBOT STARTING")
    print(f"📊 Model loaded: {model_loaded}")
    print(f"🎯 Available intents: {len(model_classes)}")
    print("🌐 Server: http://localhost:5000")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)