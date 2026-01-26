# minimal_chatbot.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import random

app = Flask(__name__)
CORS(app)

# Simple intent detection
def detect_intent(query):
    query_lower = query.lower()
    
    if 'financing' in query_lower or 'loan' in query_lower or 'pag-ibig' in query_lower:
        return 'financing'
    elif 'near' in query_lower:
        return 'find_near_landmark'
    elif 'under' in query_lower or 'below' in query_lower:
        return 'find_property_with_criteria'
    elif 'ready' in query_lower or 'available now' in query_lower:
        return 'find_ready_property'
    elif 'find' in query_lower or 'show me' in query_lower:
        return 'find_property'
    elif 'about' in query_lower or 'tell me' in query_lower:
        return 'location_info'
    else:
        return 'unknown'

# Simple response generation
def generate_response(intent, query):
    responses = {
        'financing': f"🏦 **Financing Information**\n\nI understand you're asking about financing for: '{query}'. I can help with bank loans, Pag-IBIG financing, and in-house developer financing options.",
        'find_near_landmark': f"📍 **Properties Near Landmarks**\n\nLooking for properties near landmarks? For '{query}', I can help you find homes close to schools, malls, hospitals, and other amenities in Batangas.",
        'find_property_with_criteria': f"🔍 **Properties Matching Criteria**\n\nFor '{query}', I can search for properties based on your budget and requirements. Would you like me to look for specific property types in certain locations?",
        'find_ready_property': f"🚚 **Ready-to-Move Properties**\n\nI see you want ready-to-move properties. For '{query}', I can find available units that are ready for immediate occupancy.",
        'find_property': f"🏠 **Property Search**\n\nSearching for properties? For '{query}', I can help you find apartments, houses, condos, and other properties in Batangas.",
        'location_info': f"📍 **Location Information**\n\nWant to know about a location? For '{query}', I can provide details about different areas in Batangas including lifestyle, prices, and amenities.",
        'unknown': f"🤔 **General Assistance**\n\nI understand you're looking for property information. Could you provide more details about what you need?"
    }
    
    return responses.get(intent, responses['unknown'])

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        query = data.get('query', '').strip()
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        print(f"💬 Received query: '{query}'")
        
        # Simple intent detection
        intent = detect_intent(query)
        
        # Generate response
        response_text = generate_response(intent, query)
        
        # Return result
        return jsonify({
            'success': True,
            'query': query,
            'intent': intent,
            'response': response_text,
            'properties_found': 0,
            'properties': []
        })
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'response': "I encountered an error. Please try again."
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'service': 'Minimal Chatbot',
        'version': '1.0',
        'message': 'Working without model or Firebase'
    })

if __name__ == '__main__':
    print("🚀 Starting MINIMAL chatbot server...")
    print("🌐 Endpoint: POST /api/chat")
    print("🔧 No model, no Firebase - just simple responses")
    app.run(host='0.0.0.0', port=5001, debug=True)