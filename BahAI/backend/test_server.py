import requests
import json

def test_chatbot():
    url = "http://localhost:5000/api/chat"
    
    test_queries = [
        "properties near schools",
        "show me properties that accept bank financing",
        "find apartments in batangas city"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Testing: '{query}'")
        
        try:
            response = requests.post(
                url,
                json={"query": query},
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Success! Intent: {result.get('intent')}")
                print(f"Response: {result.get('response', '')[:100]}...")
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
                
        except requests.ConnectionError:
            print("❌ Cannot connect to server. Is chatbot_backend.py running?")
            print("   Run: python chatbot_backend.py")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_chatbot()