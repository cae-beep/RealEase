# test_fixed.py
import requests
import json

def test_queries():
    url = "http://localhost:5000/api/chat"
    
    test_cases = [
        ("Find ready to move in properties for students in Batangas City", "find_ready_property"),
        ("show me properties under 2 million", "find_property_with_criteria"),
        ("houses near schools", "find_near_landmark"),
        ("properties for family needs", "find_property_for_need"),
        ("apartments with swimming pool", "find_with_feature")
    ]
    
    for query, expected in test_cases:
        print(f"\n🔍 Testing: '{query}'")
        response = requests.post(url, json={'query': query})
        
        if response.status_code == 200:
            data = response.json()
            actual = data.get('intent')
            
            if actual == expected:
                print(f"✅ Correct! Intent: {actual}")
            else:
                print(f"❌ Wrong! Got: {actual}, Expected: {expected}")
            
            print(f"Response preview: {data.get('response', '')[:150]}...")
        else:
            print(f"❌ HTTP Error: {response.status_code}")

if __name__ == "__main__":
    test_queries()