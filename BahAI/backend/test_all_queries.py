import requests
import json
import time

def test_all_member_queries():
    url = "http://localhost:5000/api/chat"
    
    test_cases = [
        # Member1 queries (should work)
        ("properties that accept bank financing", "financing"),
        ("show me properties that accept pag-ibig financing", "financing"),
        ("find apartments in batangas city", "find_property"),
        ("tell me about lipa city", "location_info"),
        
        # Member2 queries (problematic)
        ("show me houses under 3M with 3 bedrooms", "find_property_with_criteria"),
        ("properties near schools", "find_near_landmark"),
        ("ready to move in properties for family", "find_ready_property"),
        ("apartments under 2M with 2 bedrooms", "find_property_with_criteria"),
        ("houses near malls", "find_near_landmark"),
        ("available now apartments for students", "find_ready_property"),
    ]
    
    print("🧪 TESTING ALL MEMBER QUERIES")
    print("=" * 70)
    
    results = []
    
    for query, expected_intent in test_cases:
        print(f"\n🔍 Query: '{query}'")
        print(f"   Expected intent: {expected_intent}")
        
        try:
            start_time = time.time()
            response = requests.post(
                url,
                json={"query": query},
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                actual_intent = result.get('intent')
                confidence = result.get('confidence', 0)
                entities = result.get('entities', {})
                response_text = result.get('response', '')
                properties_found = result.get('properties_found', 0)
                
                print(f"   ✅ Response received in {elapsed:.2f}s")
                print(f"   🎯 Intent: {actual_intent} (confidence: {confidence:.1%})")
                print(f"   🏷️ Entities: {entities}")
                print(f"   🏠 Properties found: {properties_found}")
                
                if actual_intent == expected_intent:
                    print(f"   ✓ CORRECT INTENT")
                else:
                    print(f"   ✗ WRONG INTENT (expected: {expected_intent})")
                
                # Show first 150 chars of response
                if response_text:
                    preview = response_text[:150] + "..." if len(response_text) > 150 else response_text
                    print(f"   💬 Response preview: {preview}")
                
                results.append({
                    'query': query,
                    'expected': expected_intent,
                    'actual': actual_intent,
                    'correct': actual_intent == expected_intent,
                    'time': elapsed,
                    'properties': properties_found
                })
            else:
                print(f"   ❌ HTTP Error {response.status_code}: {response.text[:100]}")
                results.append({
                    'query': query,
                    'error': f"HTTP {response.status_code}",
                    'time': elapsed
                })
                
        except requests.exceptions.Timeout:
            print(f"   ⏰ TIMEOUT (took > 10 seconds)")
            results.append({
                'query': query,
                'error': 'Timeout',
                'time': 10
            })
        except requests.exceptions.ConnectionError:
            print(f"   🔌 CONNECTION ERROR - Is server running?")
            break
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({
                'query': query,
                'error': str(e)
            })
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    if results:
        total = len(results)
        correct = sum(1 for r in results if r.get('correct', False))
        member1_tests = [r for r in results if r['query'].startswith(('properties that', 'find', 'tell me'))]
        member2_tests = [r for r in results if r['query'].startswith(('show me', 'properties near', 'ready to', 'apartments', 'houses near', 'available now'))]
        
        print(f"Total tests: {total}")
        print(f"Correct intents: {correct}/{total} ({correct/total*100:.1f}%)")
        
        if member1_tests:
            member1_correct = sum(1 for r in member1_tests if r.get('correct', False))
            print(f"\nMember1 queries: {member1_correct}/{len(member1_tests)} correct")
        
        if member2_tests:
            member2_correct = sum(1 for r in member2_tests if r.get('correct', False))
            print(f"Member2 queries: {member2_correct}/{len(member2_tests)} correct")
            
            # Show member2 failures
            failures = [r for r in member2_tests if not r.get('correct', True)]
            if failures:
                print(f"\n❌ Member2 failures:")
                for f in failures:
                    print(f"  - '{f['query']}'")
                    if 'actual' in f:
                        print(f"    Expected: {f['expected']}, Got: {f['actual']}")
    
    return results

if __name__ == "__main__":
    test_all_member_queries()