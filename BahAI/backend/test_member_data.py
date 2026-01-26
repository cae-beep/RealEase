import json
import os

def check_member_coverage():
    """Check which members have which queries"""
    members = ['member1', 'member2', 'member3']
    
    for member in members:
        filepath = f'data/{member}/training_data.json'
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"\n=== {member} ===")
            print(f"Assigned Questions: {data.get('assigned_questions', [])}")
            
            # Check for "near schools" samples
            training_samples = data.get('training_samples', [])
            near_school_samples = []
            
            for sample in training_samples:
                query = sample.get('query', '').lower()
                variations = sample.get('variations', [])
                
                if 'near school' in query or 'near schools' in query:
                    near_school_samples.append(sample['query'])
                
                for variation in variations:
                    if isinstance(variation, str) and ('near school' in variation.lower() or 'near schools' in variation.lower()):
                        near_school_samples.append(variation)
            
            if near_school_samples:
                print(f"Found {len(near_school_samples)} 'near schools' samples:")
                for sample in near_school_samples[:5]:
                    print(f"  - {sample}")
            else:
                print("No 'near schools' samples found")
            
            # Check intent for member2
            print(f"Intents in {member}:")
            intents = set()
            for sample in training_samples[:20]:
                intents.add(sample.get('intent', 'unknown'))
            print(f"  {', '.join(intents)}")

if __name__ == "__main__":
    check_member_coverage()