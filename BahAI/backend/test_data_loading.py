# test_data_loading.py
import sys
sys.path.append('.')  # Add current directory to path

# Simulate loading the backend
import json
import os

def test_member_data():
    member_folders = ['member1', 'member2', 'member3']
    data_folder = 'data'
    
    print("🔍 Checking member data files...")
    
    for member in member_folders:
        member_path = os.path.join(data_folder, member, 'training_data.json')
        if os.path.exists(member_path):
            with open(member_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            samples = data.get('training_samples', [])
            print(f"\n📂 {member}:")
            print(f"   📊 Samples: {len(samples)}")
            
            # Count intents
            intents = {}
            for sample in samples:
                intent = sample.get('intent', 'unknown')
                intents[intent] = intents.get(intent, 0) + 1
            
            print(f"   🎯 Intents found: {list(intents.keys())}")
            for intent, count in intents.items():
                print(f"      • {intent}: {count} samples")
        else:
            print(f"\n❌ {member}: File not found at {member_path}")

if __name__ == "__main__":
    test_member_data()