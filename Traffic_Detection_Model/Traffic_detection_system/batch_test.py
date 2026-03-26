import os
from traffic_classifier import TrafficClassifier

def batch_test():
    """Test multiple images"""
    
    # Load model
    print("Loading model...")
    classifier = TrafficClassifier()
    classifier.load_model('traffic_model.h5')
    print("✓ Model loaded\n")
    
    # Test heavy images
    print("="*60)
    print("HEAVY TRAFFIC IMAGES")
    print("="*60)
    
    heavy_test = [
        'data/heavy/heavy_000_30cars.jpg',
        'data/heavy/heavy_001_34cars.jpg',
        'data/heavy/heavy_005_25cars.jpg',
        'data/heavy/heavy_010_22cars.jpg'
    ]
    
    for img_path in heavy_test:
        if os.path.exists(img_path):
            try:
                traffic, confidence, prob = classifier.predict(img_path)
                print(f"\n📷 {os.path.basename(img_path)}")
                print(f"   Result: {traffic}")
                print(f"   Confidence: {confidence:.2%}")
                print(f"   Probability: {prob:.4f}")
            except Exception as e:
                print(f"\n✗ {img_path}: {e}")
        else:
            print(f"\n✗ File not found: {img_path}")
    
    # Test low images
    print("\n" + "="*60)
    print("LOW TRAFFIC IMAGES")
    print("="*60)
    
    low_test = [
        'data/low/low_000_4cars.jpg',
        'data/low/low_005_2cars.jpg',
        'data/low/low_007_2cars.jpg',
        'data/low/low_010_8cars.jpg'
    ]
    
    for img_path in low_test:
        if os.path.exists(img_path):
            try:
                traffic, confidence, prob = classifier.predict(img_path)
                print(f"\n📷 {os.path.basename(img_path)}")
                print(f"   Result: {traffic}")
                print(f"   Confidence: {confidence:.2%}")
                print(f"   Probability: {prob:.4f}")
            except Exception as e:
                print(f"\n✗ {img_path}: {e}")
        else:
            print(f"\n✗ File not found: {img_path}")

if __name__ == "__main__":
    batch_test()