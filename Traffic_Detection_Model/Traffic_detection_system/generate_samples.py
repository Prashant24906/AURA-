import cv2
import os
import random

def generate_simple_samples():
    """Generate simple traffic images"""
    
    # Create directories
    os.makedirs('data/heavy', exist_ok=True)
    os.makedirs('data/low', exist_ok=True)
    
    # Colors
    SKY_BLUE = (135, 206, 235)
    ROAD_GRAY = (100, 100, 100)
    RED = (0, 0, 255)
    BLUE = (255, 0, 0)
    GREEN = (0, 255, 0)
    ORANGE = (0, 165, 255)
    YELLOW = (0, 255, 255)
    
    # Generate heavy traffic images
    print("Generating heavy traffic images (20 images)...")
    for i in range(20):
        # Create image
        img = cv2.imread('data/heavy/heavy_template.jpg') if os.path.exists('data/heavy/heavy_template.jpg') else None
        
        if img is None:
            # Create new image
            img = np.zeros((150, 150, 3), dtype=np.uint8)
            img[0:75, :] = SKY_BLUE
            img[75:150, :] = ROAD_GRAY
        
        # Add many cars
        num_cars = random.randint(25, 40)
        for _ in range(num_cars):
            x = random.randint(10, 125)
            y = random.randint(80, 140)
            color = random.choice([RED, BLUE])
            cv2.rectangle(img, (x, y), (x+12, y+8), color, -1)
        
        cv2.imwrite(f'data/heavy/heavy_{i:03d}.jpg', img)
        if (i+1) % 5 == 0:
            print(f"  Progress: {i+1}/20")
    
    # Generate low traffic images
    print("\nGenerating low traffic images (20 images)...")
    for i in range(20):
        # Create image
        img = np.zeros((150, 150, 3), dtype=np.uint8)
        img[0:75, :] = SKY_BLUE
        img[75:150, :] = ROAD_GRAY
        
        # Add few cars
        num_cars = random.randint(3, 10)
        for _ in range(num_cars):
            x = random.randint(10, 125)
            y = random.randint(80, 140)
            color = random.choice([GREEN, ORANGE, YELLOW])
            cv2.rectangle(img, (x, y), (x+12, y+8), color, -1)
        
        cv2.imwrite(f'data/low/low_{i:03d}.jpg', img)
        if (i+1) % 5 == 0:
            print(f"  Progress: {i+1}/20")
    
    print("\n✅ Images generated successfully!")
    print(f"Heavy images: {len([f for f in os.listdir('data/heavy') if f.endswith('.jpg')])}")
    print(f"Low images: {len([f for f in os.listdir('data/low') if f.endswith('.jpg')])}")

if __name__ == "__main__":
    import numpy as np
    generate_simple_samples()