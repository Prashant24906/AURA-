import cv2
import numpy as np
import os
import random

def generate_traffic_images():
    """Generate synthetic traffic images for training"""
    
    # Create directories if they don't exist
    os.makedirs('data/heavy', exist_ok=True)
    os.makedirs('data/low', exist_ok=True)
    
    # Colors (BGR format for OpenCV)
    SKY_BLUE = (135, 206, 235)
    ROAD_GRAY = (100, 100, 100)
    RED = (0, 0, 255)
    BLUE = (255, 0, 0)
    GREEN = (0, 255, 0)
    YELLOW = (0, 255, 255)
    ORANGE = (0, 165, 255)
    
    # Generate 30 heavy traffic images
    print("Generating heavy traffic images...")
    for i in range(30):
        # Create base image
        img = np.zeros((150, 150, 3), dtype=np.uint8)
        img[0:75, :] = SKY_BLUE  # Sky
        img[75:150, :] = ROAD_GRAY  # Road
        
        # Add many cars for heavy traffic
        num_cars = random.randint(20, 35)
        for _ in range(num_cars):
            x = random.randint(10, 125)
            y = random.randint(80, 140)
            color = random.choice([RED, BLUE, RED])  # Mostly red/blue for heavy
            cv2.rectangle(img, (x, y), (x+15, y+10), color, -1)
            # Add wheels
            cv2.circle(img, (x+3, y+10), 2, (0, 0, 0), -1)
            cv2.circle(img, (x+12, y+10), 2, (0, 0, 0), -1)
        
        # Add some noise for realism
        noise = np.random.randint(0, 25, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
        
        # Save image
        filename = f'heavy_{i:03d}_{num_cars}cars.jpg'
        cv2.imwrite(f'data/heavy/{filename}', img)
        
        if (i + 1) % 10 == 0:
            print(f"  Generated {i+1}/30 heavy images")
    
    # Generate 30 low traffic images
    print("\nGenerating low traffic images...")
    for i in range(30):
        # Create base image
        img = np.zeros((150, 150, 3), dtype=np.uint8)
        img[0:75, :] = SKY_BLUE  # Sky
        img[75:150, :] = ROAD_GRAY  # Road
        
        # Add few cars for low traffic
        num_cars = random.randint(2, 8)
        for _ in range(num_cars):
            x = random.randint(10, 125)
            y = random.randint(80, 140)
            color = random.choice([GREEN, YELLOW, ORANGE])  # Bright colors for low
            cv2.rectangle(img, (x, y), (x+15, y+10), color, -1)
            # Add wheels
            cv2.circle(img, (x+3, y+10), 2, (0, 0, 0), -1)
            cv2.circle(img, (x+12, y+10), 2, (0, 0, 0), -1)
        
        # Add less noise for low traffic
        noise = np.random.randint(0, 15, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
        
        # Save image
        filename = f'low_{i:03d}_{num_cars}cars.jpg'
        cv2.imwrite(f'data/low/{filename}', img)
        
        if (i + 1) % 10 == 0:
            print(f"  Generated {i+1}/30 low images")
    
    # Count generated images
    heavy_count = len([f for f in os.listdir('data/heavy') if f.endswith('.jpg')])
    low_count = len([f for f in os.listdir('data/low') if f.endswith('.jpg')])
    
    print("\n" + "="*50)
    print("✅ IMAGES GENERATED SUCCESSFULLY!")
    print("="*50)
    print(f"Heavy traffic images: {heavy_count}")
    print(f"Low traffic images: {low_count}")
    print("="*50)
    print("\nYou can now train the model using: python main.py")
    print("Then select option 1 and enter 'data' as the directory")

if __name__ == "__main__":
    generate_traffic_images()