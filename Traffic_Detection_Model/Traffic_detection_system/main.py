# main.py
import os
import sys
import time
import re
from traffic_classifier import TrafficClassifier
from speech_handler import SpeechHandler
from mongodb_handler import MongoDBHandler


class TrafficDetectionSystem:
    def __init__(self, model_path='traffic_model.h5'):
        self.classifier = TrafficClassifier()
        self.speech = SpeechHandler()
        self.model_path = model_path
        self.mongodb = MongoDBHandler()
        
        # Try to connect to MongoDB Atlas
        print("\n" + "="*50)
        print("🌐 TRAFFIC DETECTION SYSTEM")
        print("="*50)
        self.mongodb.connect()
        
        # Try to load existing model
        if os.path.exists(model_path):
            self.classifier.load_model(model_path)
            self.speech.speak("Traffic detection system initialized. Model loaded successfully.")
        else:
            self.speech.speak("Traffic detection system initialized. No pre-trained model found.")
    
    def speak_with_fix(self, text):
        """Helper method to speak with error handling."""
        try:
            self.speech.speak(text)
        except Exception as e:
            print(f"[SPEAKING]: {text}")
            print(f"[WARNING] Speech error: {e}")
    
    def train_new_model(self, data_dir='data'):
        """Train a new model from data directory."""
        self.speak_with_fix("Starting model training. This may take several minutes.")
        
        print("\nLoading dataset...")
        X, y = self.classifier.load_dataset(data_dir)
        
        if len(X) == 0:
            self.speech.speak("No training data found. Please add images to the data folder.")
            return None
        
        print(f"Loaded {len(X)} images")
        
        # Build and train model
        self.classifier.build_model()
        metrics, splits = self.classifier.train(X, y, test_size=0.2, epochs=50)
        
        # Save model
        self.classifier.save_model(self.model_path)
        
        # Save metrics to MongoDB
        if self.mongodb.connected:
            self.mongodb.save_model_metrics(
                model_name='TrafficClassifier',
                accuracy=metrics.get('accuracy', 0),
                f1_score=metrics.get('f1_score', 0),
                roc_auc=metrics.get('roc_auc', 0),
                training_samples=len(X),
                epochs=50
            )
        
        # Announce results
        f1 = metrics['f1_score']
        roc = metrics['roc_auc']
        self.speak_with_fix(f"Training complete. F1 score is {f1:.2f}. ROC AUC score is {roc:.2f}.")
        
        return metrics
    
    def analyze_image(self, image_path):
        """Analyze a single image and announce results."""
        if self.classifier.model is None:
            self.speech.speak("No model loaded. Please train a model first.")
            return None
        
        # Fix path if needed
        if not os.path.exists(image_path):
            # Try adding data/ prefix if it's just a filename
            if os.path.exists(f'data/heavy/{image_path}'):
                image_path = f'data/heavy/{image_path}'
            elif os.path.exists(f'data/low/{image_path}'):
                image_path = f'data/low/{image_path}'
        
        try:
            traffic_level, confidence, probability = self.classifier.predict(image_path)
            
            print(f"\n{'='*50}")
            print(f"PREDICTION RESULT")
            print(f"{'='*50}")
            print(f"Image: {image_path}")
            print(f"Traffic Level: {traffic_level}")
            print(f"Confidence: {confidence:.2%}")
            print(f"Probability: {probability:.4f}")
            print(f"{'='*50}\n")
            
            # Save to MongoDB
            if self.mongodb.connected:
                self.mongodb.save_analysis_result(
                    image_path=image_path,
                    traffic_level=traffic_level,
                    confidence=confidence,
                    probability=probability,
                    metadata={'source': 'analysis_mode'}
                )
                
                # Also save to traffic data collection
                self.mongodb.save_traffic_data(
                    traffic_level=traffic_level,
                    confidence=confidence,
                    location=os.path.dirname(image_path),
                    additional_data={'image': os.path.basename(image_path)}
                )
            
            # Announce result via speech
            self.speech.announce_traffic(traffic_level, confidence)
            
            return traffic_level, confidence
            
        except Exception as e:
            error_msg = f"Error analyzing image: {str(e)}"
            print(error_msg)
            self.speech.speak("Could not analyze the image. Please check the file path.")
            return None
    
    def check_model_status(self):
        """Check and announce model status."""
        if self.classifier.model is None:
            self.speak_with_fix("No model is currently loaded. Please train a model first.")
        else:
            total_params = self.classifier.model.count_params()
            self.speak_with_fix(f"Model is loaded and ready. It has {total_params:,} parameters.")
            print(f"\n[STATUS] Model loaded with {total_params:,} parameters")
            
            heavy_count = len([f for f in os.listdir('data/heavy') if f.endswith(('.jpg', '.png', '.jpeg'))]) if os.path.exists('data/heavy') else 0
            low_count = len([f for f in os.listdir('data/low') if f.endswith(('.jpg', '.png', '.jpeg'))]) if os.path.exists('data/low') else 0
            
            print(f"[STATUS] Dataset: {heavy_count} heavy images, {low_count} low images")
            self.speak_with_fix(f"Dataset has {heavy_count} heavy and {low_count} low traffic images.")
            
            # Show MongoDB stats if connected
            if self.mongodb.connected:
                stats = self.mongodb.get_traffic_statistics()
                if stats and stats['total_analyses'] > 0:
                    print(f"\n[DB STATS] Total analyses: {stats['total_analyses']}")
                    print(f"[DB STATS] Heavy: {stats['heavy_count']}, Low: {stats['low_count']}")
                    self.speak_with_fix(f"Database shows {stats['total_analyses']} total analyses.")
    
    def extract_image_path_from_voice(self, voice_text):
        """Extract image path from voice command."""
        voice_text = voice_text.lower().strip()
        
        # Check for numbers in the command
        numbers = re.findall(r'\d+', voice_text)
        
        # Handle heavy images
        if 'heavy' in voice_text:
            heavy_dir = 'data/heavy'
            if os.path.exists(heavy_dir):
                heavy_images = [f for f in os.listdir(heavy_dir) if f.endswith('.jpg')]
                if numbers:
                    for img in heavy_images:
                        if numbers[0] in img:
                            return f'data/heavy/{img}'
                if heavy_images:
                    return f'data/heavy/{heavy_images[0]}'
        
        # Handle low images
        if 'low' in voice_text:
            low_dir = 'data/low'
            if os.path.exists(low_dir):
                low_images = [f for f in os.listdir(low_dir) if f.endswith('.jpg')]
                if numbers:
                    for img in low_images:
                        if numbers[0] in img:
                            return f'data/low/{img}'
                if low_images:
                    return f'data/low/{low_images[0]}'
        
        # Handle direct filenames
        if voice_text.endswith('.jpg'):
            if os.path.exists(f'data/heavy/{voice_text}'):
                return f'data/heavy/{voice_text}'
            elif os.path.exists(f'data/low/{voice_text}'):
                return f'data/low/{voice_text}'
        
        return None
    
    def handle_analyze_command(self, command):
        """Handle analyze commands with voice path extraction."""
        self.speak_with_fix("Say a number 1 to 10 to select an image.")
        
        print("\n" + "="*50)
        print("📸 IMAGE SELECTION")
        print("="*50)
        
        heavy_images = [f for f in os.listdir('data/heavy') if f.endswith('.jpg')][:5]
        low_images = [f for f in os.listdir('data/low') if f.endswith('.jpg')][:5]
        
        print("\n📁 Heavy Traffic Images (1-5):")
        for i, img in enumerate(heavy_images[:5], 1):
            print(f"   {i}. {img}")
        
        print("\n📁 Low Traffic Images (6-10):")
        for i, img in enumerate(low_images[:5], 1):
            print(f"   {i+5}. {img}")
        
        print("\n[🎤] Say a number (one, two, three, etc.) or type the path:")
        
        # Map spoken numbers to digits
        number_map = {
            'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
            'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
            '1': '1', '2': '2', '3': '3', '4': '4', '5': '5',
            '6': '6', '7': '7', '8': '8', '9': '9', '10': '10'
        }
        
        # Get voice input
        voice_path = self.speech.get_voice_command(timeout=5)
        
        if voice_path:
            # Try to extract number from voice
            number = None
            for word, num in number_map.items():
                if word in voice_path.lower():
                    number = int(num)
                    break
            
            if number and 1 <= number <= 5 and number-1 < len(heavy_images):
                # Heavy image
                image_path = f'data/heavy/{heavy_images[number-1]}'
                print(f"\n✅ Selected: {heavy_images[number-1]}")
                self.analyze_image(image_path)
            elif number and 6 <= number <= 10:
                # Low image
                low_index = number - 6
                if low_index < len(low_images):
                    image_path = f'data/low/{low_images[low_index]}'
                    print(f"\n✅ Selected: {low_images[low_index]}")
                    self.analyze_image(image_path)
                else:
                    self.speak_with_fix("Number out of range. Please try again.")
                    print("❌ Number out of range.")
            else:
                # Try to extract path
                extracted_path = self.extract_image_path_from_voice(voice_path)
                if extracted_path:
                    self.analyze_image(extracted_path)
                else:
                    self.speak_with_fix("Could not understand. Please type the path.")
                    print("\n📝 Type the image path:")
                    image_path = input("Enter image path: ").strip()
                    if image_path:
                        self.analyze_image(image_path)
                    else:
                        self.speak_with_fix("No image path provided.")
        else:
            self.speak_with_fix("No voice detected. Please type the image path.")
            print("\n📝 Type the image path:")
            image_path = input("Enter image path: ").strip()
            if image_path:
                self.analyze_image(image_path)
            else:
                self.speak_with_fix("No image path provided.")
    
    def handle_test_command(self):
        """Test the system with sample images."""
        self.speak_with_fix("Testing with sample images.")
        
        # Show available test images
        print("\n" + "="*50)
        print("🧪 TEST MODE")
        print("="*50)
        
        test_images = []
        
        # Get heavy test images
        heavy_dir = 'data/heavy'
        if os.path.exists(heavy_dir):
            heavy_images = [f for f in os.listdir(heavy_dir) if f.endswith('.jpg')][:3]
            for img in heavy_images:
                test_images.append(('data/heavy/' + img, 'heavy'))
        
        # Get low test images
        low_dir = 'data/low'
        if os.path.exists(low_dir):
            low_images = [f for f in os.listdir(low_dir) if f.endswith('.jpg')][:3]
            for img in low_images:
                test_images.append(('data/low/' + img, 'low'))
        
        if test_images:
            self.speak_with_fix(f"Found {len(test_images)} test images. Starting analysis.")
            print(f"\n📸 Testing {len(test_images)} images...")
            
            for i, (img_path, expected) in enumerate(test_images, 1):
                print(f"\n[{i}/{len(test_images)}] Analyzing: {os.path.basename(img_path)}")
                self.analyze_image(img_path)
                time.sleep(1.5)  # Brief pause between tests
            
            # Summary
            self.speak_with_fix("Test complete.")
            print("\n✅ Test complete!")
            
            # Show database summary
            if self.mongodb.connected:
                stats = self.mongodb.get_traffic_statistics()
                print(f"\n📊 Database Summary:")
                print(f"   Total analyses: {stats.get('total_analyses', 0)}")
                print(f"   Heavy traffic: {stats.get('heavy_count', 0)}")
                print(f"   Low traffic: {stats.get('low_count', 0)}")
        else:
            self.speak_with_fix("No test images found. Please add images to data folder.")
            print("\n❌ No test images found in data/heavy/ or data/low/")
            
            # Show available directories
            if os.path.exists('data'):
                print("\n📁 Available directories in data:")
                for item in os.listdir('data'):
                    if os.path.isdir(os.path.join('data', item)):
                        count = len([f for f in os.listdir(os.path.join('data', item)) if f.endswith('.jpg')])
                        print(f"   - {item}: {count} images")
    
    def handle_train_command(self):
        """Handle train command with confirmation and options."""
        print("\n" + "="*50)
        print("🚀 TRAINING MODE")
        print("="*50)
        
        # Check if model already exists
        model_exists = os.path.exists(self.model_path)
        
        if model_exists:
            self.speak_with_fix("A model already exists. Do you want to retrain? Say yes or no.")
            print("\n⚠️  A model already exists. Retrain? (yes/no)")
            
            # Get confirmation
            confirmation = self.speech.get_voice_command(timeout=5)
            
            if confirmation is None:
                print("No response. Using default data directory...")
                data_dir = 'data'
            elif 'yes' in confirmation.lower() or 'yeah' in confirmation.lower() or 'sure' in confirmation.lower() or 'proceed' in confirmation.lower() or 'start' in confirmation.lower() or 'go' in confirmation.lower():
                print("✅ Proceeding with retraining...")
                data_dir = 'data'
            else:
                self.speak_with_fix("Training cancelled.")
                print("Training cancelled.")
                return
        else:
            self.speak_with_fix("No existing model found. Starting training with default data directory.")
            data_dir = 'data'
        
        # Ask if they want to use custom data directory
        self.speak_with_fix("Use default data directory? Say yes or no.")
        print("\n📁 Use default data directory 'data'? (yes/no)")
        
        use_default = self.speech.get_voice_command(timeout=5)
        
        if use_default and ('no' in use_default.lower() or 'nope' in use_default.lower() or 'not' in use_default.lower() or 'negative' in use_default.lower() or 'nah' in use_default.lower()):
            self.speak_with_fix("Please type the data directory path.")
            print("\n📁 Enter custom data directory path:")
            data_dir = input("Data directory path: ").strip()
            if not data_dir:
                data_dir = 'data'
                print("Using default: data")
        else:
            data_dir = 'data'
            print("Using default data directory: data")
        
        # Final confirmation
        print(f"\n📂 Training with data directory: {data_dir}")
        self.speak_with_fix(f"Training with data directory. Confirm? Say yes to start.")
        print("Confirm training? (yes/no)")
        
        confirm = self.speech.get_voice_command(timeout=5)
        
        if confirm and ('yes' in confirm.lower() or 'yeah' in confirm.lower() or 'start' in confirm.lower() or 'go' in confirm.lower() or 'proceed' in confirm.lower()):
            self.speak_with_fix("Starting model training. This may take several minutes.")
            self.train_new_model(data_dir)
        else:
            self.speak_with_fix("Training cancelled.")
            print("Training cancelled.")
    
    def show_database_stats(self):
        """Show database statistics."""
        if not self.mongodb.connected:
            self.speak_with_fix("Not connected to database.")
            return
        
        stats = self.mongodb.get_traffic_statistics()
        
        print("\n" + "="*50)
        print("📊 DATABASE STATISTICS")
        print("="*50)
        print(f"Total Analyses: {stats.get('total_analyses', 0)}")
        print(f"Heavy Traffic: {stats.get('heavy_count', 0)} ({stats.get('heavy_percentage', 0):.1f}%)")
        print(f"Low Traffic: {stats.get('low_count', 0)} ({100 - stats.get('heavy_percentage', 0):.1f}%)")
        print(f"Average Confidence: {stats.get('avg_confidence', 0):.2%}")
        print("="*50)
        
        self.speak_with_fix(f"Database has {stats.get('total_analyses', 0)} total analyses. "
                           f"{stats.get('heavy_count', 0)} heavy, "
                           f"{stats.get('low_count', 0)} low traffic.")
    
    def export_database_data(self):
        """Export database data to CSV."""
        if not self.mongodb.connected:
            self.speak_with_fix("Not connected to database.")
            return
        
        self.speak_with_fix("Exporting data to CSV files.")
        
        # Export analysis results
        if self.mongodb.export_to_csv('analysis_results'):
            print("✅ Exported analysis_results.csv")
        
        # Export model metrics
        if self.mongodb.export_to_csv('model_metrics'):
            print("✅ Exported model_metrics.csv")
        
        # Export traffic data
        if self.mongodb.export_to_csv('traffic_data'):
            print("✅ Exported traffic_data.csv")
        
        self.speak_with_fix("Export complete. CSV files saved in current directory.")
    
    def run_interactive_mode(self):
        """Run in simplified interactive voice-controlled mode."""
        self.speak_with_fix("Voice mode active. Say a command.")
        print("\n" + "="*50)
        print("🎤 VOICE MODE - SIMPLE COMMANDS")
        print("="*50)
        print("\n🎤 Say one of these words:")
        print("   • 'info' - Show model status")
        print("   • 'test' - Test with sample images")
        print("   • 'scan' - Analyze an image")
        print("   • 'train' - Train new model")
        print("   • 'stats' - Show database statistics")
        print("   • 'export' - Export data to CSV")
        print("   • 'stop' - Cancel")
        print("   • 'bye' - Exit voice mode")
        print("\n📸 For images, say numbers 1-10")
        print("="*50)
        
        while True:
            command = self.speech.get_voice_command(timeout=8)
            
            if command is None:
                continue
            
            cmd_lower = command.lower()
            print(f"[DEBUG] Command: '{cmd_lower}'")
            
            # Exit commands
            if cmd_lower in ['bye', 'goodbye', 'exit', 'quit']:
                self.speak_with_fix("Goodbye!")
                break
            
            # Stop/Cancel
            elif cmd_lower in ['stop', 'cancel']:
                self.speak_with_fix("Cancelled.")
                continue
            
            # Info/Status
            elif cmd_lower in ['info', 'status', 'model']:
                self.check_model_status()
            
            # Test
            elif cmd_lower in ['test', 'try', 'sample']:
                self.handle_test_command()
            
            # Train
            elif cmd_lower in ['train', 'new', 'retrain', 'build']:
                self.handle_train_command()
            
            # Database statistics
            elif cmd_lower in ['stats', 'statistics', 'db stats']:
                self.show_database_stats()
            
            # Export data
            elif cmd_lower in ['export', 'save', 'backup']:
                self.export_database_data()
            
            # Scan/Analyze
            elif cmd_lower in ['scan', 'analyze', 'check', 'examine', 'detect']:
                self.handle_analyze_command(command)
            
            # Numbers for quick image selection
            elif cmd_lower in ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']:
                self.handle_analyze_command(command)
            
            else:
                self.speak_with_fix("Say info, test, scan, train, stats, or bye.")
                print(f"[UNKNOWN]: '{cmd_lower}'")


def main():
    """Main entry point."""
    system = TrafficDetectionSystem()
    
    print("\n" + "="*50)
    print("TRAFFIC DETECTION SYSTEM")
    print("="*50)
    print("\nOptions:")
    print("1. Train new model")
    print("2. Analyze image")
    print("3. Voice mode")
    print("4. Exit")
    
    while True:
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            data_dir = input("Enter data directory path [data]: ").strip() or 'data'
            system.train_new_model(data_dir)
        
        elif choice == '2':
            image_path = input("Enter image path: ").strip()
            system.analyze_image(image_path)
        
        elif choice == '3':
            system.run_interactive_mode()
        
        elif choice == '4':
            # Close MongoDB connection before exit
            if hasattr(system, 'mongodb') and system.mongodb.connected:
                system.mongodb.close()
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please enter 1-4.")


if __name__ == "__main__":
    main()