# speech_handler.py
import pyttsx3
import speech_recognition as sr
import time
import threading

class SpeechHandler:
    def __init__(self):
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Adjust for ambient noise
        try:
            with self.microphone as source:
                print("[INFO] Calibrating microphone...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                print("[INFO] Microphone ready!")
        except Exception as e:
            print(f"[WARNING] Microphone setup failed: {e}")
        
        # Initialize TTS engine separately for each use to avoid blocking
        self._init_tts_engine()
    
    def _init_tts_engine(self):
        """Initialize TTS engine (called each time to avoid blocking)."""
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 0.9)
            
            # Set voice if available
            voices = engine.getProperty('voices')
            if voices:
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        engine.setProperty('voice', voice.id)
                        break
            return engine
        except Exception as e:
            print(f"[ERROR] TTS init failed: {e}")
            return None
    
    def speak(self, text):
        """Convert text to speech - creates new engine each time."""
        print(f"[SPEAKING]: {text}")
        try:
            engine = self._init_tts_engine()
            if engine:
                engine.say(text)
                engine.runAndWait()
                engine.stop()
        except Exception as e:
            print(f"[SPEAKING ERROR]: {e}")
    
    def announce_traffic(self, traffic_level, confidence):
        """Announce traffic detection result with appropriate message."""
        if "HEAVY" in traffic_level.upper():
            self.speak(f"Traffic detected is HEAVY with {confidence:.1%} confidence. Expect delays.")
        elif "LOW" in traffic_level.upper():
            self.speak(f"Traffic detected is LOW with {confidence:.1%} confidence. Roads appear clear.")
        else:
            self.speak(f"Traffic detected: {traffic_level} with {confidence:.1%} confidence.")
    
    def get_voice_command(self, timeout=5):
        """Get voice command from microphone."""
        try:
            with self.microphone as source:
                print("\n[🎤 LISTENING]...")
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=5)
            
            print("[🔄 RECOGNIZING]...")
            command = self.recognizer.recognize_google(audio).lower()
            print(f"[✅ RECOGNIZED]: {command}")
            return command
            
        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            print("[❓ INFO] Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"[❌ ERROR] Recognition error: {e}")
            return None
    
    def show_available_commands(self):
        """Display available voice commands."""
        commands = """
        🎤 Voice Commands:
        
        • "analyze image" - Analyze a traffic image
        • "analyze [path]" - Analyze specific image
        • "status" - Check model status
        • "test" - Test with sample images
        • "help" - Show this help
        • "exit" - Exit voice mode
        """
        print(commands)
        self.speak("Available commands. Say help for details.")