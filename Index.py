import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import nltk
import re
import json
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
except:
    print("NLTK downloads may require manual installation")

class NLPChatbot:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.le = LabelEncoder()
        self.max_sequence_length = 20
        self.vocab_size = 1000
        self.embedding_dim = 50
        self.intents = {}
        self.responses = {}
        
    def create_training_data(self):
        """Create comprehensive training data for the chatbot"""
        print("📚 Creating training data for AI Chatbot...")
        
        # Define intents and patterns
        self.intents = {
            "greeting": {
                "patterns": [
                    "hello", "hi", "hey", "good morning", "good afternoon", 
                    "good evening", "howdy", "what's up", "hello there",
                    "hi there", "hey there", "greetings", "how are you",
                    "how do you do", "nice to meet you"
                ],
                "responses": [
                    "Hello! How can I assist you today?",
                    "Hi there! What can I help you with?",
                    "Hey! Nice to see you. How can I help?",
                    "Greetings! I'm here to assist you.",
                    "Hello! I'm your AI assistant. What do you need help with?"
                ]
            },
            "goodbye": {
                "patterns": [
                    "bye", "goodbye", "see you", "farewell", "take care",
                    "have a good day", "see you later", "bye bye", "until next time",
                    "I'm leaving", "catch you later"
                ],
                "responses": [
                    "Goodbye! Have a great day!",
                    "See you later! Feel free to come back if you have more questions.",
                    "Take care! I'm here if you need me.",
                    "Farewell! It was nice talking to you.",
                    "Bye! Don't hesitate to ask if you need help later."
                ]
            },
            "thanks": {
                "patterns": [
                    "thank you", "thanks", "thank you very much", "thanks a lot",
                    "appreciate it", "thank you so much", "much obliged",
                    "I appreciate your help", "thanks for your assistance"
                ],
                "responses": [
                    "You're welcome!",
                    "Happy to help!",
                    "Glad I could assist you!",
                    "You're very welcome!",
                    "Anytime! That's what I'm here for."
                ]
            },
            "about_bot": {
                "patterns": [
                    "who are you", "what are you", "tell me about yourself",
                    "what can you do", "what is your purpose", "are you a bot",
                    "are you human", "what is your name", "introduce yourself"
                ],
                "responses": [
                    "I'm an AI chatbot created to assist with answering questions and providing information.",
                    "I'm a friendly AI assistant designed to help you with various queries.",
                    "I'm a chatbot powered by NLP and deep learning. I can answer questions and have conversations!",
                    "I'm your AI assistant! I can help with information, answer questions, and chat with you.",
                    "I'm a smart chatbot that uses natural language processing to understand and respond to your messages."
                ]
            },
            "help": {
                "patterns": [
                    "help", "I need help", "can you help me", "what can you do",
                    "how does this work", "I need assistance", "help me please",
                    "what help can you provide", "how can you help me"
                ],
                "responses": [
                    "I can help you with answering questions, providing information, and having conversations!",
                    "I'm here to assist with various topics. Just ask me anything!",
                    "I can answer questions, provide information, and chat with you. What do you need help with?",
                    "I'm designed to help with information and conversations. Feel free to ask me anything!",
                    "I can assist with many topics. Try asking me about technology, science, or just chat!"
                ]
            },
            "technology": {
                "patterns": [
                    "what is ai", "tell me about machine learning", "what is python",
                    "explain neural networks", "what is nlp", "artificial intelligence",
                    "deep learning", "data science", "what is tensorflow",
                    "how does machine learning work"
                ],
                "responses": [
                    "AI (Artificial Intelligence) involves creating intelligent machines that can perform tasks typically requiring human intelligence.",
                    "Machine learning is a subset of AI that enables computers to learn and improve from experience without explicit programming.",
                    "Python is a popular programming language widely used in AI, data science, and web development.",
                    "Neural networks are computing systems inspired by the human brain, used in deep learning for pattern recognition.",
                    "NLP (Natural Language Processing) helps computers understand, interpret, and generate human language."
                ]
            },
            "weather": {
                "patterns": [
                    "what's the weather", "how is the weather", "weather today",
                    "is it raining", "temperature today", "weather forecast",
                    "is it sunny", "what's the temperature", "weather update"
                ],
                "responses": [
                    "I don't have real-time weather data, but you can check your local weather app or website!",
                    "For accurate weather information, I recommend checking a reliable weather service.",
                    "I'm not connected to weather services, but you can find current weather online.",
                    "Weather updates are best obtained from dedicated weather applications.",
                    "I specialize in conversation rather than real-time data like weather."
                ]
            },
            "time": {
                "patterns": [
                    "what time is it", "current time", "what's the time",
                    "time please", "tell me the time", "what is the current time"
                ],
                "responses": [
                    f"The current time is {datetime.now().strftime('%H:%M:%S')}.",
                    f"It's currently {datetime.now().strftime('%I:%M %p')}.",
                    f"Right now it's {datetime.now().strftime('%H:%M')}.",
                    f"The time is {datetime.now().strftime('%I:%M:%S %p')}.",
                    f"Currently it's {datetime.now().strftime('%H:%M:%S')} on {datetime.now().strftime('%A')}."
                ]
            },
            "joke": {
                "patterns": [
                    "tell me a joke", "make me laugh", "say something funny",
                    "do you know any jokes", "joke please", "entertain me"
                ],
                "responses": [
                    "Why don't scientists trust atoms? Because they make up everything!",
                    "Why did the scarecrow win an award? He was outstanding in his field!",
                    "Why don't eggs tell jokes? They'd crack each other up!",
                    "What do you call a fake noodle? An impasta!",
                    "Why did the math book look so sad? Because it had too many problems!"
                ]
            },
            "feelings": {
                "patterns": [
                    "how are you", "how do you feel", "are you okay",
                    "how is everything", "how are you doing", "you good"
                ],
                "responses": [
                    "I'm doing great! Thanks for asking. How about you?",
                    "I'm functioning perfectly! Ready to help you.",
                    "I'm good! Just here waiting to assist you.",
                    "I'm doing well! How can I make your day better?",
                    "I'm excellent! Always happy to chat with you."
                ]
            }
        }
        
        # Prepare training data
        patterns = []
        labels = []
        
        for intent, data in self.intents.items():
            for pattern in data["patterns"]:
                patterns.append(pattern)
                labels.append(intent)
        
        self.training_data = pd.DataFrame({
            'pattern': patterns,
            'intent': labels
        })
        
        print(f"✅ Created training data with {len(patterns)} patterns across {len(self.intents)} intents")
        return self.training_data
    
    def preprocess_text(self, text):
        """Preprocess text for NLP processing"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = nltk.word_tokenize(text)
        
        # Remove stopwords
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        
        # Lemmatization
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def prepare_features(self):
        """Prepare features for model training"""
        print("\n🔧 Preparing features for NLP model...")
        
        # Preprocess patterns
        self.training_data['processed_pattern'] = self.training_data['pattern'].apply(self.preprocess_text)
        
        # Create tokenizer and sequences
        self.tokenizer = keras.preprocessing.text.Tokenizer(
            num_words=self.vocab_size,
            oov_token="<OOV>"
        )
        self.tokenizer.fit_on_texts(self.training_data['processed_pattern'])
        
        # Convert text to sequences
        sequences = self.tokenizer.texts_to_sequences(self.training_data['processed_pattern'])
        X = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=self.max_sequence_length)
        
        # Encode labels
        y = self.le.fit_transform(self.training_data['intent'])
        
        print(f"Vocabulary size: {len(self.tokenizer.word_index)}")
        print(f"Feature shape: {X.shape}")
        print(f"Number of classes: {len(self.le.classes_)}")
        
        return X, y
    
    def build_model(self, input_shape, num_classes):
        """Build the neural network model for intent classification"""
        print("\n🧠 Building Neural Network Model...")
        
        model = keras.Sequential([
            # Embedding layer
            layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_sequence_length
            ),
            
            # LSTM layer for sequence understanding
            layers.LSTM(128, return_sequences=True, dropout=0.2),
            layers.LSTM(64, dropout=0.2),
            
            # Dense layers
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Model Architecture:")
        model.summary()
        
        return model
    
    def train_model(self, X, y, epochs=100):
        """Train the intent classification model"""
        print("\n🎯 Training the AI Model...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.model = self.build_model(X.shape[1], len(self.le.classes_))
        
        # Callbacks for better training
        callbacks = [
            keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=10, factor=0.5)
        ]
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate the model
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"✅ Training completed! Test Accuracy: {test_accuracy:.4f}")
        
        return history, test_accuracy
    
    def plot_training_history(self, history):
        """Plot training history"""
        print("\n📊 Plotting Training History...")
        
        plt.figure(figsize=(12, 4))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def predict_intent(self, text):
        """Predict the intent of user input"""
        # Preprocess the text
        processed_text = self.preprocess_text(text)
        
        # Convert to sequence
        sequence = self.tokenizer.texts_to_sequences([processed_text])
        padded_sequence = keras.preprocessing.sequence.pad_sequences(
            sequence, maxlen=self.max_sequence_length
        )
        
        # Predict
        prediction = self.model.predict(padded_sequence, verbose=0)
        intent_index = np.argmax(prediction)
        confidence = prediction[0][intent_index]
        
        intent = self.le.inverse_transform([intent_index])[0]
        
        return intent, confidence
    
    def get_response(self, intent):
        """Get a random response for the given intent"""
        if intent in self.intents:
            return random.choice(self.intents[intent]["responses"])
        else:
            return "I'm not sure how to respond to that. Can you try asking differently?"
    
    def chat(self):
        """Start interactive chat session"""
        print("\n🤖 AI Chatbot Activated!")
        print("=" * 50)
        print("Hello! I'm your AI assistant. Type 'quit' to exit.")
        print("You can ask me about technology, time, jokes, or just chat!")
        print("=" * 50)
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                print("Bot: Goodbye! Thanks for chatting with me! 👋")
                break
            
            if not user_input:
                continue
            
            # Predict intent and get response
            intent, confidence = self.predict_intent(user_input)
            response = self.get_response(intent)
            
            print(f"Bot: {response}")
            print(f"   [Intent: {intent}, Confidence: {confidence:.2f}]")
    
    def evaluate_chatbot(self):
        """Evaluate chatbot performance with sample inputs"""
        print("\n🧪 Chatbot Evaluation")
        print("=" * 40)
        
        test_cases = [
            "hello there",
            "what is artificial intelligence?",
            "tell me a joke",
            "what time is it?",
            "how are you feeling?",
            "thank you for your help",
            "what can you do?",
            "explain machine learning"
        ]
        
        print("Testing chatbot with sample inputs:\n")
        for test_input in test_cases:
            intent, confidence = self.predict_intent(test_input)
            response = self.get_response(intent)
            
            print(f"Input: '{test_input}'")
            print(f"Intent: {intent} (Confidence: {confidence:.2f})")
            print(f"Response: {response}")
            print("-" * 50)
    
    def run_complete_pipeline(self):
        """Run the complete chatbot creation pipeline"""
        print("🚀 AI CHATBOT WITH NLP - COMPLETE PIPELINE")
        print("=" * 55)
        
        # Step 1: Create training data
        self.create_training_data()
        
        # Step 2: Prepare features
        X, y = self.prepare_features()
        
        # Step 3: Train model
        history, accuracy = self.train_model(X, y)
        
        # Step 4: Plot training history
        self.plot_training_history(history)
        
        # Step 5: Evaluate chatbot
        self.evaluate_chatbot()
        
        print(f"\n✅ Chatbot Pipeline Completed!")
        print(f"📊 Final Test Accuracy: {accuracy:.4f}")
        print(f"🎯 Number of Intents: {len(self.intents)}")
        print(f"💬 Vocabulary Size: {len(self.tokenizer.word_index)}")
        
        return accuracy

# Advanced Chatbot Features
class AdvancedChatbot(NLPChatbot):
    """Extended chatbot with additional features"""
    
    def __init__(self):
        super().__init__()
        self.conversation_history = []
        self.user_name = None
    
    def remember_user_info(self, user_input):
        """Basic named entity recognition for user information"""
        # Simple name extraction
        name_patterns = [
            r"my name is (\w+)",
            r"i am (\w+)",
            r"call me (\w+)",
            r"you can call me (\w+)"
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, user_input.lower())
            if match:
                self.user_name = match.group(1).title()
                return True
        return False
    
    def get_personalized_response(self, intent, user_input):
        """Get personalized response based on conversation history"""
        base_response = self.get_response(intent)
        
        # Add personalization if we know user's name
        if self.user_name and random.random() > 0.7:  # 30% chance to use name
            if "greeting" in intent:
                base_response = base_response.replace("!", f", {self.user_name}!")
            elif "goodbye" in intent:
                base_response = f"Goodbye, {self.user_name}! {base_response}"
        
        return base_response
    
    def handle_unknown_intent(self, user_input):
        """Handle cases where intent is not confidently recognized"""
        low_confidence_responses = [
            "I'm not sure I understand. Could you rephrase that?",
            "That's interesting! Could you tell me more?",
            "I'm still learning. Could you try asking differently?",
            "I want to make sure I understand correctly. Could you elaborate?",
            "That's a new one for me! What do you mean by that?"
        ]
        
        return random.choice(low_confidence_responses)
    
    def advanced_chat(self):
        """Enhanced chat session with additional features"""
        print("\n🤖 ADVANCED AI Chatbot Activated!")
        print("=" * 50)
        print("Hello! I'm your advanced AI assistant.")
        print("I can remember your name and have more natural conversations!")
        print("Type 'quit' to exit.")
        print("=" * 50)
        
        conversation_count = 0
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                farewell = f"Goodbye{', ' + self.user_name if self.user_name else ''}! "
                farewell += f"We had {conversation_count} exchanges. Come back soon! 👋"
                print(f"Bot: {farewell}")
                break
            
            if not user_input:
                continue
            
            # Remember user information
            if not self.user_name:
                self.remember_user_info(user_input)
            
            # Predict intent
            intent, confidence = self.predict_intent(user_input)
            
            # Get appropriate response
            if confidence < 0.6:  # Low confidence threshold
                response = self.handle_unknown_intent(user_input)
            else:
                response = self.get_personalized_response(intent, user_input)
            
            # Store conversation
            self.conversation_history.append({
                'user': user_input,
                'bot': response,
                'intent': intent,
                'confidence': confidence,
                'timestamp': datetime.now()
            })
            
            print(f"Bot: {response}")
            if confidence < 0.8:
                print(f"   [I'm {confidence:.0%} sure about this]")
            
            conversation_count += 1
    
    def show_conversation_stats(self):
        """Display conversation statistics"""
        if not self.conversation_history:
            print("No conversation history yet.")
            return
        
        print("\n📈 Conversation Statistics")
        print("=" * 30)
        print(f"Total exchanges: {len(self.conversation_history)}")
        
        if self.user_name:
            print(f"User name: {self.user_name}")
        
        # Most common intents
        intents = [chat['intent'] for chat in self.conversation_history]
        intent_counts = pd.Series(intents).value_counts()
        
        print("\nMost common conversation topics:")
        for intent, count in intent_counts.head().items():
            print(f"  {intent}: {count} times")
        
        # Average confidence
        avg_confidence = np.mean([chat['confidence'] for chat in self.conversation_history])
        print(f"\nAverage confidence: {avg_confidence:.2f}")

# Demo and Testing Functions
def demo_chatbot():
    """Demo the basic chatbot"""
    print("🎭 DEMO: Basic AI Chatbot")
    print("=" * 30)
    
    bot = NLPChatbot()
    bot.run_complete_pipeline()
    
    print("\n" + "=" * 50)
    print("Starting interactive chat session...")
    print("=" * 50)
    
    bot.chat()

def demo_advanced_chatbot():
    """Demo the advanced chatbot"""
    print("🎭 DEMO: Advanced AI Chatbot")
    print("=" * 35)
    
    advanced_bot = AdvancedChatbot()
    advanced_bot.run_complete_pipeline()
    
    print("\n" + "=" * 50)
    print("Starting advanced interactive chat session...")
    print("=" * 50)
    
    advanced_bot.advanced_chat()
    advanced_bot.show_conversation_stats()

def performance_analysis():
    """Analyze chatbot performance"""
    print("\n📊 CHATBOT PERFORMANCE ANALYSIS")
    print("=" * 40)
    
    bot = NLPChatbot()
    bot.create_training_data()
    X, y = bot.prepare_features()
    
    # Analyze class distribution
    intent_distribution = bot.training_data['intent'].value_counts()
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    intent_distribution.plot(kind='bar', color='skyblue')
    plt.title('Intent Distribution in Training Data')
    plt.xlabel('Intents')
    plt.ylabel('Number of Patterns')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    # Show pattern length distribution
    pattern_lengths = bot.training_data['pattern'].str.len()
    plt.hist(pattern_lengths, bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
    plt.title('Pattern Length Distribution')
    plt.xlabel('Pattern Length (characters)')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    print("Training Data Summary:")
    print(f"Total patterns: {len(bot.training_data)}")
    print(f"Number of intents: {len(intent_distribution)}")
    print(f"Average pattern length: {pattern_lengths.mean():.1f} characters")
    print(f"Vocabulary size: {len(bot.tokenizer.word_index)}")

# Main execution
if __name__ == "__main__":
    print("🤖 AI CHATBOT WITH NATURAL LANGUAGE PROCESSING")
    print("=" * 55)
    
    # Run performance analysis
    performance_analysis()
    
    # Demo basic chatbot
    demo_chatbot()
    
    # Uncomment to demo advanced chatbot
    # print("\n" + "="*60)
    # demo_advanced_chatbot()
    
    print("\n🎉 CHATBOT PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print("\nKey Features Demonstrated:")
    print("✅ Natural Language Processing (NLP)")
    print("✅ Text Preprocessing & Tokenization")
    print("✅ LSTM Neural Network for Intent Classification")
    print("✅ Multi-class Classification")
    print("✅ Interactive Chat Interface")
    print("✅ Confidence Scoring")
    print("✅ Response Generation")
