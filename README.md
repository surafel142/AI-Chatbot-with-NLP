# 🤖 AI Chatbot with Natural Language Processing

## 📖 Overview

A sophisticated AI chatbot built with Python, NLTK, and TensorFlow/Keras that demonstrates advanced natural language processing capabilities. This portfolio-ready project features intent classification, response generation, and interactive conversation handling using deep learning techniques.

## 🚀 Features

- **💬 Natural Language Understanding** - Advanced text preprocessing and tokenization
- **🧠 LSTM Neural Network** - Deep learning for intent classification
- **🔍 Intent Recognition** - Multi-class classification with 10+ conversation categories
- **🎯 Confidence Scoring** - Probability-based response selection
- **👤 Personalization** - Name recognition and memory
- **📊 Conversation Analytics** - History tracking and performance metrics
- **🛡️ Error Handling** - Graceful handling of unknown inputs
- **⚡ Real-time Interaction** - Instant response generation

## 🛠️ Technologies Used

- **Python 3.7+**
- **TensorFlow/Keras** - Deep learning framework
- **NLTK** - Natural Language Processing toolkit
- **Pandas & NumPy** - Data manipulation and analysis
- **Matplotlib & Seaborn** - Data visualization
- **Scikit-learn** - Data preprocessing and evaluation
- **Re** - Regular expressions for text processing

## 📦 Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Step-by-Step Setup

1. **Clone or download the project**
```bash
git clone <repository-url>
cd ai-chatbot-nlp
```

2. **Create virtual environment (recommended)**
```bash
python -m venv chatbot_env
source chatbot_env/bin/activate  # On Windows: chatbot_env\Scripts\activate
```

3. **Install required packages**
```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, install manually:
```bash
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn nltk jupyter
```

4. **Download NLTK data**
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## 🎯 Quick Start

### Run the Complete Project
```python
from ai_chatbot import NLPChatbot

# Initialize and run the complete chatbot pipeline
chatbot = NLPChatbot()
chatbot.run_complete_pipeline()

# Start chatting
chatbot.chat()
```

### Basic Usage Example
```python
# Quick setup and prediction
chatbot = NLPChatbot()
chatbot.create_training_data()
X, y = chatbot.prepare_features()
chatbot.train_model(X, y)

# Predict intent of a message
intent, confidence = chatbot.predict_intent("hello there")
response = chatbot.get_response(intent)
print(f"Response: {response}")
```

## 📁 Project Structure

```
ai-chatbot-nlp/
│
├── ai_chatbot.py              # Main chatbot class
├── advanced_chatbot.py        # Enhanced version with additional features
├── requirements.txt           # Dependencies
├── README.md                  # Project documentation
├── examples/                  # Example usage scripts
│   ├── basic_demo.py
│   ├── advanced_features.py
│   └── performance_test.py
├── data/                      # Training data and configurations
│   ├── intents.json
│   └── patterns.csv
├── models/                    # Saved model files
│   └── intent_classifier.h5
└── notebooks/                 # Jupyter notebooks for analysis
    ├── model_training.ipynb
    └── nlp_analysis.ipynb
```

## 🗣️ Conversation Capabilities

### Supported Intents
1. **`greeting`** - Hello, hi, greetings
2. **`goodbye`** - Bye, see you, farewell
3. **`thanks`** - Thank you, appreciation
4. **`about_bot`** - Who are you, what can you do
5. **`help`** - Assistance, what can you help with
6. **`technology`** - AI, ML, Python, NLP explanations
7. **`weather`** - Weather-related queries
8. **`time`** - Current time and date
9. **`joke`** - Entertainment and humor
10. **`feelings`** - How are you, emotional check-ins

### Sample Conversations
```
User: Hello there!
Bot: Hello! I'm your AI assistant. What can I help you with?
[Intent: greeting, Confidence: 0.99]

User: What is machine learning?
Bot: Machine learning is a subset of AI that enables computers to learn and improve from experience without explicit programming.
[Intent: technology, Confidence: 0.94]

User: Tell me a joke
Bot: Why don't scientists trust atoms? Because they make up everything!
[Intent: joke, Confidence: 0.96]
```

## 🔧 Core Components

### 1. Natural Language Processing
```python
# Text preprocessing pipeline
def preprocess_text(self, text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation
    tokens = nltk.word_tokenize(text)        # Tokenization
    tokens = [t for t in tokens if t not in stopwords]  # Remove stopwords
    tokens = [lemmatizer.lemmatize(t) for t in tokens]  # Lemmatization
    return ' '.join(tokens)
```

### 2. Neural Network Architecture
```python
# LSTM-based intent classifier
model = keras.Sequential([
    layers.Embedding(vocab_size, 50, input_length=20),
    layers.LSTM(128, return_sequences=True, dropout=0.2),
    layers.LSTM(64, dropout=0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation='softmax')
])
```

### 3. Training Pipeline
```python
# Complete training process
def run_complete_pipeline(self):
    self.create_training_data()          # Step 1: Data creation
    X, y = self.prepare_features()       # Step 2: Feature engineering
    history, accuracy = self.train_model(X, y)  # Step 3: Model training
    self.plot_training_history(history)  # Step 4: Visualization
    self.evaluate_chatbot()              # Step 5: Performance evaluation
    return accuracy
```

## 🧠 Model Architecture

### LSTM Network Details
- **Embedding Layer**: 1000 vocabulary size, 50 dimensions
- **LSTM Layers**: 
  - Layer 1: 128 units with return sequences
  - Layer 2: 64 units for final sequence processing
- **Dense Layers**: 64 → 32 neurons with ReLU activation
- **Output Layer**: Softmax for multi-class classification
- **Regularization**: Dropout (20-30%) to prevent overfitting

### Training Configuration
- **Optimizer**: Adam with adaptive learning rate
- **Loss Function**: Sparse Categorical Crossentropy
- **Metrics**: Accuracy
- **Batch Size**: 32 sequences
- **Validation Split**: 20%
- **Early Stopping**: Patience of 15 epochs
- **Callbacks**: Learning rate reduction on plateau

## 📊 Performance Metrics

### Typical Results
- **Test Accuracy**: 90-95% on intent classification
- **Training Time**: 1-2 minutes on CPU
- **Prediction Speed**: < 100ms per message
- **Vocabulary Size**: 150-200 words
- **Number of Patterns**: 80-100 training examples

### Evaluation Metrics
```python
# Model evaluation output
📊 Model Performance:
  Test Accuracy: 0.9412
  Number of Intents: 10
  Vocabulary Size: 187
  Average Confidence: 0.89
```

## 🎮 Interactive Features

### Basic Chat Interface
```python
def chat(self):
    print("🤖 AI Chatbot Activated!")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['quit', 'exit']:
            break
        
        intent, confidence = self.predict_intent(user_input)
        response = self.get_response(intent)
        print(f"Bot: {response}")
        print(f"   [Intent: {intent}, Confidence: {confidence:.2f}]")
```

### Advanced Features
- **Name Recognition**: Remembers user's name from conversation
- **Confidence Thresholding**: Handles low-confidence predictions gracefully
- **Conversation History**: Tracks dialogue for analytics
- **Personalized Responses**: Uses context for more natural interactions

## ⚡ Advanced Usage

### Custom Intents
```python
# Add new conversation categories
new_intent = {
    "shopping": {
        "patterns": [
            "I want to buy", "show me products", "shopping",
            "what do you sell", "online store"
        ],
        "responses": [
            "I can help you with shopping! What are you looking for?",
            "Great! Let me show you our products.",
            "I'd be happy to assist with your shopping needs."
        ]
    }
}
chatbot.intents.update(new_intent)
```

### Model Persistence
```python
# Save and load trained model
chatbot.model.save('chatbot_model.h5')

# Load for future use
from tensorflow import keras
loaded_model = keras.models.load_model('chatbot_model.h5')
chatbot.model = loaded_model
```

### Integration with Web Services
```python
# Example: Add weather API integration
def get_weather_response(self):
    # Call weather API
    weather_data = requests.get(weather_api_url).json()
    return f"The current temperature is {weather_data['temp']}°C"
```

## 🔍 Performance Analysis

### Training Visualization
The project includes comprehensive visualization of:
- Training vs Validation accuracy
- Training vs Validation loss
- Intent distribution in training data
- Pattern length analysis
- Confidence score distributions

### Model Comparison
```python
# Compare different architectures
from advanced_analysis import compare_models
results = compare_models()
print("Model Performance Comparison:")
print(results)
```

## 🚀 Deployment Ready

### Production Considerations
- **Error Handling**: Graceful degradation for unknown inputs
- **Scalability**: Modular design for easy expansion
- **Performance**: Optimized preprocessing and prediction
- **Maintainability**: Clean code structure and documentation

### API Integration Example
```python
from flask import Flask, request, jsonify

app = Flask(__name__)
chatbot = NLPChatbot()

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    user_message = request.json['message']
    intent, confidence = chatbot.predict_intent(user_message)
    response = chatbot.get_response(intent)
    
    return jsonify({
        'response': response,
        'intent': intent,
        'confidence': float(confidence)
    })
```

## 🎓 Educational Value

This project demonstrates:

- ✅ **Natural Language Processing** - Text preprocessing and analysis
- ✅ **Deep Learning** - LSTM networks for sequence classification
- ✅ **Intent Recognition** - Multi-class classification techniques
- ✅ **Model Evaluation** - Comprehensive performance metrics
- ✅ **Software Engineering** - Modular, maintainable code structure
- ✅ **User Experience** - Interactive and engaging chatbot design
- ✅ **Production Readiness** - Error handling and scalability

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Areas for Contribution
- Additional NLP features (sentiment analysis, entity recognition)
- More sophisticated neural architectures (Transformers, BERT)
- Integration with external APIs
- Multi-language support
- Voice interface integration

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built for educational purposes to demonstrate NLP and deep learning
- Inspired by real-world chatbot applications and conversational AI
- Uses TensorFlow, Keras, NLTK, and other amazing open-source libraries
- Incorporates best practices from NLP research and industry applications

## 📞 Support

For questions or issues:

1. Check the examples directory for usage patterns
2. Review the code documentation and comments
3. Test with the provided sample conversations
4. Open an issue on GitHub with detailed information

---

**Happy Chatting! 🤖💬**

*Transforming natural language into meaningful conversations with AI*
