#!/usr/bin/env python3
"""
Performance testing and analysis
"""

from ai_chatbot import NLPChatbot
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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

def test_response_times():
    """Test chatbot response times"""
    print("\n⏱️ RESPONSE TIME TESTING")
    print("=" * 30)
    
    bot = NLPChatbot()
    bot.run_complete_pipeline()
    
    test_inputs = [
        "hello",
        "what is AI?",
        "tell me a joke",
        "what time is it?",
        "thank you",
        "how are you?",
        "what can you do?",
        "goodbye"
    ]
    
    import time
    
    print("Testing response times for various inputs:\n")
    for test_input in test_inputs:
        start_time = time.time()
        intent, confidence = bot.predict_intent(test_input)
        response = bot.get_response(intent)
        end_time = time.time()
        
        response_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        print(f"Input: '{test_input}'")
        print(f"Response: {response}")
        print(f"Time: {response_time:.2f} ms")
        print(f"Confidence: {confidence:.2f}")
        print("-" * 40)

if __name__ == "__main__":
    performance_analysis()
    test_response_times()