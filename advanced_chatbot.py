from ai_chatbot import NLPChatbot
import re
import random
import pandas as pd
import numpy as np
from datetime import datetime

class AdvancedChatbot(NLPChatbot):
    """Extended chatbot with additional features"""
    
    def __init__(self):
        super().__init__()
        self.conversation_history = []
        self.user_name = None
        self.user_context = {}
    
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

if __name__ == "__main__":
    demo_advanced_chatbot()