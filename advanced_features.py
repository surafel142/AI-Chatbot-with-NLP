#!/usr/bin/env python3
"""
Advanced features demonstration
"""

from advanced_chatbot import AdvancedChatbot

def demo_advanced_features():
    """Showcase advanced chatbot features"""
    print("🚀 ADVANCED CHATBOT FEATURES DEMO")
    print("=" * 40)
    
    # Create advanced chatbot
    advanced_bot = AdvancedChatbot()
    
    # Train the model
    advanced_bot.run_complete_pipeline()
    
    # Demo conversation with statistics
    print("\n🤖 Starting Advanced Chat Session")
    print("Features included:")
    print("✅ Personalized responses")
    print("✅ Conversation memory")
    print("✅ User name recognition")
    print("✅ Confidence scoring")
    print("✅ Conversation statistics")
    print("=" * 50)
    
    # Start chat
    advanced_bot.advanced_chat()
    
    # Show statistics
    advanced_bot.show_conversation_stats()

if __name__ == "__main__":
    demo_advanced_features()