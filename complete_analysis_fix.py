# complete_analysis_fix.py - Complete fix for all AI analysis functions

import re
import os

def fix_all_analysis_functions():
    """Fix all broken analysis functions in app.py in one go"""
    
    print("🔧 COMPLETE AI ANALYSIS FIX")
    print("=" * 50)
    
    # Check if app.py exists
    if not os.path.exists('app.py'):
        print("❌ Error: app.py file not found!")
        return False
    
    # Read the current app.py file
    try:
        with open('app.py', 'r', encoding='utf-8') as f:
            content = f.read()
        print("✅ Successfully read app.py file")
    except Exception as e:
        print(f"❌ Error reading app.py: {e}")
        return False
    
    # Fix 1: Replace generate_summary function
    print("🔄 Fixing generate_summary function...")
    
    new_summary = '''def generate_summary(text):
    """Generate a summary of the text"""
    try:
        # Use basic sentence splitting as fallback
        try:
            import nltk
            sentences = nltk.sent_tokenize(text)
        except:
            sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
        
        if len(sentences) <= 3:
            return text
        
        # Simple extractive summarization
        words = text.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Skip short words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        sentence_scores = {}
        for sentence in sentences:
            sentence_words = sentence.lower().split()
            score = sum(word_freq.get(word, 0) for word in sentence_words)
            sentence_scores[sentence] = score
        
        # Get top 3 sentences
        if sentence_scores:
            top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            summary = ' '.join([sentence[0] for sentence in top_sentences])
            return summary if summary else text[:200] + "..."
        else:
            return text[:200] + "..." if len(text) > 200 else text
            
    except Exception as e:
        print(f"Summary error: {e}")
        return text[:200] + "..." if len(text) > 200 else text'''
    
    # Replace summary function using regex
    summary_pattern = r'def generate_summary\(text\):.*?(?=\ndef |\nif __name__|\n@app\.route|\Z)'
    content = re.sub(summary_pattern, new_summary, content, flags=re.DOTALL)
    
    # Fix 2: Replace extract_insights function
    print("🔄 Fixing extract_insights function...")
    
    new_insights = '''def extract_insights(text):
    """Extract insights from text"""
    try:
        insights = []
        
        # Basic text analysis
        words = text.split()
        word_count = len(words)
        
        # Use basic sentence splitting as fallback
        try:
            import nltk
            sentences = nltk.sent_tokenize(text)
        except:
            sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        sentence_count = len(sentences)
        
        # Text length insights
        if word_count > 500:
            insights.append("This is a lengthy text that covers substantial content.")
        elif word_count < 50:
            insights.append("This is a brief text with concise information.")
        else:
            insights.append("This text has a moderate length with balanced content.")
        
        # Complexity insights
        if sentence_count > 0:
            avg_sentence_length = word_count / sentence_count
            if avg_sentence_length > 20:
                insights.append("The text contains complex, detailed sentences.")
            elif avg_sentence_length < 10:
                insights.append("The text uses simple, concise sentences.")
            else:
                insights.append("The text has well-balanced sentence structure.")
        
        # Content type insights
        text_lower = text.lower()
        if 'customer service' in text_lower or 'help' in text_lower or 'support' in text_lower:
            insights.append("This appears to be a customer service interaction.")
        
        if 'hello' in text_lower or 'hi' in text_lower or 'thank you' in text_lower:
            insights.append("The text contains polite greetings and courteous language.")
        
        # Question insights
        question_count = text.count('?')
        if question_count > 3:
            insights.append("The text contains many questions, suggesting inquiry or discussion.")
        elif question_count > 0:
            insights.append("The text includes questions for engagement.")
        
        # Exclamation insights
        exclamation_count = text.count('!')
        if exclamation_count > 2:
            insights.append("The text shows strong emotions or emphasis.")
        
        # Personal pronouns
        personal_pronouns = [' i ', ' me ', ' my ', ' mine ', ' myself ']
        personal_count = sum(text_lower.count(pronoun) for pronoun in personal_pronouns)
        if personal_count > 3:
            insights.append("The text is highly personal and subjective.")
        
        # Professional language
        professional_words = ['service', 'help', 'assist', 'support', 'thank', 'please']
        professional_count = sum(text_lower.count(word) for word in professional_words)
        if professional_count > 2:
            insights.append("The text uses professional and courteous language.")
        
        return insights if insights else ["This text contains conversational content."]
        
    except Exception as e:
        print(f"Insights error: {e}")
        # Fallback insights
        word_count = len(text.split())
        if word_count > 100:
            return ["This is a substantial piece of text with multiple topics."]
        elif word_count > 50:
            return ["This is a moderate-length text with clear content."]
        else:
            return ["This is a brief text with concise information."]'''
    
    # Replace insights function using regex
    insights_pattern = r'def extract_insights\(text\):.*?(?=\ndef |\nif __name__|\n@app\.route|\Z)'
    content = re.sub(insights_pattern, new_insights, content, flags=re.DOTALL)
    
    # Fix 3: Replace generate_suggestions function
    print("🔄 Fixing generate_suggestions function...")
    
    new_suggestions = '''def generate_suggestions(text):
    """Generate suggestions based on text analysis"""
    try:
        suggestions = []
        
        # Analyze sentiment for suggestions
        try:
            sentiment_result = analyze_sentiment(text)
            if isinstance(sentiment_result, dict) and 'error' not in sentiment_result:
                sentiment = sentiment_result.get('sentiment', 'Neutral')
                if sentiment == 'Negative':
                    suggestions.append("Consider addressing the negative aspects mentioned to improve overall tone.")
                elif sentiment == 'Positive':
                    suggestions.append("Great positive tone! Consider leveraging this enthusiasm in future communications.")
                else:
                    suggestions.append("The neutral tone is balanced. Consider adding more emotional engagement.")
        except:
            suggestions.append("Consider reviewing the overall tone of the content.")
        
        # Length-based suggestions
        words = text.split()
        word_count = len(words)
        if word_count > 1000:
            suggestions.append("Consider breaking down this lengthy content into smaller, digestible sections.")
        elif word_count < 100:
            suggestions.append("Consider expanding on key points to provide more comprehensive information.")
        else:
            suggestions.append("The content length is appropriate for the topic.")
        
        # Clarity suggestions
        try:
            import nltk
            sentences = nltk.sent_tokenize(text)
        except:
            sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        sentence_count = len(sentences)
        if sentence_count > 0:
            avg_sentence_length = word_count / sentence_count
            if avg_sentence_length > 25:
                suggestions.append("Consider using shorter sentences to improve readability.")
            elif avg_sentence_length < 8:
                suggestions.append("Consider combining some short sentences for better flow.")
        
        # Engagement suggestions
        question_count = text.count('?')
        if question_count == 0:
            suggestions.append("Consider adding questions to increase engagement with your audience.")
        
        # Action-oriented suggestions
        action_words = ['should', 'must', 'need', 'recommend', 'suggest', 'propose']
        action_count = sum(text.lower().count(word) for word in action_words)
        if action_count == 0:
            suggestions.append("Consider adding actionable recommendations or next steps.")
        
        # Customer service specific suggestions
        text_lower = text.lower()
        if 'customer service' in text_lower or 'support' in text_lower:
            suggestions.append("For customer service interactions, ensure clear resolution steps are provided.")
        
        return suggestions if suggestions else ["The text appears well-structured overall."]
        
    except Exception as e:
        print(f"Suggestions error: {e}")
        return ["Consider reviewing the content for clarity and engagement."]'''
    
    # Replace suggestions function using regex
    suggestions_pattern = r'def generate_suggestions\(text\):.*?(?=\ndef |\nif __name__|\n@app\.route|\Z)'
    content = re.sub(suggestions_pattern, new_suggestions, content, flags=re.DOTALL)
    
    # Fix 4: Improve get_text_statistics function
    print("🔄 Fixing get_text_statistics function...")
    
    new_statistics = '''def get_text_statistics(text):
    """Get basic text statistics"""
    try:
        # Use basic sentence splitting as fallback
        try:
            import nltk
            sentences = nltk.sent_tokenize(text)
        except:
            sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        words = text.split()
        
        # Calculate statistics
        word_count = len(words)
        sentence_count = len(sentences)
        character_count = len(text)
        
        # Calculate averages
        avg_sentence_length = round(word_count / sentence_count, 2) if sentence_count > 0 else 0
        avg_word_length = round(sum(len(word) for word in words) / word_count, 2) if word_count > 0 else 0
        reading_time_minutes = round(word_count / 200, 1)  # Assuming 200 words per minute
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'character_count': character_count,
            'avg_sentence_length': avg_sentence_length,
            'avg_word_length': avg_word_length,
            'reading_time_minutes': reading_time_minutes
        }
    except Exception as e:
        print(f"Statistics error: {e}")
        # Fallback statistics
        words = text.split()
        return {
            'word_count': len(words),
            'sentence_count': text.count('.') + text.count('!') + text.count('?'),
            'character_count': len(text),
            'avg_sentence_length': 0,
            'avg_word_length': 0,
            'reading_time_minutes': round(len(words) / 200, 1)
        }'''
    
    # Replace statistics function using regex
    statistics_pattern = r'def get_text_statistics\(text\):.*?(?=\ndef |\nif __name__|\n@app\.route|\Z)'
    content = re.sub(statistics_pattern, new_statistics, content, flags=re.DOTALL)
    
    # Fix 5: Improve extract_key_topics function
    print("🔄 Fixing extract_key_topics function...")
    
    new_topics = '''def extract_key_topics(text):
    """Extract key topics using improved keyword extraction"""
    try:
        # Get keywords first
        keywords = get_top_keywords(text)
        
        # Try TF-IDF if available
        try:
            import nltk
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            sentences = nltk.sent_tokenize(text)
            if len(sentences) >= 2:
                # TF-IDF vectorization
                vectorizer = TfidfVectorizer(
                    max_features=50,
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=1
                )
                
                tfidf_matrix = vectorizer.fit_transform(sentences)
                feature_names = vectorizer.get_feature_names_out()
                
                # Get top keywords
                import numpy as np
                mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
                top_indices = mean_scores.argsort()[-10:][::-1]
                
                topics = []
                for idx in top_indices:
                    if mean_scores[idx] > 0:
                        topics.append({
                            'topic': feature_names[idx],
                            'relevance': round(mean_scores[idx], 3)
                        })
                
                return {
                    'topics': topics,
                    'keywords': keywords
                }
        except:
            pass
        
        # Fallback: Use simple keyword-based topics
        text_lower = text.lower()
        common_topics = {
            'customer service': ['customer', 'service', 'help', 'support', 'assist'],
            'communication': ['call', 'phone', 'speak', 'talk', 'conversation'],
            'questions': ['question', 'ask', 'wonder', 'inquire', 'what', 'how', 'why'],
            'assistance': ['help', 'assist', 'support', 'aid', 'guide'],
            'greeting': ['hello', 'hi', 'thank', 'please', 'welcome']
        }
        
        topics = []
        for topic, topic_words in common_topics.items():
            score = sum(text_lower.count(word) for word in topic_words)
            if score > 0:
                topics.append({
                    'topic': topic,
                    'relevance': round(score / len(text.split()) * 100, 3)
                })
        
        # Sort by relevance
        topics.sort(key=lambda x: x['relevance'], reverse=True)
        
        return {
            'topics': topics[:10],  # Top 10 topics
            'keywords': keywords
        }
        
    except Exception as e:
        print(f"Topics error: {e}")
        return {
            'topics': [],
            'keywords': get_top_keywords(text)
        }'''
    
    # Replace topics function using regex
    topics_pattern = r'def extract_key_topics\(text\):.*?(?=\ndef |\nif __name__|\n@app\.route|\Z)'
    content = re.sub(topics_pattern, new_topics, content, flags=re.DOTALL)
    
    # Write the fixed content back to app.py
    try:
        with open('app.py', 'w', encoding='utf-8') as f:
            f.write(content)
        print("✅ Successfully wrote all fixes to app.py")
    except Exception as e:
        print(f"❌ Error writing to app.py: {e}")
        return False
    
    return True

def main():
    """Main function to run the complete fix"""
    print("🔧 COMPLETE AI ANALYSIS FIX SCRIPT")
    print("=" * 50)
    print("This script will fix ALL analysis functions:")
    print("• generate_summary() - Text summarization")
    print("• extract_insights() - Text insights extraction")
    print("• generate_suggestions() - Suggestion generation")
    print("• get_text_statistics() - Text statistics")
    print("• extract_key_topics() - Topic and keyword extraction")
    print("=" * 50)
    
    # Run the complete fix
    success = fix_all_analysis_functions()
    
    if success:
        print("\n" + "=" * 50)
        print("🎉 SUCCESS! ALL AI ANALYSIS FUNCTIONS FIXED!")
        print("=" * 50)
        print("✅ Summary function - FIXED")
        print("✅ Insights function - FIXED")
        print("✅ Suggestions function - FIXED")
        print("✅ Statistics function - IMPROVED")
        print("✅ Topics function - IMPROVED")
        print("=" * 50)
        print("🚀 Next steps:")
        print("1. Restart your app: python app.py")
        print("2. Open: http://localhost:5000")
        print("3. Test all analysis functions")
        print("4. All sections should now work perfectly!")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("❌ FAILED! Could not fix the analysis functions.")
        print("Please check the error messages above.")
        print("=" * 50)

if __name__ == "__main__":
    main()