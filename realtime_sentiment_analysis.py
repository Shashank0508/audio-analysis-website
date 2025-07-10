#!/usr/bin/env python3
"""
Real-time Sentiment Analysis
Processes transcripts in real-time to provide live sentiment scoring and analysis.
"""

import asyncio
import threading
import time
import queue
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import re
from collections import deque, defaultdict
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealTimeSentimentAnalysis:
    """Real-time sentiment analysis for streaming transcripts."""
    
    def __init__(self):
        self.sentiment_queue = queue.Queue()
        self.processing_thread = None
        self.running = False
        self.session_data = {}
        self.sentiment_history = defaultdict(deque)
        self.callbacks = []
        
        # Sentiment scoring weights
        self.sentiment_weights = {
            'positive_words': 1.0,
            'negative_words': -1.0,
            'neutral_words': 0.0,
            'exclamation_boost': 0.2,
            'question_penalty': -0.1,
            'caps_penalty': -0.3,
            'repetition_penalty': -0.2
        }
        
        # Predefined sentiment lexicons (simplified)
        self.positive_words = {
            'excellent', 'great', 'amazing', 'wonderful', 'fantastic', 'perfect',
            'love', 'like', 'enjoy', 'happy', 'pleased', 'satisfied', 'good',
            'awesome', 'brilliant', 'outstanding', 'superb', 'marvelous',
            'thank', 'thanks', 'appreciate', 'grateful', 'helpful', 'friendly',
            'yes', 'absolutely', 'definitely', 'certainly', 'sure', 'agreed',
            'recommend', 'impressed', 'delighted', 'thrilled', 'excited'
        }
        
        self.negative_words = {
            'terrible', 'awful', 'horrible', 'bad', 'worst', 'hate', 'dislike',
            'angry', 'frustrated', 'annoyed', 'disappointed', 'upset', 'sad',
            'problem', 'issue', 'trouble', 'difficult', 'hard', 'impossible',
            'wrong', 'error', 'mistake', 'fail', 'failed', 'broken', 'bug',
            'slow', 'expensive', 'cheap', 'poor', 'useless', 'worthless',
            'no', 'never', 'nothing', 'nobody', 'none', 'neither', 'nor',
            'complain', 'complaint', 'refund', 'cancel', 'quit', 'leave'
        }
        
        self.neutral_words = {
            'maybe', 'perhaps', 'possibly', 'probably', 'might', 'could',
            'would', 'should', 'okay', 'ok', 'fine', 'normal', 'regular',
            'standard', 'typical', 'usual', 'average', 'medium', 'moderate'
        }
        
        # Emotion indicators
        self.emotion_patterns = {
            'joy': {'happy', 'joy', 'excited', 'thrilled', 'delighted', 'cheerful'},
            'sadness': {'sad', 'depressed', 'disappointed', 'unhappy', 'down'},
            'anger': {'angry', 'mad', 'furious', 'rage', 'annoyed', 'frustrated'},
            'fear': {'scared', 'afraid', 'worried', 'anxious', 'nervous', 'concerned'},
            'surprise': {'surprised', 'shocked', 'amazed', 'astonished', 'wow'},
            'disgust': {'disgusted', 'sick', 'gross', 'awful', 'terrible'},
            'neutral': {'neutral', 'okay', 'fine', 'normal', 'regular'},
            'confusion': {'confused', 'puzzled', 'unclear', 'lost', 'what', 'huh'}
        }
        
    def start_analysis(self):
        """Start real-time sentiment analysis."""
        if self.running:
            logger.warning("Sentiment analysis already running")
            return
            
        self.running = True
        self.processing_thread = threading.Thread(target=self._process_sentiment_queue)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info("✅ Real-time sentiment analysis started")
    
    def stop_analysis(self):
        """Stop real-time sentiment analysis."""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        
        logger.info("⏹️ Real-time sentiment analysis stopped")
    
    def register_callback(self, callback):
        """Register callback for sentiment updates."""
        self.callbacks.append(callback)
        logger.info("✅ Registered sentiment analysis callback")
    
    def analyze_transcript(self, session_id: str, transcript_data: Dict):
        """Queue transcript for sentiment analysis."""
        try:
            analysis_item = {
                'session_id': session_id,
                'transcript': transcript_data,
                'timestamp': datetime.now().isoformat()
            }
            
            self.sentiment_queue.put(analysis_item)
            
        except Exception as e:
            logger.error(f"❌ Error queuing transcript for analysis: {e}")
    
    def _process_sentiment_queue(self):
        """Process sentiment analysis queue."""
        while self.running:
            try:
                # Get item from queue with timeout
                analysis_item = self.sentiment_queue.get(timeout=1)
                
                # Perform sentiment analysis
                result = self._perform_sentiment_analysis(analysis_item)
                
                # Store result
                self._store_sentiment_result(analysis_item['session_id'], result)
                
                # Trigger callbacks
                for callback in self.callbacks:
                    try:
                        callback(analysis_item['session_id'], result)
                    except Exception as e:
                        logger.error(f"Error in sentiment callback: {e}")
                
                # Mark task as done
                self.sentiment_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ Error processing sentiment queue: {e}")
    
    def _perform_sentiment_analysis(self, analysis_item: Dict) -> Dict:
        """Perform sentiment analysis on transcript."""
        try:
            session_id = analysis_item['session_id']
            transcript = analysis_item['transcript']
            text = transcript['text'].lower()
            speaker = transcript.get('speaker', 'Unknown')
            
            # Basic sentiment scoring
            sentiment_score = self._calculate_sentiment_score(text)
            
            # Emotion detection
            emotions = self._detect_emotions(text)
            
            # Context analysis
            context = self._analyze_context(text)
            
            # Speaker-specific analysis
            speaker_sentiment = self._get_speaker_sentiment_trend(session_id, speaker)
            
            # Confidence calculation
            confidence = self._calculate_confidence(text, sentiment_score)
            
            # Determine sentiment category
            sentiment_category = self._categorize_sentiment(sentiment_score)
            
            # Calculate intensity
            intensity = self._calculate_intensity(text, sentiment_score)
            
            result = {
                'session_id': session_id,
                'speaker': speaker,
                'text': transcript['text'],
                'timestamp': analysis_item['timestamp'],
                'sentiment': {
                    'score': sentiment_score,
                    'category': sentiment_category,
                    'intensity': intensity,
                    'confidence': confidence
                },
                'emotions': emotions,
                'context': context,
                'speaker_trend': speaker_sentiment,
                'word_count': len(text.split()),
                'analysis_metadata': {
                    'processing_time': time.time(),
                    'text_length': len(text),
                    'contains_question': '?' in transcript['text'],
                    'contains_exclamation': '!' in transcript['text'],
                    'caps_ratio': self._calculate_caps_ratio(transcript['text'])
                }
            }
            
            logger.info(f"📊 Sentiment analysis complete for {session_id}: {sentiment_category} ({sentiment_score:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in sentiment analysis: {e}")
            return {
                'session_id': analysis_item['session_id'],
                'error': str(e),
                'timestamp': analysis_item['timestamp']
            }
    
    def _calculate_sentiment_score(self, text: str) -> float:
        """Calculate sentiment score for text."""
        try:
            words = re.findall(r'\b\w+\b', text.lower())
            
            positive_count = sum(1 for word in words if word in self.positive_words)
            negative_count = sum(1 for word in words if word in self.negative_words)
            neutral_count = sum(1 for word in words if word in self.neutral_words)
            
            total_sentiment_words = positive_count + negative_count + neutral_count
            
            if total_sentiment_words == 0:
                return 0.0
            
            # Base score
            base_score = (positive_count - negative_count) / len(words)
            
            # Apply modifiers
            score = base_score
            
            # Exclamation boost
            if '!' in text:
                score += self.sentiment_weights['exclamation_boost']
            
            # Question penalty (uncertainty)
            if '?' in text:
                score += self.sentiment_weights['question_penalty']
            
            # Caps penalty (aggressive tone)
            caps_ratio = self._calculate_caps_ratio(text)
            if caps_ratio > 0.3:
                score += self.sentiment_weights['caps_penalty']
            
            # Repetition penalty
            if self._has_repetition(text):
                score += self.sentiment_weights['repetition_penalty']
            
            # Normalize to [-1, 1]
            score = max(-1.0, min(1.0, score))
            
            return score
            
        except Exception as e:
            logger.error(f"❌ Error calculating sentiment score: {e}")
            return 0.0
    
    def _detect_emotions(self, text: str) -> Dict:
        """Detect emotions in text."""
        try:
            words = set(re.findall(r'\b\w+\b', text.lower()))
            
            emotion_scores = {}
            
            for emotion, emotion_words in self.emotion_patterns.items():
                matches = len(words.intersection(emotion_words))
                score = matches / len(words) if words else 0.0
                emotion_scores[emotion] = score
            
            # Find dominant emotion
            dominant_emotion = max(emotion_scores, key=emotion_scores.get)
            dominant_score = emotion_scores[dominant_emotion]
            
            return {
                'scores': emotion_scores,
                'dominant_emotion': dominant_emotion,
                'dominant_score': dominant_score,
                'emotion_intensity': self._calculate_emotion_intensity(emotion_scores)
            }
            
        except Exception as e:
            logger.error(f"❌ Error detecting emotions: {e}")
            return {'scores': {}, 'dominant_emotion': 'neutral', 'dominant_score': 0.0}
    
    def _analyze_context(self, text: str) -> Dict:
        """Analyze context and tone indicators."""
        try:
            context = {
                'is_question': '?' in text,
                'is_exclamation': '!' in text,
                'is_formal': self._is_formal_language(text),
                'is_urgent': self._is_urgent_language(text),
                'is_polite': self._is_polite_language(text),
                'contains_negation': self._contains_negation(text),
                'word_count': len(text.split()),
                'sentence_count': len(re.split(r'[.!?]+', text)),
                'avg_word_length': self._calculate_avg_word_length(text),
                'complexity_score': self._calculate_complexity_score(text)
            }
            
            return context
            
        except Exception as e:
            logger.error(f"❌ Error analyzing context: {e}")
            return {}
    
    def _get_speaker_sentiment_trend(self, session_id: str, speaker: str) -> Dict:
        """Get sentiment trend for specific speaker."""
        try:
            if session_id not in self.session_data:
                return {'trend': 'neutral', 'change': 0.0, 'history_length': 0}
            
            speaker_history = [
                entry for entry in self.session_data[session_id]['sentiment_history']
                if entry.get('speaker') == speaker
            ]
            
            if len(speaker_history) < 2:
                return {'trend': 'neutral', 'change': 0.0, 'history_length': len(speaker_history)}
            
            # Calculate trend
            recent_scores = [entry['sentiment']['score'] for entry in speaker_history[-5:]]
            trend_change = recent_scores[-1] - recent_scores[0] if len(recent_scores) > 1 else 0.0
            
            if trend_change > 0.1:
                trend = 'improving'
            elif trend_change < -0.1:
                trend = 'declining'
            else:
                trend = 'stable'
            
            return {
                'trend': trend,
                'change': trend_change,
                'history_length': len(speaker_history),
                'avg_sentiment': statistics.mean(recent_scores),
                'sentiment_variance': statistics.variance(recent_scores) if len(recent_scores) > 1 else 0.0
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting speaker sentiment trend: {e}")
            return {'trend': 'neutral', 'change': 0.0, 'history_length': 0}
    
    def _calculate_confidence(self, text: str, sentiment_score: float) -> float:
        """Calculate confidence in sentiment analysis."""
        try:
            # Base confidence from sentiment strength
            base_confidence = min(abs(sentiment_score) * 2, 1.0)
            
            # Adjust based on text characteristics
            words = text.split()
            word_count = len(words)
            
            # More words generally mean higher confidence
            word_confidence = min(word_count / 20, 1.0)
            
            # Presence of clear sentiment indicators
            sentiment_words = sum(1 for word in words if word.lower() in 
                                (self.positive_words | self.negative_words))
            sentiment_confidence = min(sentiment_words / word_count, 1.0) if word_count > 0 else 0.0
            
            # Combine confidences
            confidence = (base_confidence + word_confidence + sentiment_confidence) / 3
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"❌ Error calculating confidence: {e}")
            return 0.5
    
    def _categorize_sentiment(self, score: float) -> str:
        """Categorize sentiment score."""
        if score > 0.3:
            return 'positive'
        elif score < -0.3:
            return 'negative'
        else:
            return 'neutral'
    
    def _calculate_intensity(self, text: str, score: float) -> str:
        """Calculate sentiment intensity."""
        abs_score = abs(score)
        
        if abs_score > 0.7:
            return 'high'
        elif abs_score > 0.3:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_caps_ratio(self, text: str) -> float:
        """Calculate ratio of capital letters."""
        if not text:
            return 0.0
        
        caps_count = sum(1 for c in text if c.isupper())
        return caps_count / len(text)
    
    def _has_repetition(self, text: str) -> bool:
        """Check for word repetition."""
        words = text.lower().split()
        return len(words) != len(set(words))
    
    def _calculate_emotion_intensity(self, emotion_scores: Dict) -> float:
        """Calculate overall emotion intensity."""
        if not emotion_scores:
            return 0.0
        
        return max(emotion_scores.values())
    
    def _is_formal_language(self, text: str) -> bool:
        """Check if language is formal."""
        formal_indicators = {'please', 'thank', 'sir', 'madam', 'would', 'could', 'may'}
        words = set(text.lower().split())
        return len(words.intersection(formal_indicators)) > 0
    
    def _is_urgent_language(self, text: str) -> bool:
        """Check if language indicates urgency."""
        urgent_indicators = {'urgent', 'asap', 'immediately', 'now', 'quickly', 'emergency'}
        words = set(text.lower().split())
        return len(words.intersection(urgent_indicators)) > 0
    
    def _is_polite_language(self, text: str) -> bool:
        """Check if language is polite."""
        polite_indicators = {'please', 'thank', 'sorry', 'excuse', 'pardon', 'kindly'}
        words = set(text.lower().split())
        return len(words.intersection(polite_indicators)) > 0
    
    def _contains_negation(self, text: str) -> bool:
        """Check if text contains negation."""
        negation_words = {'not', 'no', 'never', 'nothing', 'nobody', 'none', 'neither', 'nor'}
        words = set(text.lower().split())
        return len(words.intersection(negation_words)) > 0
    
    def _calculate_avg_word_length(self, text: str) -> float:
        """Calculate average word length."""
        words = text.split()
        if not words:
            return 0.0
        
        return sum(len(word) for word in words) / len(words)
    
    def _calculate_complexity_score(self, text: str) -> float:
        """Calculate text complexity score."""
        words = text.split()
        if not words:
            return 0.0
        
        # Simple complexity based on word length and sentence structure
        avg_word_length = self._calculate_avg_word_length(text)
        sentence_count = len(re.split(r'[.!?]+', text))
        words_per_sentence = len(words) / sentence_count if sentence_count > 0 else 0
        
        # Normalize to 0-1 scale
        complexity = (avg_word_length / 10 + words_per_sentence / 20) / 2
        return min(1.0, complexity)
    
    def _store_sentiment_result(self, session_id: str, result: Dict):
        """Store sentiment analysis result."""
        try:
            if session_id not in self.session_data:
                self.session_data[session_id] = {
                    'sentiment_history': deque(maxlen=100),
                    'speaker_stats': defaultdict(list),
                    'session_start': datetime.now().isoformat()
                }
            
            # Add to history
            self.session_data[session_id]['sentiment_history'].append(result)
            
            # Update speaker stats
            speaker = result.get('speaker', 'Unknown')
            self.session_data[session_id]['speaker_stats'][speaker].append(result['sentiment'])
            
            # Keep only last 50 entries per speaker
            if len(self.session_data[session_id]['speaker_stats'][speaker]) > 50:
                self.session_data[session_id]['speaker_stats'][speaker] = \
                    self.session_data[session_id]['speaker_stats'][speaker][-50:]
            
        except Exception as e:
            logger.error(f"❌ Error storing sentiment result: {e}")
    
    def get_session_sentiment_summary(self, session_id: str) -> Dict:
        """Get sentiment summary for session."""
        try:
            if session_id not in self.session_data:
                return {'error': 'Session not found'}
            
            session_data = self.session_data[session_id]
            sentiment_history = list(session_data['sentiment_history'])
            
            if not sentiment_history:
                return {'error': 'No sentiment data available'}
            
            # Overall statistics
            all_scores = [entry['sentiment']['score'] for entry in sentiment_history]
            
            overall_stats = {
                'avg_sentiment': statistics.mean(all_scores),
                'sentiment_variance': statistics.variance(all_scores) if len(all_scores) > 1 else 0.0,
                'min_sentiment': min(all_scores),
                'max_sentiment': max(all_scores),
                'total_entries': len(sentiment_history)
            }
            
            # Category distribution
            categories = [entry['sentiment']['category'] for entry in sentiment_history]
            category_counts = {
                'positive': categories.count('positive'),
                'negative': categories.count('negative'),
                'neutral': categories.count('neutral')
            }
            
            # Speaker-specific stats
            speaker_stats = {}
            for speaker, speaker_sentiments in session_data['speaker_stats'].items():
                scores = [s['score'] for s in speaker_sentiments]
                speaker_stats[speaker] = {
                    'avg_sentiment': statistics.mean(scores),
                    'sentiment_variance': statistics.variance(scores) if len(scores) > 1 else 0.0,
                    'entry_count': len(scores),
                    'dominant_category': max(
                        ['positive', 'negative', 'neutral'],
                        key=lambda cat: sum(1 for s in speaker_sentiments if s['category'] == cat)
                    )
                }
            
            # Recent trend
            recent_scores = all_scores[-10:] if len(all_scores) >= 10 else all_scores
            trend_change = recent_scores[-1] - recent_scores[0] if len(recent_scores) > 1 else 0.0
            
            if trend_change > 0.1:
                trend = 'improving'
            elif trend_change < -0.1:
                trend = 'declining'
            else:
                trend = 'stable'
            
            return {
                'session_id': session_id,
                'overall_stats': overall_stats,
                'category_distribution': category_counts,
                'speaker_stats': speaker_stats,
                'recent_trend': {
                    'trend': trend,
                    'change': trend_change
                },
                'session_start': session_data['session_start'],
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting sentiment summary: {e}")
            return {'error': str(e)}
    
    def get_live_sentiment_data(self, session_id: str, limit: int = 10) -> Dict:
        """Get live sentiment data for real-time display."""
        try:
            if session_id not in self.session_data:
                return {'error': 'Session not found'}
            
            sentiment_history = list(self.session_data[session_id]['sentiment_history'])
            
            # Get recent entries
            recent_entries = sentiment_history[-limit:] if sentiment_history else []
            
            # Current sentiment
            current_sentiment = recent_entries[-1]['sentiment'] if recent_entries else None
            
            # Sentiment timeline
            timeline = [
                {
                    'timestamp': entry['timestamp'],
                    'speaker': entry['speaker'],
                    'sentiment_score': entry['sentiment']['score'],
                    'sentiment_category': entry['sentiment']['category'],
                    'emotions': entry['emotions']['dominant_emotion']
                }
                for entry in recent_entries
            ]
            
            return {
                'session_id': session_id,
                'current_sentiment': current_sentiment,
                'timeline': timeline,
                'total_entries': len(sentiment_history),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting live sentiment data: {e}")
            return {'error': str(e)}

# Global sentiment analyzer instance
sentiment_analyzer = RealTimeSentimentAnalysis()

# Integration functions
def start_sentiment_analysis():
    """Start sentiment analysis."""
    sentiment_analyzer.start_analysis()

def stop_sentiment_analysis():
    """Stop sentiment analysis."""
    sentiment_analyzer.stop_analysis()

def analyze_transcript_sentiment(session_id: str, transcript_data: Dict):
    """Analyze transcript sentiment."""
    sentiment_analyzer.analyze_transcript(session_id, transcript_data)

def get_sentiment_summary(session_id: str) -> Dict:
    """Get sentiment summary."""
    return sentiment_analyzer.get_session_sentiment_summary(session_id)

def get_live_sentiment(session_id: str, limit: int = 10) -> Dict:
    """Get live sentiment data."""
    return sentiment_analyzer.get_live_sentiment_data(session_id, limit)

def register_sentiment_callback(callback):
    """Register sentiment callback."""
    sentiment_analyzer.register_callback(callback)

if __name__ == "__main__":
    # Test the sentiment analyzer
    print("🚀 Testing Real-time Sentiment Analysis...")
    
    # Start analyzer
    sentiment_analyzer.start_analysis()
    
    # Test sentences
    test_sentences = [
        {"text": "I love this service! It's absolutely amazing!", "speaker": "Customer"},
        {"text": "I'm having trouble with my account. This is frustrating.", "speaker": "Customer"},
        {"text": "Let me help you with that right away.", "speaker": "Agent"},
        {"text": "Thank you so much for your patience.", "speaker": "Agent"},
        {"text": "This is the worst experience I've ever had!", "speaker": "Customer"}
    ]
    
    # Analyze test sentences
    for i, sentence in enumerate(test_sentences):
        transcript_data = {
            'text': sentence['text'],
            'speaker': sentence['speaker'],
            'timestamp': datetime.now().isoformat(),
            'is_final': True
        }
        
        sentiment_analyzer.analyze_transcript(f"test_session", transcript_data)
        time.sleep(0.5)  # Small delay to see processing
    
    # Get summary
    time.sleep(2)  # Wait for processing
    summary = sentiment_analyzer.get_session_sentiment_summary("test_session")
    print(f"📊 Sentiment Summary: {summary}")
    
    # Stop analyzer
    sentiment_analyzer.stop_analysis()
    
    print("🎉 Real-time Sentiment Analysis test completed!") 