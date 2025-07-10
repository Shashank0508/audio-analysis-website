#!/usr/bin/env python3
"""
Real-time Analysis Integration
Connects live audio stream to existing AI analysis system for real-time processing.
"""

import os
import sys
import json
import time
import asyncio
import websockets
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealTimeAnalysisIntegrator:
    """Integrates live audio stream with existing AI analysis system."""
    
    def __init__(self):
        self.active_sessions = {}
        self.analysis_queue = asyncio.Queue()
        self.websocket_server = None
        self.analysis_thread = None
        self.running = False
        
        # Import existing analysis modules
        self.setup_analysis_modules()
        
    def setup_analysis_modules(self):
        """Setup existing AI analysis modules."""
        try:
            # Import existing analysis functions from app.py
            import sys
            import os
            
            # Add current directory to path
            sys.path.insert(0, os.getcwd())
            
            # Import app module
            import app
            
            # Use existing functions from app.py
            self.sentiment_analyzer = app.analyze_sentiment
            self.emotion_detector = app.analyze_emotions
            self.keyword_extractor = app.extract_key_topics
            self.topic_classifier = app.extract_key_topics
            
            logger.info("✅ Existing AI analysis modules loaded successfully")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not import analysis modules: {e}")
            # Fallback to mock analysis
            self.setup_mock_analysis()
    
    def setup_mock_analysis(self):
        """Setup mock analysis functions for testing."""
        def mock_sentiment(text):
            return {
                'sentiment': 'neutral',
                'confidence': 0.75,
                'scores': {'positive': 0.3, 'neutral': 0.5, 'negative': 0.2}
            }
        
        def mock_emotion(text):
            return {
                'dominant_emotion': 'neutral',
                'confidence': 0.8,
                'emotions': {
                    'joy': 0.2, 'sadness': 0.1, 'anger': 0.1, 'fear': 0.1,
                    'surprise': 0.1, 'disgust': 0.1, 'neutral': 0.3
                }
            }
        
        def mock_keywords(text):
            words = text.lower().split()
            return {
                'keywords': words[:5],
                'phrases': [' '.join(words[i:i+2]) for i in range(0, min(len(words)-1, 3))],
                'topics': ['business', 'conversation']
            }
        
        def mock_topics(text):
            return {
                'primary_topic': 'business',
                'confidence': 0.7,
                'topics': ['business', 'conversation', 'general']
            }
        
        self.sentiment_analyzer = mock_sentiment
        self.emotion_detector = mock_emotion
        self.keyword_extractor = mock_keywords
        self.topic_classifier = mock_topics
        
        logger.info("✅ Mock analysis functions setup for testing")
    
    async def process_live_transcript(self, session_id: str, transcript_data: Dict):
        """Process live transcript with AI analysis."""
        try:
            text = transcript_data.get('text', '')
            speaker = transcript_data.get('speaker', 'Unknown')
            timestamp = transcript_data.get('timestamp', datetime.now().isoformat())
            is_final = transcript_data.get('is_final', False)
            
            # Only analyze final transcripts for accuracy
            if not is_final or len(text.strip()) < 10:
                return
            
            logger.info(f"🔍 Analyzing transcript for session {session_id}: {text[:50]}...")
            
            # Run AI analysis
            analysis_results = await self.run_ai_analysis(text)
            
            # Prepare real-time analysis data
            realtime_data = {
                'session_id': session_id,
                'timestamp': timestamp,
                'speaker': speaker,
                'text': text,
                'analysis': analysis_results,
                'type': 'realtime_analysis'
            }
            
            # Send to connected clients
            await self.broadcast_analysis(session_id, realtime_data)
            
            # Store in session data
            if session_id in self.active_sessions:
                self.active_sessions[session_id]['analysis_history'].append(realtime_data)
                await self.update_session_metrics(session_id, analysis_results)
            
            logger.info(f"✅ Analysis completed for session {session_id}")
            
        except Exception as e:
            logger.error(f"❌ Error processing transcript: {e}")
    
    async def run_ai_analysis(self, text: str) -> Dict:
        """Run comprehensive AI analysis on text."""
        try:
            # Run analysis in parallel for better performance
            tasks = [
                asyncio.create_task(self.async_sentiment_analysis(text)),
                asyncio.create_task(self.async_emotion_detection(text)),
                asyncio.create_task(self.async_keyword_extraction(text)),
                asyncio.create_task(self.async_topic_classification(text))
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine results
            analysis = {
                'sentiment': results[0] if not isinstance(results[0], Exception) else None,
                'emotion': results[1] if not isinstance(results[1], Exception) else None,
                'keywords': results[2] if not isinstance(results[2], Exception) else None,
                'topics': results[3] if not isinstance(results[3], Exception) else None,
                'timestamp': datetime.now().isoformat()
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error in AI analysis: {e}")
            return {'error': str(e)}
    
    async def async_sentiment_analysis(self, text: str):
        """Async wrapper for sentiment analysis."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.sentiment_analyzer, text)
    
    async def async_emotion_detection(self, text: str):
        """Async wrapper for emotion detection."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.emotion_detector, text)
    
    async def async_keyword_extraction(self, text: str):
        """Async wrapper for keyword extraction."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.keyword_extractor, text)
    
    async def async_topic_classification(self, text: str):
        """Async wrapper for topic classification."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.topic_classifier, text)
    
    async def update_session_metrics(self, session_id: str, analysis: Dict):
        """Update session-level metrics and insights."""
        try:
            session = self.active_sessions[session_id]
            
            # Update sentiment metrics
            if analysis.get('sentiment'):
                sentiment_data = analysis['sentiment']
                session['metrics']['sentiment_history'].append(sentiment_data)
                session['metrics']['avg_sentiment'] = self.calculate_avg_sentiment(
                    session['metrics']['sentiment_history']
                )
            
            # Update emotion metrics
            if analysis.get('emotion'):
                emotion_data = analysis['emotion']
                session['metrics']['emotion_history'].append(emotion_data)
                session['metrics']['dominant_emotions'] = self.calculate_dominant_emotions(
                    session['metrics']['emotion_history']
                )
            
            # Update keyword metrics
            if analysis.get('keywords'):
                keywords = analysis['keywords'].get('keywords', [])
                session['metrics']['all_keywords'].extend(keywords)
                session['metrics']['trending_keywords'] = self.calculate_trending_keywords(
                    session['metrics']['all_keywords']
                )
            
            # Update topic metrics
            if analysis.get('topics'):
                topics = analysis['topics'].get('topics', [])
                session['metrics']['all_topics'].extend(topics)
                session['metrics']['topic_distribution'] = self.calculate_topic_distribution(
                    session['metrics']['all_topics']
                )
            
            # Generate insights
            session['insights'] = await self.generate_session_insights(session_id)
            
        except Exception as e:
            logger.error(f"❌ Error updating session metrics: {e}")
    
    def calculate_avg_sentiment(self, sentiment_history: List[Dict]) -> Dict:
        """Calculate average sentiment from history."""
        if not sentiment_history:
            return {'sentiment': 'neutral', 'confidence': 0.0}
        
        total_positive = sum(s.get('scores', {}).get('positive', 0) for s in sentiment_history)
        total_negative = sum(s.get('scores', {}).get('negative', 0) for s in sentiment_history)
        total_neutral = sum(s.get('scores', {}).get('neutral', 0) for s in sentiment_history)
        
        count = len(sentiment_history)
        avg_scores = {
            'positive': total_positive / count,
            'negative': total_negative / count,
            'neutral': total_neutral / count
        }
        
        # Determine overall sentiment
        max_score = max(avg_scores.values())
        overall_sentiment = [k for k, v in avg_scores.items() if v == max_score][0]
        
        return {
            'sentiment': overall_sentiment,
            'confidence': max_score,
            'scores': avg_scores
        }
    
    def calculate_dominant_emotions(self, emotion_history: List[Dict]) -> Dict:
        """Calculate dominant emotions from history."""
        if not emotion_history:
            return {}
        
        emotion_totals = {}
        for emotion_data in emotion_history:
            emotions = emotion_data.get('emotions', {})
            for emotion, score in emotions.items():
                emotion_totals[emotion] = emotion_totals.get(emotion, 0) + score
        
        # Calculate averages
        count = len(emotion_history)
        emotion_averages = {emotion: total / count for emotion, total in emotion_totals.items()}
        
        # Sort by score
        sorted_emotions = sorted(emotion_averages.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'top_emotions': sorted_emotions[:3],
            'all_emotions': emotion_averages
        }
    
    def calculate_trending_keywords(self, all_keywords: List[str]) -> List[Dict]:
        """Calculate trending keywords with frequency."""
        if not all_keywords:
            return []
        
        # Count frequency
        keyword_counts = {}
        for keyword in all_keywords:
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        # Sort by frequency
        sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {'keyword': keyword, 'frequency': count}
            for keyword, count in sorted_keywords[:10]
        ]
    
    def calculate_topic_distribution(self, all_topics: List[str]) -> Dict:
        """Calculate topic distribution."""
        if not all_topics:
            return {}
        
        topic_counts = {}
        for topic in all_topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        total = len(all_topics)
        topic_percentages = {
            topic: (count / total) * 100
            for topic, count in topic_counts.items()
        }
        
        return topic_percentages
    
    async def generate_session_insights(self, session_id: str) -> Dict:
        """Generate comprehensive session insights."""
        try:
            session = self.active_sessions[session_id]
            metrics = session['metrics']
            
            insights = {
                'overall_sentiment': metrics.get('avg_sentiment', {}),
                'emotional_state': metrics.get('dominant_emotions', {}),
                'key_topics': list(metrics.get('topic_distribution', {}).keys())[:3],
                'trending_keywords': [k['keyword'] for k in metrics.get('trending_keywords', [])[:5]],
                'conversation_tone': self.determine_conversation_tone(metrics),
                'engagement_level': self.calculate_engagement_level(metrics),
                'recommendations': self.generate_recommendations(metrics)
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"❌ Error generating insights: {e}")
            return {}
    
    def determine_conversation_tone(self, metrics: Dict) -> str:
        """Determine overall conversation tone."""
        avg_sentiment = metrics.get('avg_sentiment', {})
        sentiment = avg_sentiment.get('sentiment', 'neutral')
        confidence = avg_sentiment.get('confidence', 0)
        
        if confidence > 0.7:
            if sentiment == 'positive':
                return 'positive'
            elif sentiment == 'negative':
                return 'negative'
        
        return 'neutral'
    
    def calculate_engagement_level(self, metrics: Dict) -> str:
        """Calculate engagement level based on various factors."""
        keyword_count = len(metrics.get('all_keywords', []))
        topic_variety = len(metrics.get('topic_distribution', {}))
        
        if keyword_count > 50 and topic_variety > 3:
            return 'high'
        elif keyword_count > 20 and topic_variety > 2:
            return 'medium'
        else:
            return 'low'
    
    def generate_recommendations(self, metrics: Dict) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Sentiment-based recommendations
        avg_sentiment = metrics.get('avg_sentiment', {})
        if avg_sentiment.get('sentiment') == 'negative':
            recommendations.append("Consider addressing concerns to improve conversation tone")
        
        # Engagement-based recommendations
        engagement = self.calculate_engagement_level(metrics)
        if engagement == 'low':
            recommendations.append("Try asking more engaging questions to increase participation")
        
        # Topic-based recommendations
        topics = metrics.get('topic_distribution', {})
        if 'business' in topics and topics['business'] > 50:
            recommendations.append("Good focus on business topics - maintain professional tone")
        
        return recommendations
    
    async def broadcast_analysis(self, session_id: str, data: Dict):
        """Broadcast analysis results to connected clients."""
        try:
            message = json.dumps(data)
            
            # Send to WebSocket clients (if implemented)
            if hasattr(self, 'websocket_clients'):
                for client in self.websocket_clients:
                    try:
                        await client.send(message)
                    except:
                        pass
            
            # Send to Flask-SocketIO (if available)
            try:
                from flask_socketio import emit
                emit('realtime_analysis', data, room=session_id)
            except ImportError:
                pass
            
            logger.info(f"📡 Broadcasted analysis for session {session_id}")
            
        except Exception as e:
            logger.error(f"❌ Error broadcasting analysis: {e}")
    
    def start_session(self, session_id: str, call_data: Dict) -> Dict:
        """Start a new real-time analysis session."""
        try:
            session = {
                'session_id': session_id,
                'start_time': datetime.now().isoformat(),
                'call_data': call_data,
                'analysis_history': [],
                'metrics': {
                    'sentiment_history': [],
                    'emotion_history': [],
                    'all_keywords': [],
                    'all_topics': [],
                    'avg_sentiment': {},
                    'dominant_emotions': {},
                    'trending_keywords': [],
                    'topic_distribution': {}
                },
                'insights': {},
                'status': 'active'
            }
            
            self.active_sessions[session_id] = session
            logger.info(f"✅ Started real-time analysis session: {session_id}")
            
            return session
            
        except Exception as e:
            logger.error(f"❌ Error starting session: {e}")
            return {}
    
    def end_session(self, session_id: str) -> Dict:
        """End a real-time analysis session."""
        try:
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                session['end_time'] = datetime.now().isoformat()
                session['status'] = 'completed'
                
                # Generate final report
                final_report = self.generate_final_report(session)
                session['final_report'] = final_report
                
                # Archive session
                archived_session = self.active_sessions.pop(session_id)
                
                logger.info(f"✅ Ended real-time analysis session: {session_id}")
                return archived_session
            
            return {}
            
        except Exception as e:
            logger.error(f"❌ Error ending session: {e}")
            return {}
    
    def generate_final_report(self, session: Dict) -> Dict:
        """Generate comprehensive final analysis report."""
        try:
            metrics = session['metrics']
            analysis_history = session['analysis_history']
            
            report = {
                'session_summary': {
                    'session_id': session['session_id'],
                    'duration': self.calculate_session_duration(session),
                    'total_analyses': len(analysis_history),
                    'status': 'completed'
                },
                'sentiment_analysis': {
                    'overall_sentiment': metrics.get('avg_sentiment', {}),
                    'sentiment_trend': self.calculate_sentiment_trend(metrics.get('sentiment_history', [])),
                    'sentiment_changes': len(set(s.get('sentiment') for s in metrics.get('sentiment_history', [])))
                },
                'emotion_analysis': {
                    'dominant_emotions': metrics.get('dominant_emotions', {}),
                    'emotion_stability': self.calculate_emotion_stability(metrics.get('emotion_history', []))
                },
                'content_analysis': {
                    'key_topics': metrics.get('topic_distribution', {}),
                    'trending_keywords': metrics.get('trending_keywords', []),
                    'topic_diversity': len(metrics.get('topic_distribution', {}))
                },
                'insights': session.get('insights', {}),
                'recommendations': session.get('insights', {}).get('recommendations', []),
                'generated_at': datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Error generating final report: {e}")
            return {}
    
    def calculate_session_duration(self, session: Dict) -> str:
        """Calculate session duration."""
        try:
            start_time = datetime.fromisoformat(session['start_time'])
            end_time = datetime.fromisoformat(session.get('end_time', datetime.now().isoformat()))
            duration = end_time - start_time
            
            total_seconds = int(duration.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            
            if hours > 0:
                return f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                return f"{minutes}m {seconds}s"
            else:
                return f"{seconds}s"
                
        except Exception as e:
            logger.error(f"❌ Error calculating duration: {e}")
            return "Unknown"
    
    def calculate_sentiment_trend(self, sentiment_history: List[Dict]) -> str:
        """Calculate sentiment trend over time."""
        if len(sentiment_history) < 2:
            return "stable"
        
        # Compare first and last sentiment
        first_sentiment = sentiment_history[0].get('sentiment', 'neutral')
        last_sentiment = sentiment_history[-1].get('sentiment', 'neutral')
        
        sentiment_scores = {'positive': 1, 'neutral': 0, 'negative': -1}
        
        first_score = sentiment_scores.get(first_sentiment, 0)
        last_score = sentiment_scores.get(last_sentiment, 0)
        
        if last_score > first_score:
            return "improving"
        elif last_score < first_score:
            return "declining"
        else:
            return "stable"
    
    def calculate_emotion_stability(self, emotion_history: List[Dict]) -> str:
        """Calculate emotion stability."""
        if len(emotion_history) < 2:
            return "stable"
        
        # Calculate variance in dominant emotions
        dominant_emotions = [e.get('dominant_emotion', 'neutral') for e in emotion_history]
        unique_emotions = set(dominant_emotions)
        
        if len(unique_emotions) <= 2:
            return "stable"
        elif len(unique_emotions) <= 4:
            return "moderate"
        else:
            return "variable"
    
    def get_session_status(self, session_id: str) -> Dict:
        """Get current session status and metrics."""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            return {
                'session_id': session_id,
                'status': session['status'],
                'start_time': session['start_time'],
                'analyses_count': len(session['analysis_history']),
                'current_metrics': session['metrics'],
                'insights': session['insights']
            }
        return {'error': 'Session not found'}
    
    def get_all_sessions(self) -> List[Dict]:
        """Get all active sessions."""
        return [
            {
                'session_id': session_id,
                'status': session['status'],
                'start_time': session['start_time'],
                'analyses_count': len(session['analysis_history'])
            }
            for session_id, session in self.active_sessions.items()
        ]

# Global instance
realtime_integrator = RealTimeAnalysisIntegrator()

# Flask integration functions
def start_realtime_analysis(session_id: str, call_data: Dict) -> Dict:
    """Start real-time analysis for a call session."""
    return realtime_integrator.start_session(session_id, call_data)

def process_transcript_analysis(session_id: str, transcript_data: Dict):
    """Process transcript with real-time analysis."""
    asyncio.create_task(realtime_integrator.process_live_transcript(session_id, transcript_data))

def end_realtime_analysis(session_id: str) -> Dict:
    """End real-time analysis session."""
    return realtime_integrator.end_session(session_id)

def get_session_insights(session_id: str) -> Dict:
    """Get current session insights."""
    return realtime_integrator.get_session_status(session_id)

def get_active_sessions() -> List[Dict]:
    """Get all active analysis sessions."""
    return realtime_integrator.get_all_sessions()

if __name__ == "__main__":
    # Test the integration
    print("🚀 Testing Real-time Analysis Integration...")
    
    # Start a test session
    test_session = realtime_integrator.start_session("test_session", {"caller": "+1234567890"})
    print(f"✅ Test session started: {test_session['session_id']}")
    
    # Process some test transcripts
    test_transcripts = [
        {"text": "Hello, I'm calling about my account", "speaker": "Customer", "is_final": True},
        {"text": "I'm having trouble with my recent order", "speaker": "Customer", "is_final": True},
        {"text": "I understand your concern, let me help you", "speaker": "Agent", "is_final": True}
    ]
    
    async def test_processing():
        for transcript in test_transcripts:
            await realtime_integrator.process_live_transcript("test_session", transcript)
            await asyncio.sleep(1)
    
    # Run test
    asyncio.run(test_processing())
    
    # Get final report
    final_session = realtime_integrator.end_session("test_session")
    print(f"✅ Test completed. Final report generated.")
    print(f"📊 Session insights: {final_session.get('insights', {})}")
    
    print("🎉 Real-time Analysis Integration ready!") 