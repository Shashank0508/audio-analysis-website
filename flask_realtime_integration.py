#!/usr/bin/env python3
"""
Flask Real-time Integration
Adds Flask routes and integration points for real-time analysis.
"""

from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit, join_room, leave_room
import json
import asyncio
import threading
import logging
from datetime import datetime
from typing import Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_realtime_routes(app, socketio):
    """Setup Flask routes for real-time analysis."""
    
    @app.route('/api/realtime/start', methods=['POST'])
    def start_realtime_analysis():
        """Start real-time analysis for a call session."""
        try:
            data = request.get_json()
            session_id = data.get('session_id')
            call_data = data.get('call_data', {})
            
            if not session_id:
                return jsonify({'error': 'session_id required'}), 400
            
            # Start all analysis systems
            start_all_analysis_systems()
            
            # Create session data
            session = {
                'session_id': session_id,
                'call_data': call_data,
                'start_time': datetime.now().isoformat(),
                'status': 'active'
            }
            
            return jsonify({
                'success': True,
                'session': session,
                'message': f'Real-time analysis started for session {session_id}'
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/realtime/process', methods=['POST'])
    def process_realtime_transcript():
        """Process real-time transcript data."""
        try:
            data = request.get_json()
            session_id = data.get('session_id')
            transcript_data = data.get('transcript_data', {})
            
            if not session_id or not transcript_data:
                return jsonify({'error': 'session_id and transcript_data required'}), 400
            
            # Process transcript through all analysis systems
            process_transcript_for_analysis(session_id, transcript_data)
            
            return jsonify({
                'success': True,
                'message': 'Transcript processing completed'
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/realtime/end', methods=['POST'])
    def end_realtime_analysis():
        """End real-time analysis session."""
        try:
            data = request.get_json()
            session_id = data.get('session_id')
            
            if not session_id:
                return jsonify({'error': 'session_id required'}), 400
            
            # Get final comprehensive analysis
            final_analysis = get_comprehensive_analysis(session_id)
            
            # Stop analysis systems (optional - they can continue for other sessions)
            # stop_all_analysis_systems()
            
            return jsonify({
                'success': True,
                'final_analysis': final_analysis,
                'message': f'Real-time analysis ended for session {session_id}'
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/realtime/status/<session_id>', methods=['GET'])
    def get_realtime_status(session_id):
        """Get real-time analysis status."""
        try:
            status = get_comprehensive_analysis(session_id)
            return jsonify(status)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/realtime/sessions', methods=['GET'])
    def get_active_sessions():
        """Get all active real-time analysis sessions."""
        try:
            # For now, return empty list - could be enhanced to track active sessions
            sessions = []
            return jsonify({'sessions': sessions})
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/test/realtime-analysis')
    def test_realtime_analysis():
        """Test page for real-time analysis."""
        return render_template('realtime_analysis_test.html')
    
    # WebSocket events
    @socketio.on('join_analysis_session')
    def on_join_analysis_session(data):
        """Join a real-time analysis session."""
        session_id = data.get('session_id')
        if session_id:
            join_room(session_id)
            emit('joined_session', {'session_id': session_id})
    
    @socketio.on('leave_analysis_session')
    def on_leave_analysis_session(data):
        """Leave a real-time analysis session."""
        session_id = data.get('session_id')
        if session_id:
            leave_room(session_id)
            emit('left_session', {'session_id': session_id})
    
    @socketio.on('request_analysis_update')
    def on_request_analysis_update(data):
        """Request current analysis update."""
        session_id = data.get('session_id')
        if session_id:
            status = get_comprehensive_analysis(session_id)
            emit('analysis_update', status, room=session_id)

def integrate_with_aws_transcribe():
    """Integrate with AWS Transcribe streaming for automatic processing."""
    
    # Modify AWS Transcribe streaming to automatically trigger analysis
    integration_code = '''
# Add this to aws_transcribe_streaming.py

from flask_realtime_integration import process_transcript_for_analysis

# In the WebSocket handler, add this after processing transcript:
async def process_transcript_with_analysis(session_id, transcript_data):
    """Process transcript and trigger real-time analysis."""
    try:
        # Send to real-time analysis
        await process_transcript_for_analysis(session_id, transcript_data)
        
        # Emit to WebSocket clients
        await websocket.send(json.dumps({
            'type': 'analysis_triggered',
            'session_id': session_id,
            'timestamp': datetime.now().isoformat()
        }))
        
    except Exception as e:
        logger.error(f"Error in transcript analysis: {e}")
'''
    
    return integration_code

def start_all_analysis_systems():
    """Start all real-time analysis systems."""
    try:
        # Import here to avoid circular imports
        from streaming_transcription_pipeline import streaming_pipeline
        from realtime_sentiment_analysis import start_sentiment_analysis
        from live_keyword_extraction import start_keyword_extraction
        from conversation_flow_analysis import start_flow_analysis
        
        # Start streaming transcription pipeline
        streaming_pipeline.start_pipeline()
        
        # Start sentiment analysis
        start_sentiment_analysis()
        
        # Start keyword extraction
        start_keyword_extraction()
        
        # Start flow analysis
        start_flow_analysis()
        
        # Register callbacks for integrated processing
        register_analysis_callbacks()
        
        logger.info("✅ All real-time analysis systems started successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error starting analysis systems: {e}")
        return False

def stop_all_analysis_systems():
    """Stop all real-time analysis systems."""
    try:
        # Import here to avoid circular imports
        from streaming_transcription_pipeline import streaming_pipeline
        from realtime_sentiment_analysis import sentiment_analyzer
        from live_keyword_extraction import keyword_extractor
        from conversation_flow_analysis import flow_analyzer
        
        # Stop all systems
        streaming_pipeline.stop_pipeline()
        sentiment_analyzer.stop_analysis()
        keyword_extractor.stop_extraction()
        flow_analyzer.stop_analysis()
        
        logger.info("⏹️ All real-time analysis systems stopped")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error stopping analysis systems: {e}")
        return False

def register_analysis_callbacks():
    """Register callbacks to connect all analysis systems."""
    try:
        # Import here to avoid circular imports
        from streaming_transcription_pipeline import streaming_pipeline
        from realtime_sentiment_analysis import sentiment_analyzer
        from live_keyword_extraction import keyword_extractor
        from conversation_flow_analysis import flow_analyzer
        
        # Register transcript callback to trigger all analyses
        streaming_pipeline.register_callback('on_transcript', process_transcript_for_analysis)
        
        # Register sentiment callback
        sentiment_analyzer.register_callback(handle_sentiment_update)
        
        # Register keyword callback
        keyword_extractor.register_callback(handle_keyword_update)
        
        # Register flow callback
        flow_analyzer.register_callback(handle_flow_update)
        
        logger.info("✅ Analysis callbacks registered successfully")
        
    except Exception as e:
        logger.error(f"❌ Error registering callbacks: {e}")

def process_transcript_for_analysis(session_id: str, transcript_data: Dict):
    """Process transcript through all analysis systems."""
    try:
        # Import here to avoid circular imports
        from realtime_sentiment_analysis import analyze_transcript_sentiment
        from live_keyword_extraction import extract_transcript_keywords
        from conversation_flow_analysis import analyze_conversation_flow
        
        # Send to sentiment analysis
        analyze_transcript_sentiment(session_id, transcript_data)
        
        # Send to keyword extraction
        extract_transcript_keywords(session_id, transcript_data)
        
        # Send to flow analysis
        analyze_conversation_flow(session_id, transcript_data)
        
        # Store integrated result
        store_integrated_analysis(session_id, transcript_data)
        
        logger.info(f"📊 Processed transcript through all analysis systems: {session_id}")
        
    except Exception as e:
        logger.error(f"❌ Error processing transcript for analysis: {e}")

def handle_sentiment_update(session_id: str, sentiment_data: Dict):
    """Handle sentiment analysis update."""
    try:
        # Emit to real-time dashboard
        emit_to_dashboard(session_id, {
            'type': 'sentiment_update',
            'data': sentiment_data
        })
        
    except Exception as e:
        logger.error(f"❌ Error handling sentiment update: {e}")

def handle_keyword_update(session_id: str, keyword_data: Dict):
    """Handle keyword extraction update."""
    try:
        # Emit to real-time dashboard
        emit_to_dashboard(session_id, {
            'type': 'keyword_update',
            'data': keyword_data
        })
        
    except Exception as e:
        logger.error(f"❌ Error handling keyword update: {e}")

def handle_flow_update(session_id: str, flow_data: Dict):
    """Handle conversation flow update."""
    try:
        # Emit to real-time dashboard
        emit_to_dashboard(session_id, {
            'type': 'flow_update',
            'data': flow_data
        })
        
    except Exception as e:
        logger.error(f"❌ Error handling flow update: {e}")

def get_comprehensive_analysis(session_id: str) -> Dict:
    """Get comprehensive analysis from all systems."""
    try:
        # Import here to avoid circular imports
        from realtime_sentiment_analysis import get_sentiment_summary
        from live_keyword_extraction import get_keyword_summary
        from conversation_flow_analysis import get_flow_summary
        
        # Get data from all systems
        sentiment_summary = get_sentiment_summary(session_id)
        keyword_summary = get_keyword_summary(session_id)
        flow_summary = get_flow_summary(session_id)
        
        # Combine into comprehensive analysis
        comprehensive = {
            'session_id': session_id,
            'sentiment_analysis': sentiment_summary,
            'keyword_analysis': keyword_summary,
            'flow_analysis': flow_summary,
            'integrated_insights': generate_integrated_insights(
                sentiment_summary, keyword_summary, flow_summary
            ),
            'timestamp': datetime.now().isoformat()
        }
        
        return comprehensive
        
    except Exception as e:
        logger.error(f"❌ Error getting comprehensive analysis: {e}")
        return {'error': str(e)}

def generate_integrated_insights(sentiment_data: Dict, keyword_data: Dict, flow_data: Dict) -> Dict:
    """Generate integrated insights from all analysis systems."""
    try:
        insights = {
            'overall_conversation_health': calculate_conversation_health(sentiment_data, flow_data),
            'key_topics_sentiment': correlate_topics_sentiment(keyword_data, sentiment_data),
            'conversation_progression': analyze_conversation_progression(flow_data, sentiment_data),
            'engagement_indicators': identify_engagement_indicators(sentiment_data, keyword_data, flow_data),
            'recommendations': generate_recommendations(sentiment_data, keyword_data, flow_data)
        }
        
        return insights
        
    except Exception as e:
        logger.error(f"❌ Error generating integrated insights: {e}")
        return {}

def calculate_conversation_health(sentiment_data: Dict, flow_data: Dict) -> Dict:
    """Calculate overall conversation health score."""
    try:
        # Get sentiment metrics
        avg_sentiment = sentiment_data.get('overall_stats', {}).get('avg_sentiment', 0.0)
        
        # Get flow metrics
        flow_metrics = flow_data.get('average_metrics', {})
        engagement = flow_metrics.get('engagement_level', 0.5)
        balance = flow_metrics.get('turn_taking_balance', 0.5)
        
        # Calculate health score
        health_score = (
            (avg_sentiment + 1) / 2 * 0.4 +  # Normalize sentiment to 0-1
            engagement * 0.3 +
            balance * 0.3
        )
        
        # Determine health category
        if health_score > 0.7:
            health_category = 'excellent'
        elif health_score > 0.5:
            health_category = 'good'
        elif health_score > 0.3:
            health_category = 'fair'
        else:
            health_category = 'poor'
        
        return {
            'health_score': health_score,
            'health_category': health_category,
            'contributing_factors': {
                'sentiment': avg_sentiment,
                'engagement': engagement,
                'balance': balance
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error calculating conversation health: {e}")
        return {'health_score': 0.5, 'health_category': 'unknown'}

def correlate_topics_sentiment(keyword_data: Dict, sentiment_data: Dict) -> Dict:
    """Correlate topics with sentiment."""
    try:
        # Get top keywords
        top_keywords = keyword_data.get('top_keywords', [])
        
        # Get sentiment timeline
        sentiment_timeline = sentiment_data.get('speaker_stats', {})
        
        # Simple correlation (could be enhanced)
        topic_sentiment = {}
        
        for keyword in top_keywords[:10]:  # Top 10 keywords
            word = keyword['word']
            # Simplified sentiment correlation
            topic_sentiment[word] = {
                'frequency': keyword['total_frequency'],
                'estimated_sentiment': 0.0,  # Placeholder
                'speakers': keyword['speakers']
            }
        
        return topic_sentiment
        
    except Exception as e:
        logger.error(f"❌ Error correlating topics with sentiment: {e}")
        return {}

def analyze_conversation_progression(flow_data: Dict, sentiment_data: Dict) -> Dict:
    """Analyze conversation progression."""
    try:
        # Get state distribution
        state_distribution = flow_data.get('state_distribution', {})
        
        # Get sentiment trend
        sentiment_trend = sentiment_data.get('recent_trend', {})
        
        # Analyze progression
        progression = {
            'current_phase': max(state_distribution, key=state_distribution.get) if state_distribution else 'unknown',
            'phase_distribution': state_distribution,
            'sentiment_trend': sentiment_trend.get('trend', 'stable'),
            'progression_health': 'good' if sentiment_trend.get('change', 0) >= 0 else 'declining'
        }
        
        return progression
        
    except Exception as e:
        logger.error(f"❌ Error analyzing conversation progression: {e}")
        return {}

def identify_engagement_indicators(sentiment_data: Dict, keyword_data: Dict, flow_data: Dict) -> Dict:
    """Identify engagement indicators."""
    try:
        indicators = {
            'positive_indicators': [],
            'negative_indicators': [],
            'neutral_indicators': []
        }
        
        # Check sentiment indicators
        avg_sentiment = sentiment_data.get('overall_stats', {}).get('avg_sentiment', 0.0)
        if avg_sentiment > 0.3:
            indicators['positive_indicators'].append('Positive overall sentiment')
        elif avg_sentiment < -0.3:
            indicators['negative_indicators'].append('Negative overall sentiment')
        
        # Check keyword indicators
        trending_keywords = keyword_data.get('trending_keywords', [])
        if trending_keywords:
            indicators['positive_indicators'].append(f'{len(trending_keywords)} trending topics')
        
        # Check flow indicators
        avg_metrics = flow_data.get('average_metrics', {})
        engagement = avg_metrics.get('engagement_level', 0.5)
        if engagement > 0.7:
            indicators['positive_indicators'].append('High engagement level')
        elif engagement < 0.3:
            indicators['negative_indicators'].append('Low engagement level')
        
        return indicators
        
    except Exception as e:
        logger.error(f"❌ Error identifying engagement indicators: {e}")
        return {'positive_indicators': [], 'negative_indicators': [], 'neutral_indicators': []}

def generate_recommendations(sentiment_data: Dict, keyword_data: Dict, flow_data: Dict) -> List[str]:
    """Generate actionable recommendations."""
    try:
        recommendations = []
        
        # Sentiment-based recommendations
        avg_sentiment = sentiment_data.get('overall_stats', {}).get('avg_sentiment', 0.0)
        if avg_sentiment < -0.3:
            recommendations.append("Consider addressing customer concerns more proactively")
        
        # Flow-based recommendations
        avg_metrics = flow_data.get('average_metrics', {})
        balance = avg_metrics.get('turn_taking_balance', 0.5)
        if balance < 0.3:
            recommendations.append("Encourage more balanced conversation participation")
        
        # Keyword-based recommendations
        top_keywords = keyword_data.get('top_keywords', [])
        if len(top_keywords) > 20:
            recommendations.append("Focus conversation on key topics to improve clarity")
        
        # Default recommendation
        if not recommendations:
            recommendations.append("Continue current conversation approach")
        
        return recommendations
        
    except Exception as e:
        logger.error(f"❌ Error generating recommendations: {e}")
        return ["Unable to generate recommendations"]

def store_integrated_analysis(session_id: str, transcript_data: Dict):
    """Store integrated analysis result."""
    try:
        if session_id not in integrated_analysis_data:
            integrated_analysis_data[session_id] = {
                'transcript_entries': [],
                'analysis_timeline': [],
                'session_start': datetime.now().isoformat()
            }
        
        # Store transcript entry
        integrated_analysis_data[session_id]['transcript_entries'].append({
            'text': transcript_data['text'],
            'speaker': transcript_data['speaker'],
            'timestamp': transcript_data['timestamp'],
            'is_final': transcript_data.get('is_final', True)
        })
        
        # Store analysis timeline entry
        integrated_analysis_data[session_id]['analysis_timeline'].append({
            'timestamp': datetime.now().isoformat(),
            'transcript_processed': True,
            'analysis_systems': ['sentiment', 'keywords', 'flow']
        })
        
        # Keep only last 100 entries
        if len(integrated_analysis_data[session_id]['transcript_entries']) > 100:
            integrated_analysis_data[session_id]['transcript_entries'] = \
                integrated_analysis_data[session_id]['transcript_entries'][-100:]
        
        if len(integrated_analysis_data[session_id]['analysis_timeline']) > 100:
            integrated_analysis_data[session_id]['analysis_timeline'] = \
                integrated_analysis_data[session_id]['analysis_timeline'][-100:]
        
    except Exception as e:
        logger.error(f"❌ Error storing integrated analysis: {e}")

def emit_to_dashboard(session_id: str, data: Dict):
    """Emit data to real-time dashboard."""
    try:
        # This would integrate with SocketIO to send real-time updates
        # For now, we'll log the update
        logger.info(f"📡 Dashboard update for {session_id}: {data['type']}")
        
        # Store for dashboard retrieval
        if session_id not in dashboard_updates:
            dashboard_updates[session_id] = []
        
        dashboard_updates[session_id].append({
            'timestamp': datetime.now().isoformat(),
            'type': data['type'],
            'data': data['data']
        })
        
        # Keep only last 50 updates
        if len(dashboard_updates[session_id]) > 50:
            dashboard_updates[session_id] = dashboard_updates[session_id][-50:]
        
    except Exception as e:
        logger.error(f"❌ Error emitting to dashboard: {e}")

def get_dashboard_updates(session_id: str, limit: int = 20) -> List[Dict]:
    """Get recent dashboard updates."""
    try:
        if session_id not in dashboard_updates:
            return []
        
        updates = dashboard_updates[session_id]
        return updates[-limit:] if updates else []
        
    except Exception as e:
        logger.error(f"❌ Error getting dashboard updates: {e}")
        return []

# Global data storage
integrated_analysis_data = {}
dashboard_updates = {}

def create_realtime_dashboard_data(session_id):
    """Create dashboard data for real-time display."""
    try:
        session_status = realtime_integrator.get_session_status(session_id)
        
        if 'error' in session_status:
            return {'error': session_status['error']}
        
        # Extract metrics for dashboard
        metrics = session_status.get('current_metrics', {})
        insights = session_status.get('insights', {})
        
        dashboard_data = {
            'session_id': session_id,
            'status': session_status.get('status', 'unknown'),
            'analyses_count': session_status.get('analyses_count', 0),
            'start_time': session_status.get('start_time', ''),
            
            # Sentiment data
            'sentiment': {
                'current': metrics.get('avg_sentiment', {}),
                'history': metrics.get('sentiment_history', [])[-10:],  # Last 10
                'trend': insights.get('sentiment_trend', 'stable')
            },
            
            # Emotion data
            'emotions': {
                'current': metrics.get('dominant_emotions', {}),
                'history': metrics.get('emotion_history', [])[-10:],  # Last 10
                'stability': insights.get('emotion_stability', 'stable')
            },
            
            # Keywords data
            'keywords': {
                'trending': metrics.get('trending_keywords', [])[:10],
                'total_count': len(metrics.get('all_keywords', [])),
                'unique_count': len(set(metrics.get('all_keywords', [])))
            },
            
            # Topics data
            'topics': {
                'distribution': metrics.get('topic_distribution', {}),
                'diversity': len(metrics.get('topic_distribution', {})),
                'primary': insights.get('key_topics', [])[:3]
            },
            
            # Overall insights
            'insights': {
                'conversation_tone': insights.get('conversation_tone', 'neutral'),
                'engagement_level': insights.get('engagement_level', 'medium'),
                'recommendations': insights.get('recommendations', [])
            },
            
            'last_updated': datetime.now().isoformat()
        }
        
        return dashboard_data
        
    except Exception as e:
        return {'error': str(e)}

# Integration with existing call system
def integrate_with_call_system():
    """Integration points with existing call system."""
    
    integration_points = {
        'call_start': '''
# Add to call webhook handler in app.py
from flask_realtime_integration import setup_realtime_routes

# In call webhook:
@app.route('/api/call/webhook', methods=['POST'])
def call_webhook():
    # ... existing code ...
    
    # Start real-time analysis
    call_data = {
        'caller': request.form.get('From'),
        'called': request.form.get('To'),
        'call_sid': request.form.get('CallSid'),
        'direction': request.form.get('Direction', 'inbound')
    }
    
    # Start analysis session
    realtime_integrator.start_session(call_data['call_sid'], call_data)
    
    # ... rest of existing code ...
''',
        
        'call_end': '''
# Add to call status handler
@app.route('/api/call/status', methods=['POST'])
def call_status():
    # ... existing code ...
    
    call_status = request.form.get('CallStatus')
    call_sid = request.form.get('CallSid')
    
    if call_status in ['completed', 'failed', 'busy', 'no-answer']:
        # End real-time analysis
        final_session = realtime_integrator.end_session(call_sid)
        
        # Store final report
        if final_session and 'final_report' in final_session:
            # Save to database or file
            pass
    
    # ... rest of existing code ...
''',
        
        'transcript_processing': '''
# Add to AWS Transcribe WebSocket handler
async def handle_transcript(websocket, path):
    # ... existing code ...
    
    # When transcript is received:
    transcript_data = {
        'text': transcript_text,
        'speaker': speaker_label,
        'timestamp': timestamp,
        'is_final': is_final,
        'confidence': confidence
    }
    
    # Process with real-time analysis
    await realtime_integrator.process_live_transcript(session_id, transcript_data)
    
    # ... rest of existing code ...
'''
    }
    
    return integration_points

if __name__ == "__main__":
    # Test the integration
    print("🚀 Testing Flask Real-time Integration...")
    
    # Create test Flask app
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'test_secret'
    socketio = SocketIO(app, cors_allowed_origins="*")
    
    # Setup routes
    setup_realtime_routes(app, socketio)
    
    print("✅ Flask routes setup completed")
    print("📊 Available endpoints:")
    print("  - POST /api/realtime/start")
    print("  - POST /api/realtime/process")
    print("  - POST /api/realtime/end")
    print("  - GET /api/realtime/status/<session_id>")
    print("  - GET /api/realtime/sessions")
    print("  - GET /test/realtime-analysis")
    
    print("🔌 WebSocket events:")
    print("  - join_analysis_session")
    print("  - leave_analysis_session")
    print("  - request_analysis_update")
    
    print("🎉 Flask Real-time Integration ready!")
    
    # Test dashboard data creation
    test_session_id = "test_session_123"
    dashboard_data = create_realtime_dashboard_data(test_session_id)
    print(f"📈 Dashboard data structure: {list(dashboard_data.keys())}") 