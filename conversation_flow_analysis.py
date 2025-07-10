#!/usr/bin/env python3
"""
Conversation Flow Analysis
Analyzes conversation flow patterns, turn-taking, topic transitions, and conversation dynamics.
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
from collections import defaultdict, deque
import statistics
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConversationState(Enum):
    """Conversation state enumeration."""
    OPENING = "opening"
    INFORMATION_GATHERING = "information_gathering"
    PROBLEM_SOLVING = "problem_solving"
    RESOLUTION = "resolution"
    CLOSING = "closing"
    ESCALATION = "escalation"

class TurnType(Enum):
    """Turn type enumeration."""
    QUESTION = "question"
    ANSWER = "answer"
    STATEMENT = "statement"
    ACKNOWLEDGMENT = "acknowledgment"
    CLARIFICATION = "clarification"
    INSTRUCTION = "instruction"

class ConversationFlowAnalysis:
    """Analyzes conversation flow and dynamics in real-time."""
    
    def __init__(self):
        self.flow_queue = queue.Queue()
        self.processing_thread = None
        self.running = False
        self.session_data = {}
        self.callbacks = []
        
        # Conversation state patterns
        self.state_patterns = {
            ConversationState.OPENING: {
                'keywords': ['hello', 'hi', 'good', 'morning', 'afternoon', 'evening', 'help', 'support'],
                'phrases': ['how can i help', 'thank you for calling', 'welcome to', 'good morning'],
                'turn_patterns': ['agent_first', 'greeting_exchange']
            },
            ConversationState.INFORMATION_GATHERING: {
                'keywords': ['tell', 'explain', 'describe', 'what', 'when', 'where', 'how', 'why'],
                'phrases': ['can you tell me', 'what seems to be', 'let me understand', 'could you explain'],
                'turn_patterns': ['question_answer', 'clarification_cycle']
            },
            ConversationState.PROBLEM_SOLVING: {
                'keywords': ['solution', 'fix', 'resolve', 'try', 'attempt', 'check', 'verify'],
                'phrases': ['let me try', 'have you tried', 'the solution is', 'we can fix this'],
                'turn_patterns': ['instruction_confirmation', 'step_by_step']
            },
            ConversationState.RESOLUTION: {
                'keywords': ['solved', 'fixed', 'resolved', 'working', 'success', 'complete'],
                'phrases': ['problem is solved', 'issue is resolved', 'working now', 'all set'],
                'turn_patterns': ['confirmation', 'satisfaction_check']
            },
            ConversationState.CLOSING: {
                'keywords': ['thank', 'goodbye', 'bye', 'end', 'finish', 'complete'],
                'phrases': ['thank you for', 'have a great day', 'anything else', 'is there anything'],
                'turn_patterns': ['closing_sequence', 'final_thanks']
            },
            ConversationState.ESCALATION: {
                'keywords': ['escalate', 'manager', 'supervisor', 'transfer', 'specialist'],
                'phrases': ['speak to manager', 'transfer to', 'escalate this', 'higher level'],
                'turn_patterns': ['escalation_request', 'transfer_preparation']
            }
        }
        
        # Turn type patterns
        self.turn_type_patterns = {
            TurnType.QUESTION: {
                'indicators': ['?', 'what', 'how', 'when', 'where', 'why', 'can', 'could', 'would'],
                'patterns': [r'\?', r'\bwhat\b', r'\bhow\b', r'\bcan you\b', r'\bcould you\b']
            },
            TurnType.ANSWER: {
                'indicators': ['yes', 'no', 'because', 'the answer is', 'it is', 'that is'],
                'patterns': [r'\byes\b', r'\bno\b', r'\bbecause\b', r'\bit is\b']
            },
            TurnType.STATEMENT: {
                'indicators': ['i think', 'i believe', 'in my opinion', 'it seems'],
                'patterns': [r'\bi think\b', r'\bi believe\b', r'\bit seems\b']
            },
            TurnType.ACKNOWLEDGMENT: {
                'indicators': ['okay', 'ok', 'i see', 'understood', 'got it', 'right'],
                'patterns': [r'\bokay\b', r'\bok\b', r'\bi see\b', r'\bunderstood\b']
            },
            TurnType.CLARIFICATION: {
                'indicators': ['you mean', 'do you mean', 'are you saying', 'so you are'],
                'patterns': [r'\byou mean\b', r'\bdo you mean\b', r'\bare you saying\b']
            },
            TurnType.INSTRUCTION: {
                'indicators': ['please', 'try', 'click', 'go to', 'open', 'close', 'follow'],
                'patterns': [r'\bplease\b', r'\btry\b', r'\bclick\b', r'\bgo to\b']
            }
        }
        
        # Flow metrics
        self.flow_metrics = {
            'turn_taking_balance': 0.0,
            'response_time_avg': 0.0,
            'interruption_count': 0,
            'topic_coherence': 0.0,
            'conversation_momentum': 0.0
        }
        
    def start_analysis(self):
        """Start conversation flow analysis."""
        if self.running:
            logger.warning("Conversation flow analysis already running")
            return
            
        self.running = True
        self.processing_thread = threading.Thread(target=self._process_flow_queue)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info("✅ Conversation flow analysis started")
    
    def stop_analysis(self):
        """Stop conversation flow analysis."""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        
        logger.info("⏹️ Conversation flow analysis stopped")
    
    def register_callback(self, callback):
        """Register callback for flow analysis updates."""
        self.callbacks.append(callback)
        logger.info("✅ Registered conversation flow callback")
    
    def analyze_conversation_turn(self, session_id: str, transcript_data: Dict):
        """Queue conversation turn for flow analysis."""
        try:
            flow_item = {
                'session_id': session_id,
                'transcript': transcript_data,
                'timestamp': datetime.now().isoformat()
            }
            
            self.flow_queue.put(flow_item)
            
        except Exception as e:
            logger.error(f"❌ Error queuing turn for flow analysis: {e}")
    
    def _process_flow_queue(self):
        """Process conversation flow analysis queue."""
        while self.running:
            try:
                # Get item from queue with timeout
                flow_item = self.flow_queue.get(timeout=1)
                
                # Perform flow analysis
                result = self._perform_flow_analysis(flow_item)
                
                # Store result
                self._store_flow_result(flow_item['session_id'], result)
                
                # Update session state
                self._update_session_state(flow_item['session_id'], result)
                
                # Trigger callbacks
                for callback in self.callbacks:
                    try:
                        callback(flow_item['session_id'], result)
                    except Exception as e:
                        logger.error(f"Error in flow callback: {e}")
                
                # Mark task as done
                self.flow_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ Error processing flow queue: {e}")
    
    def _perform_flow_analysis(self, flow_item: Dict) -> Dict:
        """Perform conversation flow analysis."""
        try:
            session_id = flow_item['session_id']
            transcript = flow_item['transcript']
            text = transcript['text']
            speaker = transcript.get('speaker', 'Unknown')
            
            # Analyze turn type
            turn_type = self._classify_turn_type(text)
            
            # Analyze conversation state
            conversation_state = self._analyze_conversation_state(session_id, text, speaker)
            
            # Analyze turn-taking patterns
            turn_taking = self._analyze_turn_taking(session_id, speaker)
            
            # Analyze topic transitions
            topic_transition = self._analyze_topic_transition(session_id, text)
            
            # Analyze conversation momentum
            momentum = self._analyze_conversation_momentum(session_id, text, speaker)
            
            # Analyze response patterns
            response_patterns = self._analyze_response_patterns(session_id, text, speaker)
            
            # Calculate flow metrics
            flow_metrics = self._calculate_flow_metrics(session_id, turn_taking, momentum)
            
            # Detect conversation events
            events = self._detect_conversation_events(session_id, text, speaker, turn_type)
            
            result = {
                'session_id': session_id,
                'speaker': speaker,
                'text': text,
                'timestamp': flow_item['timestamp'],
                'turn_analysis': {
                    'turn_type': turn_type,
                    'turn_length': len(text.split()),
                    'turn_duration': self._estimate_turn_duration(text),
                    'turn_complexity': self._calculate_turn_complexity(text)
                },
                'conversation_state': conversation_state,
                'turn_taking': turn_taking,
                'topic_transition': topic_transition,
                'momentum': momentum,
                'response_patterns': response_patterns,
                'flow_metrics': flow_metrics,
                'events': events,
                'analysis_metadata': {
                    'processing_time': time.time(),
                    'text_length': len(text),
                    'word_count': len(text.split()),
                    'sentence_count': len(re.split(r'[.!?]+', text))
                }
            }
            
            logger.info(f"🔄 Flow analysis complete for {session_id}: {turn_type.value}, state: {conversation_state['current_state'].value}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in flow analysis: {e}")
            return {
                'session_id': flow_item['session_id'],
                'error': str(e),
                'timestamp': flow_item['timestamp']
            }
    
    def _classify_turn_type(self, text: str) -> TurnType:
        """Classify the type of conversation turn."""
        try:
            text_lower = text.lower()
            
            # Score each turn type
            type_scores = {}
            
            for turn_type, patterns in self.turn_type_patterns.items():
                score = 0
                
                # Check indicators
                for indicator in patterns['indicators']:
                    if indicator in text_lower:
                        score += 1
                
                # Check regex patterns
                for pattern in patterns['patterns']:
                    if re.search(pattern, text_lower):
                        score += 2
                
                type_scores[turn_type] = score
            
            # Return type with highest score
            best_type = max(type_scores, key=type_scores.get)
            
            # Default to STATEMENT if no clear indicators
            if type_scores[best_type] == 0:
                return TurnType.STATEMENT
            
            return best_type
            
        except Exception as e:
            logger.error(f"❌ Error classifying turn type: {e}")
            return TurnType.STATEMENT
    
    def _analyze_conversation_state(self, session_id: str, text: str, speaker: str) -> Dict:
        """Analyze current conversation state."""
        try:
            text_lower = text.lower()
            
            # Score each state
            state_scores = {}
            
            for state, patterns in self.state_patterns.items():
                score = 0
                
                # Check keywords
                for keyword in patterns['keywords']:
                    if keyword in text_lower:
                        score += 1
                
                # Check phrases
                for phrase in patterns['phrases']:
                    if phrase in text_lower:
                        score += 2
                
                state_scores[state] = score
            
            # Get current state
            current_state = max(state_scores, key=state_scores.get)
            confidence = state_scores[current_state] / max(1, sum(state_scores.values()))
            
            # Get state history
            state_history = self._get_state_history(session_id)
            
            # Detect state transition
            previous_state = state_history[-1] if state_history else None
            is_transition = previous_state and previous_state != current_state
            
            return {
                'current_state': current_state,
                'confidence': confidence,
                'state_scores': {state.value: score for state, score in state_scores.items()},
                'is_transition': is_transition,
                'previous_state': previous_state.value if previous_state else None,
                'state_duration': self._calculate_state_duration(session_id, current_state)
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing conversation state: {e}")
            return {
                'current_state': ConversationState.INFORMATION_GATHERING,
                'confidence': 0.5,
                'is_transition': False
            }
    
    def _analyze_turn_taking(self, session_id: str, current_speaker: str) -> Dict:
        """Analyze turn-taking patterns."""
        try:
            # Get recent turns
            recent_turns = self._get_recent_turns(session_id, limit=10)
            
            if not recent_turns:
                return {
                    'speaker_balance': 0.5,
                    'avg_turn_length': 0,
                    'turn_frequency': 0,
                    'speaker_dominance': 'balanced'
                }
            
            # Calculate speaker statistics
            speaker_stats = defaultdict(lambda: {'count': 0, 'total_words': 0})
            
            for turn in recent_turns:
                speaker = turn['speaker']
                word_count = len(turn['text'].split())
                
                speaker_stats[speaker]['count'] += 1
                speaker_stats[speaker]['total_words'] += word_count
            
            # Calculate balance
            total_turns = len(recent_turns)
            speaker_ratios = {
                speaker: stats['count'] / total_turns
                for speaker, stats in speaker_stats.items()
            }
            
            # Determine dominance
            if len(speaker_ratios) >= 2:
                speakers = list(speaker_ratios.keys())
                ratio_diff = abs(speaker_ratios[speakers[0]] - speaker_ratios[speakers[1]])
                
                if ratio_diff > 0.3:
                    dominant_speaker = max(speaker_ratios, key=speaker_ratios.get)
                    dominance = f"{dominant_speaker}_dominant"
                else:
                    dominance = "balanced"
            else:
                dominance = "single_speaker"
            
            # Calculate average turn length
            avg_turn_length = statistics.mean([
                len(turn['text'].split()) for turn in recent_turns
            ])
            
            # Calculate turn frequency (turns per minute)
            if len(recent_turns) >= 2:
                time_span = self._calculate_time_span(recent_turns)
                turn_frequency = len(recent_turns) / max(1, time_span)
            else:
                turn_frequency = 0
            
            return {
                'speaker_balance': min(speaker_ratios.values()) if speaker_ratios else 0.5,
                'avg_turn_length': avg_turn_length,
                'turn_frequency': turn_frequency,
                'speaker_dominance': dominance,
                'speaker_ratios': speaker_ratios,
                'total_turns': total_turns
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing turn-taking: {e}")
            return {'speaker_balance': 0.5, 'avg_turn_length': 0, 'turn_frequency': 0}
    
    def _analyze_topic_transition(self, session_id: str, text: str) -> Dict:
        """Analyze topic transitions in conversation."""
        try:
            # Get recent topics (simplified - could integrate with keyword extraction)
            current_topics = self._extract_simple_topics(text)
            recent_topics = self._get_recent_topics(session_id, limit=5)
            
            # Detect topic changes
            topic_changes = []
            for current_topic in current_topics:
                if current_topic not in recent_topics:
                    topic_changes.append({
                        'new_topic': current_topic,
                        'transition_type': 'new_topic'
                    })
            
            # Calculate topic coherence
            coherence_score = self._calculate_topic_coherence(current_topics, recent_topics)
            
            # Detect topic drift
            topic_drift = len(topic_changes) > 2
            
            return {
                'current_topics': current_topics,
                'topic_changes': topic_changes,
                'coherence_score': coherence_score,
                'topic_drift': topic_drift,
                'transition_smoothness': 1.0 - (len(topic_changes) / max(1, len(current_topics)))
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing topic transition: {e}")
            return {'current_topics': [], 'topic_changes': [], 'coherence_score': 0.5}
    
    def _analyze_conversation_momentum(self, session_id: str, text: str, speaker: str) -> Dict:
        """Analyze conversation momentum and energy."""
        try:
            # Calculate text energy indicators
            word_count = len(text.split())
            exclamation_count = text.count('!')
            question_count = text.count('?')
            caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
            
            # Base energy score
            energy_score = (
                min(word_count / 20, 1.0) * 0.4 +
                min(exclamation_count / 3, 1.0) * 0.3 +
                min(question_count / 2, 1.0) * 0.2 +
                min(caps_ratio * 5, 1.0) * 0.1
            )
            
            # Get momentum history
            momentum_history = self._get_momentum_history(session_id, limit=5)
            
            # Calculate momentum trend
            if len(momentum_history) >= 2:
                recent_momentum = statistics.mean(momentum_history[-3:])
                earlier_momentum = statistics.mean(momentum_history[:-3]) if len(momentum_history) > 3 else momentum_history[0]
                
                momentum_change = recent_momentum - earlier_momentum
                
                if momentum_change > 0.1:
                    momentum_trend = 'increasing'
                elif momentum_change < -0.1:
                    momentum_trend = 'decreasing'
                else:
                    momentum_trend = 'stable'
            else:
                momentum_trend = 'stable'
                momentum_change = 0.0
            
            # Calculate conversation pace
            recent_turns = self._get_recent_turns(session_id, limit=5)
            if len(recent_turns) >= 2:
                time_span = self._calculate_time_span(recent_turns)
                pace = len(recent_turns) / max(1, time_span)
            else:
                pace = 0.0
            
            return {
                'energy_score': energy_score,
                'momentum_trend': momentum_trend,
                'momentum_change': momentum_change,
                'conversation_pace': pace,
                'engagement_level': self._calculate_engagement_level(energy_score, pace)
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing conversation momentum: {e}")
            return {'energy_score': 0.5, 'momentum_trend': 'stable', 'momentum_change': 0.0}
    
    def _analyze_response_patterns(self, session_id: str, text: str, speaker: str) -> Dict:
        """Analyze response patterns and timing."""
        try:
            recent_turns = self._get_recent_turns(session_id, limit=5)
            
            if len(recent_turns) < 2:
                return {
                    'response_type': 'initial',
                    'response_length_ratio': 1.0,
                    'response_style': 'neutral'
                }
            
            # Get previous turn
            previous_turn = recent_turns[-2]
            current_turn = {'text': text, 'speaker': speaker}
            
            # Analyze response type
            response_type = self._classify_response_type(previous_turn, current_turn)
            
            # Calculate response length ratio
            prev_length = len(previous_turn['text'].split())
            curr_length = len(text.split())
            length_ratio = curr_length / max(1, prev_length)
            
            # Analyze response style
            response_style = self._analyze_response_style(text, previous_turn['text'])
            
            # Calculate response time (estimated)
            response_time = self._estimate_response_time(previous_turn, current_turn)
            
            return {
                'response_type': response_type,
                'response_length_ratio': length_ratio,
                'response_style': response_style,
                'estimated_response_time': response_time,
                'is_interruption': self._detect_interruption(previous_turn, current_turn)
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing response patterns: {e}")
            return {'response_type': 'neutral', 'response_length_ratio': 1.0}
    
    def _calculate_flow_metrics(self, session_id: str, turn_taking: Dict, momentum: Dict) -> Dict:
        """Calculate overall flow metrics."""
        try:
            # Get session history
            session_history = self._get_session_history(session_id)
            
            if not session_history:
                return self.flow_metrics.copy()
            
            # Calculate metrics
            metrics = {
                'turn_taking_balance': turn_taking.get('speaker_balance', 0.5),
                'conversation_pace': momentum.get('conversation_pace', 0.0),
                'topic_coherence': self._calculate_session_topic_coherence(session_id),
                'engagement_level': momentum.get('engagement_level', 0.5),
                'conversation_efficiency': self._calculate_conversation_efficiency(session_id),
                'flow_smoothness': self._calculate_flow_smoothness(session_id)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Error calculating flow metrics: {e}")
            return self.flow_metrics.copy()
    
    def _detect_conversation_events(self, session_id: str, text: str, speaker: str, turn_type: TurnType) -> List[Dict]:
        """Detect significant conversation events."""
        try:
            events = []
            text_lower = text.lower()
            
            # Detect specific events
            event_patterns = {
                'escalation_request': ['manager', 'supervisor', 'escalate', 'higher level'],
                'problem_resolution': ['solved', 'fixed', 'resolved', 'working now'],
                'customer_frustration': ['frustrated', 'angry', 'upset', 'disappointed'],
                'positive_feedback': ['great', 'excellent', 'amazing', 'thank you'],
                'topic_change': ['by the way', 'also', 'another thing', 'speaking of'],
                'clarification_needed': ['confused', 'unclear', 'don\'t understand', 'what do you mean'],
                'agreement': ['yes', 'correct', 'exactly', 'that\'s right'],
                'disagreement': ['no', 'wrong', 'incorrect', 'disagree']
            }
            
            for event_type, keywords in event_patterns.items():
                for keyword in keywords:
                    if keyword in text_lower:
                        events.append({
                            'event_type': event_type,
                            'trigger_keyword': keyword,
                            'speaker': speaker,
                            'confidence': 0.8,
                            'timestamp': datetime.now().isoformat()
                        })
                        break  # Only one event per type per turn
            
            # Detect turn-based events
            if turn_type == TurnType.QUESTION:
                events.append({
                    'event_type': 'question_asked',
                    'speaker': speaker,
                    'confidence': 0.9,
                    'timestamp': datetime.now().isoformat()
                })
            
            return events
            
        except Exception as e:
            logger.error(f"❌ Error detecting conversation events: {e}")
            return []
    
    # Helper methods
    def _get_recent_turns(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Get recent conversation turns."""
        if session_id not in self.session_data:
            return []
        
        history = self.session_data[session_id].get('turn_history', [])
        return history[-limit:] if history else []
    
    def _get_state_history(self, session_id: str) -> List[ConversationState]:
        """Get conversation state history."""
        if session_id not in self.session_data:
            return []
        
        return self.session_data[session_id].get('state_history', [])
    
    def _get_recent_topics(self, session_id: str, limit: int = 5) -> List[str]:
        """Get recent conversation topics."""
        if session_id not in self.session_data:
            return []
        
        history = self.session_data[session_id].get('topic_history', [])
        return history[-limit:] if history else []
    
    def _get_momentum_history(self, session_id: str, limit: int = 5) -> List[float]:
        """Get conversation momentum history."""
        if session_id not in self.session_data:
            return []
        
        history = self.session_data[session_id].get('momentum_history', [])
        return history[-limit:] if history else []
    
    def _get_session_history(self, session_id: str) -> List[Dict]:
        """Get full session history."""
        if session_id not in self.session_data:
            return []
        
        return self.session_data[session_id].get('flow_history', [])
    
    def _extract_simple_topics(self, text: str) -> List[str]:
        """Extract simple topics from text (basic implementation)."""
        # Simple topic extraction based on keywords
        topics = []
        
        topic_keywords = {
            'technical': ['software', 'system', 'computer', 'application', 'network'],
            'billing': ['payment', 'bill', 'charge', 'invoice', 'subscription'],
            'account': ['account', 'profile', 'settings', 'login', 'password'],
            'support': ['help', 'support', 'assistance', 'problem', 'issue']
        }
        
        text_lower = text.lower()
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    def _calculate_topic_coherence(self, current_topics: List[str], recent_topics: List[str]) -> float:
        """Calculate topic coherence score."""
        if not current_topics and not recent_topics:
            return 1.0
        
        if not current_topics or not recent_topics:
            return 0.0
        
        # Calculate overlap
        overlap = len(set(current_topics) & set(recent_topics))
        total = len(set(current_topics) | set(recent_topics))
        
        return overlap / total if total > 0 else 0.0
    
    def _calculate_engagement_level(self, energy_score: float, pace: float) -> float:
        """Calculate engagement level."""
        # Combine energy and pace for engagement
        engagement = (energy_score * 0.7 + min(pace / 2, 1.0) * 0.3)
        return min(1.0, engagement)
    
    def _calculate_time_span(self, turns: List[Dict]) -> float:
        """Calculate time span between turns in minutes."""
        if len(turns) < 2:
            return 1.0
        
        # Simple estimation based on turn count (real implementation would use timestamps)
        return len(turns) * 0.5  # Assume 30 seconds per turn
    
    def _estimate_turn_duration(self, text: str) -> float:
        """Estimate turn duration in seconds."""
        # Simple estimation: ~150 words per minute
        word_count = len(text.split())
        return (word_count / 150) * 60
    
    def _calculate_turn_complexity(self, text: str) -> float:
        """Calculate turn complexity score."""
        word_count = len(text.split())
        sentence_count = len(re.split(r'[.!?]+', text))
        avg_word_length = sum(len(word) for word in text.split()) / max(1, word_count)
        
        # Normalize complexity
        complexity = (
            min(word_count / 50, 1.0) * 0.4 +
            min(sentence_count / 5, 1.0) * 0.3 +
            min(avg_word_length / 8, 1.0) * 0.3
        )
        
        return complexity
    
    def _classify_response_type(self, previous_turn: Dict, current_turn: Dict) -> str:
        """Classify response type."""
        prev_text = previous_turn['text'].lower()
        curr_text = current_turn['text'].lower()
        
        if '?' in prev_text and any(word in curr_text for word in ['yes', 'no', 'because']):
            return 'direct_answer'
        elif any(word in curr_text for word in ['what', 'how', 'why', 'when']):
            return 'counter_question'
        elif any(word in curr_text for word in ['okay', 'i see', 'understood']):
            return 'acknowledgment'
        else:
            return 'elaboration'
    
    def _analyze_response_style(self, current_text: str, previous_text: str) -> str:
        """Analyze response style."""
        curr_lower = current_text.lower()
        
        if any(word in curr_lower for word in ['please', 'thank', 'sorry']):
            return 'polite'
        elif any(word in curr_lower for word in ['quickly', 'immediately', 'urgent']):
            return 'urgent'
        elif len(current_text.split()) > 30:
            return 'detailed'
        elif len(current_text.split()) < 5:
            return 'brief'
        else:
            return 'neutral'
    
    def _estimate_response_time(self, previous_turn: Dict, current_turn: Dict) -> float:
        """Estimate response time (simplified)."""
        # Simple estimation based on turn length
        prev_length = len(previous_turn['text'].split())
        return max(1.0, prev_length * 0.2)  # Assume 0.2 seconds per word to process
    
    def _detect_interruption(self, previous_turn: Dict, current_turn: Dict) -> bool:
        """Detect if current turn is an interruption."""
        # Simple heuristic: if previous turn was long and current is from different speaker
        prev_length = len(previous_turn['text'].split())
        return (prev_length > 20 and 
                previous_turn['speaker'] != current_turn['speaker'] and
                current_turn['text'].startswith(('but', 'wait', 'hold on', 'excuse me')))
    
    def _calculate_session_topic_coherence(self, session_id: str) -> float:
        """Calculate overall session topic coherence."""
        # Simplified implementation
        return 0.7  # Placeholder
    
    def _calculate_conversation_efficiency(self, session_id: str) -> float:
        """Calculate conversation efficiency."""
        # Simplified implementation
        return 0.8  # Placeholder
    
    def _calculate_flow_smoothness(self, session_id: str) -> float:
        """Calculate conversation flow smoothness."""
        # Simplified implementation
        return 0.75  # Placeholder
    
    def _calculate_state_duration(self, session_id: str, state: ConversationState) -> float:
        """Calculate how long conversation has been in current state."""
        # Simplified implementation
        return 2.5  # Placeholder minutes
    
    def _store_flow_result(self, session_id: str, result: Dict):
        """Store flow analysis result."""
        try:
            if session_id not in self.session_data:
                self.session_data[session_id] = {
                    'flow_history': [],
                    'turn_history': [],
                    'state_history': [],
                    'topic_history': [],
                    'momentum_history': [],
                    'session_start': datetime.now().isoformat()
                }
            
            # Add to flow history
            self.session_data[session_id]['flow_history'].append(result)
            
            # Add to turn history
            self.session_data[session_id]['turn_history'].append({
                'text': result['text'],
                'speaker': result['speaker'],
                'timestamp': result['timestamp']
            })
            
            # Update state history
            current_state = result['conversation_state']['current_state']
            if (not self.session_data[session_id]['state_history'] or 
                self.session_data[session_id]['state_history'][-1] != current_state):
                self.session_data[session_id]['state_history'].append(current_state)
            
            # Update topic history
            current_topics = result['topic_transition']['current_topics']
            self.session_data[session_id]['topic_history'].extend(current_topics)
            
            # Update momentum history
            momentum_score = result['momentum']['energy_score']
            self.session_data[session_id]['momentum_history'].append(momentum_score)
            
            # Keep only last 100 entries
            for key in ['flow_history', 'turn_history', 'topic_history', 'momentum_history']:
                if len(self.session_data[session_id][key]) > 100:
                    self.session_data[session_id][key] = self.session_data[session_id][key][-100:]
            
            # Keep only last 20 states
            if len(self.session_data[session_id]['state_history']) > 20:
                self.session_data[session_id]['state_history'] = self.session_data[session_id]['state_history'][-20:]
            
        except Exception as e:
            logger.error(f"❌ Error storing flow result: {e}")
    
    def _update_session_state(self, session_id: str, result: Dict):
        """Update overall session state."""
        try:
            if session_id not in self.session_data:
                return
            
            # Update current conversation state
            self.session_data[session_id]['current_state'] = result['conversation_state']['current_state']
            self.session_data[session_id]['last_updated'] = datetime.now().isoformat()
            
            # Update flow metrics
            self.session_data[session_id]['current_metrics'] = result['flow_metrics']
            
        except Exception as e:
            logger.error(f"❌ Error updating session state: {e}")
    
    def get_session_flow_summary(self, session_id: str) -> Dict:
        """Get conversation flow summary for session."""
        try:
            if session_id not in self.session_data:
                return {'error': 'Session not found'}
            
            session_data = self.session_data[session_id]
            flow_history = session_data.get('flow_history', [])
            
            if not flow_history:
                return {'error': 'No flow data available'}
            
            # Calculate summary statistics
            total_turns = len(flow_history)
            
            # State distribution
            state_counts = defaultdict(int)
            for entry in flow_history:
                state = entry['conversation_state']['current_state'].value
                state_counts[state] += 1
            
            # Turn type distribution
            turn_type_counts = defaultdict(int)
            for entry in flow_history:
                turn_type = entry['turn_analysis']['turn_type'].value
                turn_type_counts[turn_type] += 1
            
            # Average metrics
            avg_metrics = {}
            if flow_history:
                metrics_keys = flow_history[0].get('flow_metrics', {}).keys()
                for key in metrics_keys:
                    values = [entry['flow_metrics'][key] for entry in flow_history if key in entry['flow_metrics']]
                    avg_metrics[key] = statistics.mean(values) if values else 0.0
            
            # Recent events
            recent_events = []
            for entry in flow_history[-10:]:
                recent_events.extend(entry.get('events', []))
            
            return {
                'session_id': session_id,
                'total_turns': total_turns,
                'state_distribution': dict(state_counts),
                'turn_type_distribution': dict(turn_type_counts),
                'average_metrics': avg_metrics,
                'recent_events': recent_events,
                'current_state': session_data.get('current_state', {}).value if session_data.get('current_state') else 'unknown',
                'session_start': session_data.get('session_start'),
                'last_updated': session_data.get('last_updated')
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting flow summary: {e}")
            return {'error': str(e)}
    
    def get_live_flow_data(self, session_id: str, limit: int = 10) -> Dict:
        """Get live flow data for real-time display."""
        try:
            if session_id not in self.session_data:
                return {'error': 'Session not found'}
            
            session_data = self.session_data[session_id]
            flow_history = session_data.get('flow_history', [])
            
            # Get recent entries
            recent_entries = flow_history[-limit:] if flow_history else []
            
            # Current state
            current_state = session_data.get('current_state', {}).value if session_data.get('current_state') else 'unknown'
            
            # Current metrics
            current_metrics = session_data.get('current_metrics', {})
            
            # Recent turn types
            recent_turn_types = [
                entry['turn_analysis']['turn_type'].value 
                for entry in recent_entries
            ]
            
            # Recent events
            recent_events = []
            for entry in recent_entries:
                recent_events.extend(entry.get('events', []))
            
            return {
                'session_id': session_id,
                'current_state': current_state,
                'current_metrics': current_metrics,
                'recent_turn_types': recent_turn_types,
                'recent_events': recent_events[-5:],  # Last 5 events
                'total_turns': len(flow_history),
                'last_updated': session_data.get('last_updated')
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting live flow data: {e}")
            return {'error': str(e)}

# Global flow analyzer instance
flow_analyzer = ConversationFlowAnalysis()

# Integration functions
def start_flow_analysis():
    """Start flow analysis."""
    flow_analyzer.start_analysis()

def stop_flow_analysis():
    """Stop flow analysis."""
    flow_analyzer.stop_analysis()

def analyze_conversation_flow(session_id: str, transcript_data: Dict):
    """Analyze conversation flow."""
    flow_analyzer.analyze_conversation_turn(session_id, transcript_data)

def get_flow_summary(session_id: str) -> Dict:
    """Get flow summary."""
    return flow_analyzer.get_session_flow_summary(session_id)

def get_live_flow(session_id: str, limit: int = 10) -> Dict:
    """Get live flow data."""
    return flow_analyzer.get_live_flow_data(session_id, limit)

def register_flow_callback(callback):
    """Register flow callback."""
    flow_analyzer.register_callback(callback)

if __name__ == "__main__":
    # Test the flow analyzer
    print("🚀 Testing Conversation Flow Analysis...")
    
    # Start analyzer
    flow_analyzer.start_analysis()
    
    # Test conversation sequence
    test_conversation = [
        {"text": "Hello, how can I help you today?", "speaker": "Agent"},
        {"text": "Hi, I'm having trouble with my account login.", "speaker": "Customer"},
        {"text": "I understand. Can you tell me more about the issue?", "speaker": "Agent"},
        {"text": "Yes, I keep getting an error message when I try to log in.", "speaker": "Customer"},
        {"text": "Let me help you troubleshoot this. Have you tried resetting your password?", "speaker": "Agent"},
        {"text": "No, I haven't. How do I do that?", "speaker": "Customer"},
        {"text": "I'll guide you through the process step by step.", "speaker": "Agent"},
        {"text": "Great! That worked perfectly. Thank you so much!", "speaker": "Customer"},
        {"text": "You're welcome! Is there anything else I can help you with?", "speaker": "Agent"},
        {"text": "No, that's all. Have a great day!", "speaker": "Customer"}
    ]
    
    # Analyze conversation flow
    for i, turn in enumerate(test_conversation):
        transcript_data = {
            'text': turn['text'],
            'speaker': turn['speaker'],
            'timestamp': datetime.now().isoformat(),
            'is_final': True
        }
        
        flow_analyzer.analyze_conversation_turn(f"test_session", transcript_data)
        time.sleep(0.3)  # Small delay to simulate real conversation
    
    # Get summary
    time.sleep(2)  # Wait for processing
    summary = flow_analyzer.get_session_flow_summary("test_session")
    print(f"🔄 Flow Summary: {summary['total_turns']} turns, current state: {summary['current_state']}")
    
    # Stop analyzer
    flow_analyzer.stop_analysis()
    
    print("🎉 Conversation Flow Analysis test completed!") 