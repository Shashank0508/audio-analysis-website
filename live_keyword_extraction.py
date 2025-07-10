#!/usr/bin/env python3
"""
Live Keyword Extraction
Real-time keyword and topic extraction from streaming transcripts.
"""

import asyncio
import threading
import time
import queue
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
import json
import re
from collections import defaultdict, Counter
import statistics
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LiveKeywordExtraction:
    """Real-time keyword extraction and topic analysis."""
    
    def __init__(self):
        self.keyword_queue = queue.Queue()
        self.processing_thread = None
        self.running = False
        self.session_data = {}
        self.callbacks = []
        
        # Stop words (common words to ignore)
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'were', 'will', 'with', 'the', 'this', 'but', 'they',
            'have', 'had', 'what', 'said', 'each', 'which', 'she', 'do', 'how',
            'their', 'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so',
            'some', 'her', 'would', 'make', 'like', 'into', 'him', 'time',
            'two', 'more', 'very', 'when', 'come', 'may', 'see', 'get', 'use',
            'your', 'way', 'about', 'just', 'first', 'also', 'new', 'because',
            'day', 'more', 'man', 'find', 'here', 'thing', 'give', 'well',
            'us', 'old', 'take', 'little', 'world', 'own', 'other', 'tell',
            'back', 'after', 'work', 'first', 'try', 'move', 'why', 'ask',
            'men', 'change', 'went', 'light', 'kind', 'off', 'need', 'house',
            'picture', 'again', 'place', 'where', 'turn', 'put', 'end', 'does',
            'another', 'home', 'around', 'small', 'however', 'found', 'still',
            'between', 'name', 'should', 'mr', 'through', 'much', 'before',
            'line', 'right', 'too', 'means', 'old', 'any', 'same', 'tell',
            'boy', 'follow', 'came', 'want', 'show', 'also', 'around', 'form',
            'three', 'set', 'small', 'every', 'sound', 'still', 'such', 'make',
            'hand', 'high', 'year', 'came', 'show', 'every', 'good', 'me',
            'give', 'our', 'under', 'name', 'very', 'through', 'just', 'form',
            'much', 'great', 'think', 'say', 'help', 'low', 'line', 'before',
            'turn', 'cause', 'same', 'mean', 'differ', 'move', 'right', 'boy',
            'old', 'too', 'any', 'day', 'get', 'use', 'man', 'new', 'now',
            'way', 'may', 'say'
        }
        
        # Business domain keywords
        self.business_keywords = {
            'account', 'service', 'support', 'customer', 'product', 'order',
            'payment', 'billing', 'invoice', 'refund', 'subscription', 'plan',
            'upgrade', 'downgrade', 'cancel', 'renewal', 'contract', 'agreement',
            'policy', 'terms', 'conditions', 'warranty', 'guarantee', 'quality',
            'delivery', 'shipping', 'tracking', 'return', 'exchange', 'discount',
            'promotion', 'offer', 'deal', 'price', 'cost', 'fee', 'charge',
            'transaction', 'purchase', 'sale', 'revenue', 'profit', 'budget'
        }
        
        # Technical keywords
        self.technical_keywords = {
            'software', 'hardware', 'system', 'application', 'platform', 'database',
            'server', 'network', 'security', 'backup', 'update', 'upgrade',
            'installation', 'configuration', 'settings', 'preferences', 'options',
            'features', 'functionality', 'performance', 'speed', 'memory',
            'storage', 'bandwidth', 'connection', 'interface', 'api', 'integration',
            'authentication', 'authorization', 'encryption', 'protocol', 'framework',
            'library', 'module', 'component', 'plugin', 'extension', 'version'
        }
        
        # Emotion/sentiment keywords
        self.emotion_keywords = {
            'happy', 'sad', 'angry', 'frustrated', 'pleased', 'disappointed',
            'excited', 'worried', 'concerned', 'satisfied', 'annoyed', 'thrilled',
            'upset', 'delighted', 'confused', 'impressed', 'surprised', 'shocked',
            'amazed', 'grateful', 'thankful', 'sorry', 'apologize', 'regret'
        }
        
        # Topic categories
        self.topic_categories = {
            'business': self.business_keywords,
            'technical': self.technical_keywords,
            'emotion': self.emotion_keywords
        }
        
        # Keyword importance weights
        self.importance_weights = {
            'frequency': 0.3,
            'recency': 0.2,
            'speaker_diversity': 0.2,
            'context_relevance': 0.15,
            'domain_relevance': 0.15
        }
        
    def start_extraction(self):
        """Start live keyword extraction."""
        if self.running:
            logger.warning("Keyword extraction already running")
            return
            
        self.running = True
        self.processing_thread = threading.Thread(target=self._process_keyword_queue)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info("✅ Live keyword extraction started")
    
    def stop_extraction(self):
        """Stop live keyword extraction."""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        
        logger.info("⏹️ Live keyword extraction stopped")
    
    def register_callback(self, callback):
        """Register callback for keyword updates."""
        self.callbacks.append(callback)
        logger.info("✅ Registered keyword extraction callback")
    
    def extract_keywords(self, session_id: str, transcript_data: Dict):
        """Queue transcript for keyword extraction."""
        try:
            extraction_item = {
                'session_id': session_id,
                'transcript': transcript_data,
                'timestamp': datetime.now().isoformat()
            }
            
            self.keyword_queue.put(extraction_item)
            
        except Exception as e:
            logger.error(f"❌ Error queuing transcript for keyword extraction: {e}")
    
    def _process_keyword_queue(self):
        """Process keyword extraction queue."""
        while self.running:
            try:
                # Get item from queue with timeout
                extraction_item = self.keyword_queue.get(timeout=1)
                
                # Perform keyword extraction
                result = self._perform_keyword_extraction(extraction_item)
                
                # Store result
                self._store_keyword_result(extraction_item['session_id'], result)
                
                # Trigger callbacks
                for callback in self.callbacks:
                    try:
                        callback(extraction_item['session_id'], result)
                    except Exception as e:
                        logger.error(f"Error in keyword callback: {e}")
                
                # Mark task as done
                self.keyword_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ Error processing keyword queue: {e}")
    
    def _perform_keyword_extraction(self, extraction_item: Dict) -> Dict:
        """Perform keyword extraction on transcript."""
        try:
            session_id = extraction_item['session_id']
            transcript = extraction_item['transcript']
            text = transcript['text']
            speaker = transcript.get('speaker', 'Unknown')
            
            # Extract keywords
            keywords = self._extract_keywords_from_text(text)
            
            # Extract phrases
            phrases = self._extract_phrases_from_text(text)
            
            # Identify topics
            topics = self._identify_topics(text, keywords)
            
            # Calculate keyword importance
            keyword_importance = self._calculate_keyword_importance(
                session_id, keywords, speaker
            )
            
            # Extract entities (basic implementation)
            entities = self._extract_entities(text)
            
            # Analyze keyword context
            keyword_context = self._analyze_keyword_context(text, keywords)
            
            # Detect trending keywords
            trending = self._detect_trending_keywords(session_id, keywords)
            
            result = {
                'session_id': session_id,
                'speaker': speaker,
                'text': text,
                'timestamp': extraction_item['timestamp'],
                'keywords': {
                    'all_keywords': keywords,
                    'important_keywords': keyword_importance,
                    'trending_keywords': trending,
                    'keyword_count': len(keywords)
                },
                'phrases': phrases,
                'topics': topics,
                'entities': entities,
                'context': keyword_context,
                'analysis_metadata': {
                    'processing_time': time.time(),
                    'text_length': len(text),
                    'word_count': len(text.split()),
                    'unique_words': len(set(text.lower().split()))
                }
            }
            
            logger.info(f"🔍 Keyword extraction complete for {session_id}: {len(keywords)} keywords, {len(topics)} topics")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in keyword extraction: {e}")
            return {
                'session_id': extraction_item['session_id'],
                'error': str(e),
                'timestamp': extraction_item['timestamp']
            }
    
    def _extract_keywords_from_text(self, text: str) -> List[Dict]:
        """Extract keywords from text."""
        try:
            # Clean and tokenize text
            cleaned_text = re.sub(r'[^\w\s]', ' ', text.lower())
            words = cleaned_text.split()
            
            # Remove stop words and short words
            filtered_words = [
                word for word in words 
                if word not in self.stop_words and len(word) > 2
            ]
            
            # Count word frequencies
            word_freq = Counter(filtered_words)
            
            # Extract keywords with metadata
            keywords = []
            for word, frequency in word_freq.items():
                keyword_data = {
                    'word': word,
                    'frequency': frequency,
                    'positions': [i for i, w in enumerate(words) if w == word],
                    'context_words': self._get_context_words(words, word),
                    'domain_category': self._categorize_keyword(word),
                    'importance_score': self._calculate_word_importance(word, frequency, len(words))
                }
                keywords.append(keyword_data)
            
            # Sort by importance
            keywords.sort(key=lambda x: x['importance_score'], reverse=True)
            
            return keywords
            
        except Exception as e:
            logger.error(f"❌ Error extracting keywords: {e}")
            return []
    
    def _extract_phrases_from_text(self, text: str) -> List[Dict]:
        """Extract meaningful phrases from text."""
        try:
            # Simple n-gram extraction (2-3 words)
            words = re.findall(r'\b\w+\b', text.lower())
            phrases = []
            
            # Extract 2-grams
            for i in range(len(words) - 1):
                phrase = f"{words[i]} {words[i+1]}"
                if not any(word in self.stop_words for word in [words[i], words[i+1]]):
                    phrases.append({
                        'phrase': phrase,
                        'type': 'bigram',
                        'position': i,
                        'relevance_score': self._calculate_phrase_relevance(phrase)
                    })
            
            # Extract 3-grams
            for i in range(len(words) - 2):
                phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
                if sum(1 for word in [words[i], words[i+1], words[i+2]] if word in self.stop_words) <= 1:
                    phrases.append({
                        'phrase': phrase,
                        'type': 'trigram',
                        'position': i,
                        'relevance_score': self._calculate_phrase_relevance(phrase)
                    })
            
            # Remove duplicates and sort by relevance
            unique_phrases = {}
            for phrase_data in phrases:
                phrase = phrase_data['phrase']
                if phrase not in unique_phrases or phrase_data['relevance_score'] > unique_phrases[phrase]['relevance_score']:
                    unique_phrases[phrase] = phrase_data
            
            result_phrases = list(unique_phrases.values())
            result_phrases.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            return result_phrases[:20]  # Top 20 phrases
            
        except Exception as e:
            logger.error(f"❌ Error extracting phrases: {e}")
            return []
    
    def _identify_topics(self, text: str, keywords: List[Dict]) -> List[Dict]:
        """Identify topics from text and keywords."""
        try:
            topics = []
            
            # Check each topic category
            for category, category_keywords in self.topic_categories.items():
                # Count matches
                matches = 0
                matched_keywords = []
                
                for keyword_data in keywords:
                    word = keyword_data['word']
                    if word in category_keywords:
                        matches += keyword_data['frequency']
                        matched_keywords.append(word)
                
                if matches > 0:
                    # Calculate topic relevance
                    relevance = matches / len(keywords) if keywords else 0
                    
                    topics.append({
                        'category': category,
                        'relevance_score': relevance,
                        'matched_keywords': matched_keywords,
                        'match_count': matches,
                        'confidence': min(relevance * 2, 1.0)  # Normalize confidence
                    })
            
            # Sort by relevance
            topics.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            return topics
            
        except Exception as e:
            logger.error(f"❌ Error identifying topics: {e}")
            return []
    
    def _calculate_keyword_importance(self, session_id: str, keywords: List[Dict], speaker: str) -> List[Dict]:
        """Calculate keyword importance based on various factors."""
        try:
            if not keywords:
                return []
            
            # Get session history
            session_history = self.session_data.get(session_id, {}).get('keyword_history', [])
            
            important_keywords = []
            
            for keyword_data in keywords:
                word = keyword_data['word']
                
                # Frequency score
                frequency_score = keyword_data['frequency'] / max(1, len(keywords))
                
                # Recency score (newer is better)
                recency_score = 1.0  # Current transcript gets max recency
                
                # Speaker diversity score
                speaker_diversity_score = self._calculate_speaker_diversity(session_id, word)
                
                # Context relevance score
                context_relevance_score = self._calculate_context_relevance(word, keyword_data['context_words'])
                
                # Domain relevance score
                domain_relevance_score = 1.0 if keyword_data['domain_category'] else 0.5
                
                # Combined importance score
                importance_score = (
                    frequency_score * self.importance_weights['frequency'] +
                    recency_score * self.importance_weights['recency'] +
                    speaker_diversity_score * self.importance_weights['speaker_diversity'] +
                    context_relevance_score * self.importance_weights['context_relevance'] +
                    domain_relevance_score * self.importance_weights['domain_relevance']
                )
                
                important_keywords.append({
                    'word': word,
                    'importance_score': importance_score,
                    'frequency': keyword_data['frequency'],
                    'domain_category': keyword_data['domain_category'],
                    'speaker': speaker,
                    'factors': {
                        'frequency': frequency_score,
                        'recency': recency_score,
                        'speaker_diversity': speaker_diversity_score,
                        'context_relevance': context_relevance_score,
                        'domain_relevance': domain_relevance_score
                    }
                })
            
            # Sort by importance
            important_keywords.sort(key=lambda x: x['importance_score'], reverse=True)
            
            return important_keywords[:10]  # Top 10 important keywords
            
        except Exception as e:
            logger.error(f"❌ Error calculating keyword importance: {e}")
            return []
    
    def _extract_entities(self, text: str) -> List[Dict]:
        """Extract named entities (basic implementation)."""
        try:
            entities = []
            
            # Simple patterns for common entities
            patterns = {
                'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
                'url': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                'money': r'\$\d+(?:,\d{3})*(?:\.\d{2})?',
                'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
                'time': r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b'
            }
            
            for entity_type, pattern in patterns.items():
                matches = re.finditer(pattern, text)
                for match in matches:
                    entities.append({
                        'type': entity_type,
                        'value': match.group(),
                        'start_pos': match.start(),
                        'end_pos': match.end(),
                        'confidence': 0.8  # Basic confidence for pattern matching
                    })
            
            return entities
            
        except Exception as e:
            logger.error(f"❌ Error extracting entities: {e}")
            return []
    
    def _analyze_keyword_context(self, text: str, keywords: List[Dict]) -> Dict:
        """Analyze context around keywords."""
        try:
            words = text.lower().split()
            context_analysis = {
                'keyword_density': len(keywords) / len(words) if words else 0,
                'avg_keyword_frequency': statistics.mean([k['frequency'] for k in keywords]) if keywords else 0,
                'keyword_distribution': self._calculate_keyword_distribution(keywords),
                'context_coherence': self._calculate_context_coherence(text, keywords)
            }
            
            return context_analysis
            
        except Exception as e:
            logger.error(f"❌ Error analyzing keyword context: {e}")
            return {}
    
    def _detect_trending_keywords(self, session_id: str, keywords: List[Dict]) -> List[Dict]:
        """Detect trending keywords in session."""
        try:
            if session_id not in self.session_data:
                return []
            
            # Get recent keyword history
            history = self.session_data[session_id].get('keyword_history', [])
            
            if len(history) < 2:
                return []
            
            # Calculate trend for each keyword
            trending_keywords = []
            
            for keyword_data in keywords:
                word = keyword_data['word']
                
                # Get historical frequencies
                historical_freq = []
                for hist_entry in history[-5:]:  # Last 5 entries
                    freq = sum(1 for k in hist_entry.get('keywords', []) if k['word'] == word)
                    historical_freq.append(freq)
                
                if len(historical_freq) >= 2:
                    # Calculate trend
                    recent_freq = historical_freq[-1]
                    earlier_freq = statistics.mean(historical_freq[:-1])
                    
                    if earlier_freq > 0:
                        trend_ratio = recent_freq / earlier_freq
                        
                        if trend_ratio > 1.5:  # Trending up
                            trending_keywords.append({
                                'word': word,
                                'trend_direction': 'up',
                                'trend_ratio': trend_ratio,
                                'current_frequency': recent_freq,
                                'historical_average': earlier_freq
                            })
                        elif trend_ratio < 0.5:  # Trending down
                            trending_keywords.append({
                                'word': word,
                                'trend_direction': 'down',
                                'trend_ratio': trend_ratio,
                                'current_frequency': recent_freq,
                                'historical_average': earlier_freq
                            })
            
            # Sort by trend strength
            trending_keywords.sort(key=lambda x: abs(x['trend_ratio'] - 1), reverse=True)
            
            return trending_keywords[:5]  # Top 5 trending keywords
            
        except Exception as e:
            logger.error(f"❌ Error detecting trending keywords: {e}")
            return []
    
    def _get_context_words(self, words: List[str], target_word: str) -> List[str]:
        """Get context words around target word."""
        try:
            context_words = []
            window_size = 2
            
            for i, word in enumerate(words):
                if word == target_word:
                    # Get surrounding words
                    start = max(0, i - window_size)
                    end = min(len(words), i + window_size + 1)
                    
                    context = words[start:end]
                    context_words.extend([w for w in context if w != target_word])
            
            return list(set(context_words))
            
        except Exception as e:
            logger.error(f"❌ Error getting context words: {e}")
            return []
    
    def _categorize_keyword(self, word: str) -> Optional[str]:
        """Categorize keyword into domain category."""
        for category, keywords in self.topic_categories.items():
            if word in keywords:
                return category
        return None
    
    def _calculate_word_importance(self, word: str, frequency: int, total_words: int) -> float:
        """Calculate importance score for a word."""
        try:
            # TF-IDF-like scoring
            tf = frequency / total_words
            
            # Simple IDF approximation (domain-specific words get higher scores)
            idf = 2.0 if self._categorize_keyword(word) else 1.0
            
            # Length bonus (longer words are often more specific)
            length_bonus = min(len(word) / 10, 0.5)
            
            importance = tf * idf + length_bonus
            
            return importance
            
        except Exception as e:
            logger.error(f"❌ Error calculating word importance: {e}")
            return 0.0
    
    def _calculate_phrase_relevance(self, phrase: str) -> float:
        """Calculate relevance score for a phrase."""
        try:
            words = phrase.split()
            
            # Base score from word count
            base_score = len(words) * 0.1
            
            # Domain relevance bonus
            domain_bonus = sum(0.2 for word in words if self._categorize_keyword(word))
            
            # Length penalty for very long phrases
            length_penalty = max(0, len(words) - 3) * 0.05
            
            relevance = base_score + domain_bonus - length_penalty
            
            return max(0, relevance)
            
        except Exception as e:
            logger.error(f"❌ Error calculating phrase relevance: {e}")
            return 0.0
    
    def _calculate_speaker_diversity(self, session_id: str, word: str) -> float:
        """Calculate speaker diversity score for a keyword."""
        try:
            if session_id not in self.session_data:
                return 0.5
            
            history = self.session_data[session_id].get('keyword_history', [])
            
            if not history:
                return 0.5
            
            # Count unique speakers who used this word
            speakers = set()
            for entry in history:
                for keyword_data in entry.get('keywords', []):
                    if keyword_data['word'] == word:
                        speakers.add(entry.get('speaker', 'Unknown'))
            
            # Normalize by total speakers in session
            all_speakers = set()
            for entry in history:
                all_speakers.add(entry.get('speaker', 'Unknown'))
            
            if not all_speakers:
                return 0.5
            
            diversity_score = len(speakers) / len(all_speakers)
            return diversity_score
            
        except Exception as e:
            logger.error(f"❌ Error calculating speaker diversity: {e}")
            return 0.5
    
    def _calculate_context_relevance(self, word: str, context_words: List[str]) -> float:
        """Calculate context relevance score."""
        try:
            if not context_words:
                return 0.5
            
            # Count domain-relevant context words
            relevant_context = sum(1 for w in context_words if self._categorize_keyword(w))
            
            relevance_score = relevant_context / len(context_words)
            
            return relevance_score
            
        except Exception as e:
            logger.error(f"❌ Error calculating context relevance: {e}")
            return 0.5
    
    def _calculate_keyword_distribution(self, keywords: List[Dict]) -> Dict:
        """Calculate keyword distribution statistics."""
        try:
            if not keywords:
                return {}
            
            frequencies = [k['frequency'] for k in keywords]
            
            return {
                'mean_frequency': statistics.mean(frequencies),
                'median_frequency': statistics.median(frequencies),
                'std_frequency': statistics.stdev(frequencies) if len(frequencies) > 1 else 0,
                'max_frequency': max(frequencies),
                'min_frequency': min(frequencies)
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating keyword distribution: {e}")
            return {}
    
    def _calculate_context_coherence(self, text: str, keywords: List[Dict]) -> float:
        """Calculate context coherence score."""
        try:
            # Simple coherence based on keyword co-occurrence
            words = text.lower().split()
            keyword_words = [k['word'] for k in keywords]
            
            # Count keyword pairs that appear close together
            coherence_score = 0
            window_size = 5
            
            for i, word in enumerate(words):
                if word in keyword_words:
                    # Check surrounding window
                    start = max(0, i - window_size)
                    end = min(len(words), i + window_size + 1)
                    
                    window_words = words[start:end]
                    related_keywords = sum(1 for w in window_words if w in keyword_words and w != word)
                    
                    coherence_score += related_keywords
            
            # Normalize by total keywords
            if keyword_words:
                coherence_score /= len(keyword_words)
            
            return min(1.0, coherence_score)
            
        except Exception as e:
            logger.error(f"❌ Error calculating context coherence: {e}")
            return 0.0
    
    def _store_keyword_result(self, session_id: str, result: Dict):
        """Store keyword extraction result."""
        try:
            if session_id not in self.session_data:
                self.session_data[session_id] = {
                    'keyword_history': [],
                    'trending_keywords': {},
                    'topic_timeline': [],
                    'session_start': datetime.now().isoformat()
                }
            
            # Add to history
            self.session_data[session_id]['keyword_history'].append(result)
            
            # Keep only last 50 entries
            if len(self.session_data[session_id]['keyword_history']) > 50:
                self.session_data[session_id]['keyword_history'] = \
                    self.session_data[session_id]['keyword_history'][-50:]
            
            # Update trending keywords
            for trending in result.get('keywords', {}).get('trending_keywords', []):
                word = trending['word']
                self.session_data[session_id]['trending_keywords'][word] = trending
            
            # Update topic timeline
            for topic in result.get('topics', []):
                self.session_data[session_id]['topic_timeline'].append({
                    'timestamp': result['timestamp'],
                    'topic': topic['category'],
                    'relevance': topic['relevance_score'],
                    'speaker': result['speaker']
                })
            
        except Exception as e:
            logger.error(f"❌ Error storing keyword result: {e}")
    
    def get_session_keyword_summary(self, session_id: str) -> Dict:
        """Get keyword summary for session."""
        try:
            if session_id not in self.session_data:
                return {'error': 'Session not found'}
            
            session_data = self.session_data[session_id]
            history = session_data['keyword_history']
            
            if not history:
                return {'error': 'No keyword data available'}
            
            # Aggregate all keywords
            all_keywords = {}
            all_topics = defaultdict(list)
            
            for entry in history:
                for keyword_data in entry.get('keywords', {}).get('all_keywords', []):
                    word = keyword_data['word']
                    if word not in all_keywords:
                        all_keywords[word] = {
                            'word': word,
                            'total_frequency': 0,
                            'appearances': 0,
                            'speakers': set(),
                            'domain_category': keyword_data.get('domain_category')
                        }
                    
                    all_keywords[word]['total_frequency'] += keyword_data['frequency']
                    all_keywords[word]['appearances'] += 1
                    all_keywords[word]['speakers'].add(entry['speaker'])
                
                for topic in entry.get('topics', []):
                    all_topics[topic['category']].append(topic['relevance_score'])
            
            # Top keywords
            top_keywords = sorted(
                all_keywords.values(),
                key=lambda x: x['total_frequency'],
                reverse=True
            )[:20]
            
            # Convert speaker sets to counts
            for keyword in top_keywords:
                keyword['speaker_count'] = len(keyword['speakers'])
                keyword['speakers'] = list(keyword['speakers'])
            
            # Topic summary
            topic_summary = {}
            for topic, scores in all_topics.items():
                topic_summary[topic] = {
                    'avg_relevance': statistics.mean(scores),
                    'max_relevance': max(scores),
                    'appearances': len(scores),
                    'trend': 'stable'  # Could be enhanced with trend calculation
                }
            
            # Current trending keywords
            trending_keywords = list(session_data['trending_keywords'].values())
            
            return {
                'session_id': session_id,
                'top_keywords': top_keywords,
                'topic_summary': topic_summary,
                'trending_keywords': trending_keywords,
                'total_entries': len(history),
                'session_start': session_data['session_start'],
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting keyword summary: {e}")
            return {'error': str(e)}
    
    def get_live_keyword_data(self, session_id: str, limit: int = 10) -> Dict:
        """Get live keyword data for real-time display."""
        try:
            if session_id not in self.session_data:
                return {'error': 'Session not found'}
            
            history = self.session_data[session_id]['keyword_history']
            
            # Get recent entries
            recent_entries = history[-limit:] if history else []
            
            # Current keywords
            current_keywords = recent_entries[-1].get('keywords', {}).get('important_keywords', []) if recent_entries else []
            
            # Recent topics
            recent_topics = []
            for entry in recent_entries:
                for topic in entry.get('topics', []):
                    recent_topics.append({
                        'timestamp': entry['timestamp'],
                        'topic': topic['category'],
                        'relevance': topic['relevance_score'],
                        'speaker': entry['speaker']
                    })
            
            # Trending keywords
            trending_keywords = list(self.session_data[session_id]['trending_keywords'].values())
            
            return {
                'session_id': session_id,
                'current_keywords': current_keywords,
                'recent_topics': recent_topics,
                'trending_keywords': trending_keywords,
                'total_entries': len(history),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting live keyword data: {e}")
            return {'error': str(e)}

# Global keyword extractor instance
keyword_extractor = LiveKeywordExtraction()

# Integration functions
def start_keyword_extraction():
    """Start keyword extraction."""
    keyword_extractor.start_extraction()

def stop_keyword_extraction():
    """Stop keyword extraction."""
    keyword_extractor.stop_extraction()

def extract_transcript_keywords(session_id: str, transcript_data: Dict):
    """Extract keywords from transcript."""
    keyword_extractor.extract_keywords(session_id, transcript_data)

def get_keyword_summary(session_id: str) -> Dict:
    """Get keyword summary."""
    return keyword_extractor.get_session_keyword_summary(session_id)

def get_live_keywords(session_id: str, limit: int = 10) -> Dict:
    """Get live keyword data."""
    return keyword_extractor.get_live_keyword_data(session_id, limit)

def register_keyword_callback(callback):
    """Register keyword callback."""
    keyword_extractor.register_callback(callback)

if __name__ == "__main__":
    # Test the keyword extractor
    print("🚀 Testing Live Keyword Extraction...")
    
    # Start extractor
    keyword_extractor.start_extraction()
    
    # Test sentences
    test_sentences = [
        {"text": "I need help with my account billing and payment issues.", "speaker": "Customer"},
        {"text": "Let me check your account settings and subscription details.", "speaker": "Agent"},
        {"text": "The software update caused problems with the database connection.", "speaker": "Customer"},
        {"text": "I'll escalate this technical issue to our development team.", "speaker": "Agent"},
        {"text": "Thank you for your excellent customer service and support.", "speaker": "Customer"}
    ]
    
    # Extract keywords from test sentences
    for i, sentence in enumerate(test_sentences):
        transcript_data = {
            'text': sentence['text'],
            'speaker': sentence['speaker'],
            'timestamp': datetime.now().isoformat(),
            'is_final': True
        }
        
        keyword_extractor.extract_keywords(f"test_session", transcript_data)
        time.sleep(0.5)  # Small delay to see processing
    
    # Get summary
    time.sleep(2)  # Wait for processing
    summary = keyword_extractor.get_session_keyword_summary("test_session")
    print(f"🔍 Keyword Summary: {len(summary.get('top_keywords', []))} keywords extracted")
    
    # Stop extractor
    keyword_extractor.stop_extraction()
    
    print("🎉 Live Keyword Extraction test completed!") 