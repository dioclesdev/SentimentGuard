from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from flask_cors import CORS
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re
import os
from datetime import datetime
import logging
import sqlite3
from pathlib import Path
import json
from textblob import TextBlob
from transformers import pipeline
import torch

# Flask App Setup
app = Flask(__name__)
app.secret_key = 'sentiment_guard_secret_key_2024'
CORS(app)

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database Setup
DATABASE_PATH = 'sentiment_database.db'

# Sprachkonfiguration (4 Sprachen)
LANGUAGES = {
    'de': {
        'title': 'SentimentGuard - KI-gestützte Sentiment-Analyse',
        'subtitle': 'Analysieren Sie Texte auf emotionale Stimmung und speichern Sie Ergebnisse',
        'analyze_text': 'Text zur Sentiment-Analyse eingeben:',
        'placeholder': 'Fügen Sie hier den zu analysierenden Text ein...',
        'analyze_btn': 'Sentiment analysieren',
        'search_btn': 'Suchen',
        'results_title': 'Sentiment-Analyse Ergebnisse',
        'search_title': 'Textsuche',
        'search_placeholder': 'Nach Stichwörtern suchen...',
        'sentiment_score': 'Sentiment-Score',
        'classification': 'Klassifikation',
        'confidence': 'Konfidenz',
        'loading': 'Analysiere Sentiment...',
        'no_analysis': 'Noch keine Analyse',
        'enter_text': 'Geben Sie einen Text ein und klicken Sie auf "Analysieren"',
        'examples_title': 'Beispiel-Texte zum Testen:',
        'positive_example': 'Positiver Text',
        'negative_example': 'Negativer Text',
        'neutral_example': 'Neutraler Text',
        'language': 'Sprache',
        'sentiment_positive': 'Positiv',
        'sentiment_negative': 'Negativ',
        'sentiment_neutral': 'Neutral',
        'sentiment_mixed': 'Gemischt',
        'save_analysis': 'Analyse speichern',
        'saved_successfully': 'Analyse erfolgreich gespeichert',
        'search_results': 'Suchergebnisse',
        'no_results': 'Keine Ergebnisse gefunden',
        'words': 'Wörter',
        'sentences': 'Sätze',
        'characters': 'Zeichen',
        'processing_time': 'Verarbeitungszeit',
        'database_status': 'Datenbank-Status',
        'total_analyses': 'Gesamt-Analysen',
        'search_hint': 'Suche in gespeicherten Texten und Analysen',
        'export_data': 'Daten exportieren',
        'advanced_search': 'Erweiterte Suche',
        'date_range': 'Zeitraum',
        'sentiment_filter': 'Sentiment-Filter',
        'word_count_filter': 'Wortanzahl-Filter'
    },
    'en': {
        'title': 'SentimentGuard - AI-Powered Sentiment Analysis',
        'subtitle': 'Analyze text emotional sentiment and save results',
        'analyze_text': 'Enter text for sentiment analysis:',
        'placeholder': 'Paste the text you want to analyze here...',
        'analyze_btn': 'Analyze Sentiment',
        'search_btn': 'Search',
        'results_title': 'Sentiment Analysis Results',
        'search_title': 'Text Search',
        'search_placeholder': 'Search for keywords...',
        'sentiment_score': 'Sentiment Score',
        'classification': 'Classification',
        'confidence': 'Confidence',
        'loading': 'Analyzing sentiment...',
        'no_analysis': 'No analysis yet',
        'enter_text': 'Enter text and click "Analyze"',
        'examples_title': 'Example texts for testing:',
        'positive_example': 'Positive Text',
        'negative_example': 'Negative Text',
        'neutral_example': 'Neutral Text',
        'language': 'Language',
        'sentiment_positive': 'Positive',
        'sentiment_negative': 'Negative',
        'sentiment_neutral': 'Neutral',
        'sentiment_mixed': 'Mixed',
        'save_analysis': 'Save Analysis',
        'saved_successfully': 'Analysis saved successfully',
        'search_results': 'Search Results',
        'no_results': 'No results found',
        'words': 'Words',
        'sentences': 'Sentences',
        'characters': 'Characters',
        'processing_time': 'Processing Time',
        'database_status': 'Database Status',
        'total_analyses': 'Total Analyses',
        'search_hint': 'Search in saved texts and analyses',
        'export_data': 'Export Data',
        'advanced_search': 'Advanced Search',
        'date_range': 'Date Range',
        'sentiment_filter': 'Sentiment Filter',
        'word_count_filter': 'Word Count Filter'
    },
    'fr': {
        'title': 'SentimentGuard - Analyse de Sentiment IA',
        'subtitle': 'Analysez le sentiment émotionnel du texte et sauvegardez les résultats',
        'analyze_text': 'Entrez le texte pour l\'analyse de sentiment:',
        'placeholder': 'Collez ici le texte que vous souhaitez analyser...',
        'analyze_btn': 'Analyser le Sentiment',
        'search_btn': 'Rechercher',
        'results_title': 'Résultats de l\'Analyse de Sentiment',
        'search_title': 'Recherche de Texte',
        'search_placeholder': 'Rechercher des mots-clés...',
        'sentiment_score': 'Score de Sentiment',
        'classification': 'Classification',
        'confidence': 'Confiance',
        'loading': 'Analyse du sentiment...',
        'no_analysis': 'Aucune analyse encore',
        'enter_text': 'Entrez un texte et cliquez sur "Analyser"',
        'examples_title': 'Exemples de textes pour tester:',
        'positive_example': 'Texte Positif',
        'negative_example': 'Texte Négatif',
        'neutral_example': 'Texte Neutre',
        'language': 'Langue',
        'sentiment_positive': 'Positif',
        'sentiment_negative': 'Négatif',
        'sentiment_neutral': 'Neutre',
        'sentiment_mixed': 'Mixte',
        'save_analysis': 'Sauvegarder l\'Analyse',
        'saved_successfully': 'Analyse sauvegardée avec succès',
        'search_results': 'Résultats de Recherche',
        'no_results': 'Aucun résultat trouvé',
        'words': 'Mots',
        'sentences': 'Phrases',
        'characters': 'Caractères',
        'processing_time': 'Temps de Traitement',
        'database_status': 'État de la Base de Données',
        'total_analyses': 'Analyses Totales',
        'search_hint': 'Recherche dans les textes et analyses sauvegardés',
        'export_data': 'Exporter les Données',
        'advanced_search': 'Recherche Avancée',
        'date_range': 'Plage de Dates',
        'sentiment_filter': 'Filtre de Sentiment',
        'word_count_filter': 'Filtre de Nombre de Mots'
    },
    'es': {
        'title': 'SentimentGuard - Análisis de Sentimiento IA',
        'subtitle': 'Analice el sentimiento emocional del texto y guarde los resultados',
        'analyze_text': 'Ingrese texto para análisis de sentimiento:',
        'placeholder': 'Pegue aquí el texto que quiere analizar...',
        'analyze_btn': 'Analizar Sentimiento',
        'search_btn': 'Buscar',
        'results_title': 'Resultados del Análisis de Sentimiento',
        'search_title': 'Búsqueda de Texto',
        'search_placeholder': 'Buscar palabras clave...',
        'sentiment_score': 'Puntuación de Sentimiento',
        'classification': 'Clasificación',
        'confidence': 'Confianza',
        'loading': 'Analizando sentimiento...',
        'no_analysis': 'Sin análisis aún',
        'enter_text': 'Ingrese texto y haga clic en "Analizar"',
        'examples_title': 'Textos de ejemplo para probar:',
        'positive_example': 'Texto Positivo',
        'negative_example': 'Texto Negativo',
        'neutral_example': 'Texto Neutral',
        'language': 'Idioma',
        'sentiment_positive': 'Positivo',
        'sentiment_negative': 'Negativo',
        'sentiment_neutral': 'Neutral',
        'sentiment_mixed': 'Mixto',
        'save_analysis': 'Guardar Análisis',
        'saved_successfully': 'Análisis guardado exitosamente',
        'search_results': 'Resultados de Búsqueda',
        'no_results': 'No se encontraron resultados',
        'words': 'Palabras',
        'sentences': 'Oraciones',
        'characters': 'Caracteres',
        'processing_time': 'Tiempo de Procesamiento',
        'database_status': 'Estado de la Base de Datos',
        'total_analyses': 'Análisis Totales',
        'search_hint': 'Buscar en textos y análisis guardados',
        'export_data': 'Exportar Datos',
        'advanced_search': 'Búsqueda Avanzada',
        'date_range': 'Rango de Fechas',
        'sentiment_filter': 'Filtro de Sentimiento',
        'word_count_filter': 'Filtro de Conteo de Palabras'
    }
}

class DatabaseManager:
    """Verwaltet die SQLite-Datenbank für Sentiment-Analysen"""
    
    def __init__(self, db_path=DATABASE_PATH):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialisiert die Datenbank mit erweiterten Tabellen"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Haupttabelle für Sentiment-Analysen
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS sentiment_analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    sentiment_score REAL NOT NULL,
                    classification TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    word_count INTEGER,
                    sentence_count INTEGER,
                    char_count INTEGER,
                    positive_score REAL,
                    negative_score REAL,
                    neutral_score REAL,
                    compound_score REAL,
                    processing_time REAL,
                    language TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    tags TEXT,
                    notes TEXT
                )
                ''')
                
                # Volltext-Suchindex für bessere Performance
                cursor.execute('''
                CREATE VIRTUAL TABLE IF NOT EXISTS sentiment_fts USING fts5(
                    text, 
                    classification, 
                    tags, 
                    notes,
                    content='sentiment_analyses',
                    content_rowid='id'
                )
                ''')
                
                # Trigger für automatisches Update des FTS-Index
                cursor.execute('''
                CREATE TRIGGER IF NOT EXISTS sentiment_fts_insert AFTER INSERT ON sentiment_analyses BEGIN
                    INSERT INTO sentiment_fts(rowid, text, classification, tags, notes) 
                    VALUES (new.id, new.text, new.classification, new.tags, new.notes);
                END
                ''')
                
                cursor.execute('''
                CREATE TRIGGER IF NOT EXISTS sentiment_fts_delete AFTER DELETE ON sentiment_analyses BEGIN
                    DELETE FROM sentiment_fts WHERE rowid = old.id;
                END
                ''')
                
                cursor.execute('''
                CREATE TRIGGER IF NOT EXISTS sentiment_fts_update AFTER UPDATE ON sentiment_analyses BEGIN
                    DELETE FROM sentiment_fts WHERE rowid = old.id;
                    INSERT INTO sentiment_fts(rowid, text, classification, tags, notes) 
                    VALUES (new.id, new.text, new.classification, new.tags, new.notes);
                END
                ''')
                
                conn.commit()
                logger.info("✅ Datenbank erfolgreich initialisiert")
                
        except Exception as e:
            logger.error(f"❌ Datenbankfehler: {e}")
    
    def save_analysis(self, analysis_data):
        """Speichert eine Sentiment-Analyse in der Datenbank"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                INSERT INTO sentiment_analyses (
                    text, sentiment_score, classification, confidence,
                    word_count, sentence_count, char_count,
                    positive_score, negative_score, neutral_score, compound_score,
                    processing_time, language, tags, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    analysis_data['text'],
                    analysis_data['sentiment_score'],
                    analysis_data['classification'],
                    analysis_data['confidence'],
                    analysis_data.get('word_count', 0),
                    analysis_data.get('sentence_count', 0),
                    analysis_data.get('char_count', 0),
                    analysis_data.get('positive_score', 0),
                    analysis_data.get('negative_score', 0),
                    analysis_data.get('neutral_score', 0),
                    analysis_data.get('compound_score', 0),
                    analysis_data.get('processing_time', 0),
                    analysis_data.get('language', 'de'),
                    analysis_data.get('tags', ''),
                    analysis_data.get('notes', '')
                ))
                
                analysis_id = cursor.lastrowid
                conn.commit()
                
                logger.info(f"💾 Analyse gespeichert mit ID: {analysis_id}")
                return analysis_id
                
        except Exception as e:
            logger.error(f"❌ Fehler beim Speichern: {e}")
            return None
    
    def search_texts(self, query, filters=None):
        """Sucht in gespeicherten Texten mit erweiterten Filtern"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if not query and not filters:
                    # Alle Analysen zurückgeben
                    cursor.execute('''
                    SELECT * FROM sentiment_analyses 
                    ORDER BY created_at DESC 
                    LIMIT 100
                    ''')
                elif query:
                    # Volltext-Suche
                    cursor.execute('''
                    SELECT sentiment_analyses.* 
                    FROM sentiment_analyses 
                    JOIN sentiment_fts ON sentiment_analyses.id = sentiment_fts.rowid
                    WHERE sentiment_fts MATCH ?
                    ORDER BY sentiment_analyses.created_at DESC
                    LIMIT 100
                    ''', (query,))
                else:
                    # Nur Filter ohne Suchbegriff
                    cursor.execute('''
                    SELECT * FROM sentiment_analyses 
                    WHERE 1=1
                    ORDER BY created_at DESC 
                    LIMIT 100
                    ''')
                
                results = cursor.fetchall()
                
                # Konvertiere zu Dictionary für bessere Handhabung
                columns = [description[0] for description in cursor.description]
                return [dict(zip(columns, row)) for row in results]
                
        except Exception as e:
            logger.error(f"❌ Suchfehler: {e}")
            return []
    
    def get_statistics(self):
        """Holt Statistiken aus der Datenbank"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Gesamt-Analysen
                cursor.execute('SELECT COUNT(*) FROM sentiment_analyses')
                result = cursor.fetchone()
                total_analyses = result[0] if result else 0
                
                # Sentiment-Verteilung
                cursor.execute('''
                SELECT classification, COUNT(*) 
                FROM sentiment_analyses 
                GROUP BY classification
                ''')
                sentiment_distribution = dict(cursor.fetchall())
                
                # Durchschnittliche Scores
                cursor.execute('''
                SELECT 
                    AVG(sentiment_score) as avg_sentiment,
                    AVG(confidence) as avg_confidence,
                    AVG(word_count) as avg_words
                FROM sentiment_analyses
                ''')
                averages = cursor.fetchone()
                
                return {
                    'total_analyses': total_analyses,
                    'sentiment_distribution': sentiment_distribution,
                    'average_sentiment': averages[0] if averages and averages[0] is not None else 0.5,
                    'average_confidence': averages[1] if averages and averages[1] is not None else 0.0,
                    'average_word_count': averages[2] if averages and averages[2] is not None else 0
                }
                
        except Exception as e:
            logger.error(f"❌ Statistikfehler: {e}")
            return {
                'total_analyses': 0,
                'sentiment_distribution': {},
                'average_sentiment': 0.5,
                'average_confidence': 0.0,
                'average_word_count': 0
            }

class SentimentAnalyzer:
    """Erweiterte Sentiment-Analyse mit mehreren Methoden"""
    
    def __init__(self):
        self.nltk_analyzer = None
        self.bert_analyzer = None
        self.setup_analyzers()
    
    def setup_analyzers(self):
        """Initialisiert alle verfügbaren Analyzer"""
        # NLTK VADER Setup
        try:
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('punkt', quiet=True)
            self.nltk_analyzer = SentimentIntensityAnalyzer()
            logger.info("✅ NLTK VADER Analyzer geladen")
        except Exception as e:
            logger.warning(f"⚠️ NLTK Setup fehlgeschlagen: {e}")
        
        # BERT Setup für bessere Sentiment-Analyse
        try:
            self.bert_analyzer = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                top_k=None  # Ersetzt return_all_scores=True
            )
            logger.info("✅ BERT Sentiment Analyzer geladen")
        except Exception as e:
            logger.warning(f"⚠️ BERT Setup fehlgeschlagen: {e}")
    
    def analyze_text(self, text, use_bert=True):
        """Führt umfassende Sentiment-Analyse durch"""
        if not text or len(text.strip()) < 3:
            return self._empty_result()
        
        start_time = datetime.now()
        
        # Basis-Features extrahieren
        features = self._extract_text_features(text)
        
        # NLTK VADER Analyse
        nltk_result = self._analyze_with_vader(text)
        
        # BERT Analyse (falls verfügbar)
        bert_result = None
        if self.bert_analyzer and use_bert:
            bert_result = self._analyze_with_bert(text)
        
        # TextBlob als zusätzliche Referenz
        textblob_result = self._analyze_with_textblob(text)
        
        # Kombiniere Ergebnisse
        final_score, confidence = self._combine_results(nltk_result, bert_result, textblob_result)
        
        # Klassifikation
        classification = self._classify_sentiment(final_score, confidence)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            'sentiment_score': final_score,
            'classification': classification,
            'confidence': confidence,
            'positive_score': nltk_result.get('pos', 0),
            'negative_score': nltk_result.get('neg', 0),
            'neutral_score': nltk_result.get('neu', 0),
            'compound_score': nltk_result.get('compound', 0),
            'word_count': features['word_count'],
            'sentence_count': features['sentence_count'],
            'char_count': features['char_count'],
            'processing_time': processing_time,
            'text': text,
            'methods_used': self._get_methods_used(bert_result),
            'detailed_scores': {
                'vader': nltk_result,
                'bert': bert_result,
                'textblob': textblob_result
            }
        }
    
    def _extract_text_features(self, text):
        """Extrahiert grundlegende Text-Features"""
        words = text.split()
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'char_count': len(text),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0
        }
    
    def _analyze_with_vader(self, text):
        """VADER Sentiment-Analyse"""
        if self.nltk_analyzer:
            try:
                scores = self.nltk_analyzer.polarity_scores(text)
                return scores
            except Exception as e:
                logger.warning(f"⚠️ VADER Analyse fehlgeschlagen: {e}")
        
        return {'compound': 0, 'pos': 0, 'neu': 1, 'neg': 0}
    
    def _analyze_with_bert(self, text):
        """BERT Sentiment-Analyse"""
        try:
            # Kürze Text für BERT falls nötig
            if len(text) > 512:
                text = text[:512]
            
            results = self.bert_analyzer(text)
            
            # Konvertiere BERT-Ergebnisse zu einheitlichem Format
            sentiment_scores = {}
            for result in results[0]:
                label = result['label'].lower()
                score = result['score']
                
                if '5' in label or 'very_positive' in label or label == 'positive':
                    sentiment_scores['positive'] = score
                elif '1' in label or 'very_negative' in label or label == 'negative':
                    sentiment_scores['negative'] = score
                else:
                    sentiment_scores['neutral'] = score
            
            return sentiment_scores
            
        except Exception as e:
            logger.warning(f"⚠️ BERT Analyse fehlgeschlagen: {e}")
            return None
    
    def _analyze_with_textblob(self, text):
        """TextBlob Sentiment-Analyse als Referenz"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 bis +1
            subjectivity = blob.sentiment.subjectivity  # 0 bis 1
            
            return {
                'polarity': polarity,
                'subjectivity': subjectivity
            }
        except Exception as e:
            logger.warning(f"⚠️ TextBlob Analyse fehlgeschlagen: {e}")
            return {'polarity': 0, 'subjectivity': 0.5}
    
    def _combine_results(self, vader_result, bert_result, textblob_result):
        """Kombiniert Ergebnisse verschiedener Analyzer"""
        scores = []
        weights = []
        
        # VADER Score (bewährt für Social Media Text)
        if vader_result:
            vader_score = (vader_result['compound'] + 1) / 2  # Normalisiere -1,1 zu 0,1
            scores.append(vader_score)
            weights.append(0.4)
        
        # BERT Score (bessere Kontext-Verständnis)
        if bert_result:
            if 'positive' in bert_result:
                bert_score = bert_result['positive']
            else:
                # Fallback Berechnung
                pos = bert_result.get('positive', 0)
                neg = bert_result.get('negative', 0)
                bert_score = pos / (pos + neg) if (pos + neg) > 0 else 0.5
            
            scores.append(bert_score)
            weights.append(0.4)
        
        # TextBlob Score (zusätzliche Referenz)
        if textblob_result:
            textblob_score = (textblob_result['polarity'] + 1) / 2  # Normalisiere
            scores.append(textblob_score)
            weights.append(0.2)
        
        # Gewichteter Durchschnitt
        if scores:
            final_score = np.average(scores, weights=weights)
            confidence = 1 - np.std(scores) if len(scores) > 1 else 0.8
        else:
            final_score = 0.5
            confidence = 0.1
        
        return final_score, min(confidence, 1.0)
    
    def _classify_sentiment(self, score, confidence):
        """Klassifiziert Sentiment basierend auf Score und Confidence"""
        if confidence < 0.3:
            return 'mixed'
        elif score > 0.65:
            return 'positive'
        elif score < 0.35:
            return 'negative'
        else:
            return 'neutral'
    
    def _get_methods_used(self, bert_result):
        """Listet verwendete Analyse-Methoden"""
        methods = ['VADER', 'TextBlob']
        if bert_result:
            methods.append('BERT')
        return methods
    
    def _empty_result(self):
        """Leeres Ergebnis für ungültige Eingaben"""
        return {
            'sentiment_score': 0.5,
            'classification': 'neutral',
            'confidence': 0.0,
            'error': 'Text zu kurz oder ungültig'
        }

# Globale Instanzen
db_manager = DatabaseManager()
sentiment_analyzer = SentimentAnalyzer()

def get_language():
    """Ermittelt die aktuelle Sprache aus der Session"""
    return session.get('language', 'de')

def get_text(key):
    """Holt lokalisierten Text"""
    lang = get_language()
    return LANGUAGES.get(lang, LANGUAGES['de']).get(key, key)

# Beispiel-Texte für 4 Sprachen
EXAMPLE_TEXTS = {
    'de': {
        'positive': "Ich bin unglaublich glücklich und dankbar für diese wunderbare Gelegenheit! Es ist ein fantastischer Tag und alles läuft perfekt. Die Ergebnisse übertreffen alle Erwartungen.",
        'negative': "Das ist wirklich enttäuschend und frustrierend. Nichts funktioniert wie geplant und ich bin sehr unzufrieden mit der ganzen Situation. Es könnte nicht schlimmer sein.",
        'neutral': "Die Quartalszahlen zeigen eine Steigerung von 3,2 Prozent im Vergleich zum Vorjahr. Die Geschäftsführung wird morgen um 14 Uhr eine Pressekonferenz abhalten."
    },
    'en': {
        'positive': "I am incredibly happy and grateful for this wonderful opportunity! It's a fantastic day and everything is going perfectly. The results exceed all expectations.",
        'negative': "This is really disappointing and frustrating. Nothing is working as planned and I am very dissatisfied with the whole situation. It couldn't be worse.",
        'neutral': "The quarterly figures show an increase of 3.2 percent compared to the previous year. Management will hold a press conference tomorrow at 2 PM."
    },
    'fr': {
        'positive': "Je suis incroyablement heureux et reconnaissant pour cette merveilleuse opportunité! C'est une journée fantastique et tout se passe parfaitement. Les résultats dépassent toutes les attentes.",
        'negative': "C'est vraiment décevant et frustrant. Rien ne fonctionne comme prévu et je suis très mécontent de toute la situation. Ça ne pourrait pas être pire.",
        'neutral': "Les chiffres trimestriels montrent une augmentation de 3,2 pour cent par rapport à l'année précédente. La direction tiendra une conférence de presse demain à 14h."
    },
    'es': {
        'positive': "¡Estoy increíblemente feliz y agradecido por esta maravillosa oportunidad! Es un día fantástico y todo va perfectamente. Los resultados superan todas las expectativas.",
        'negative': "Esto es realmente decepcionante y frustrante. Nada funciona como estaba planeado y estoy muy insatisfecho con toda la situación. No podría ser peor.",
        'neutral': "Las cifras trimestrales muestran un aumento del 3,2 por ciento en comparación con el año anterior. La dirección celebrará una conferencia de prensa mañana a las 14:00."
    }
}

# Flask Routes
@app.route('/')
def index():
    """Hauptseite mit Sentiment-Analyse Interface"""
    lang = get_language()
    stats = db_manager.get_statistics()
    
    return render_template('index.html', 
                         lang=lang, 
                         get_text=get_text,
                         example_texts=EXAMPLE_TEXTS[lang],
                         supported_languages=['de', 'en', 'fr', 'es'],
                         stats=stats)

@app.route('/set_language/<language>')
def set_language(language):
    """Sprache wechseln"""
    if language in LANGUAGES:
        session['language'] = language
    return redirect(url_for('index'))

@app.route('/api/analyze', methods=['POST'])
def analyze_sentiment():
    """Sentiment-Analyse durchführen"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Kein Text in der Anfrage gefunden'}), 400
        
        text = data['text']
        use_bert = data.get('use_bert', True)
        save_result = data.get('save', False)
        
        if not isinstance(text, str) or len(text.strip()) < 3:
            return jsonify({'error': 'Text ist zu kurz (mindestens 3 Zeichen erforderlich)'}), 400
        
        # Sentiment-Analyse durchführen
        result = sentiment_analyzer.analyze_text(text, use_bert=use_bert)
        result['language'] = get_language()
        
        # Optional: Ergebnis in Datenbank speichern
        if save_result:
            analysis_id = db_manager.save_analysis(result)
            result['saved_id'] = analysis_id
            result['saved'] = True
        
        logger.info(f"📊 Sentiment analysiert - Score: {result.get('sentiment_score', 0):.2f}, Klassifikation: {result.get('classification', 'unknown')}")
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"❌ Fehler in analyze_sentiment: {str(e)}")
        return jsonify({'error': f'Server-Fehler: {str(e)}'}), 500

@app.route('/api/search', methods=['GET'])
def search_texts():
    """Suche in gespeicherten Texten"""
    try:
        query = request.args.get('q', '').strip()
        
        # Erweiterte Filter (optional für zukünftige Entwicklung)
        filters = {
            'sentiment': request.args.get('sentiment'),
            'date_from': request.args.get('date_from'),
            'date_to': request.args.get('date_to'),
            'min_words': request.args.get('min_words'),
            'max_words': request.args.get('max_words')
        }
        
        results = db_manager.search_texts(query, filters)
        
        logger.info(f"🔍 Suche nach '{query}' - {len(results)} Ergebnisse")
        
        return jsonify({
            'query': query,
            'results': results,
            'total': len(results)
        })
    
    except Exception as e:
        logger.error(f"❌ Suchfehler: {str(e)}")
        return jsonify({'error': f'Suchfehler: {str(e)}'}), 500

@app.route('/api/save', methods=['POST'])
def save_analysis():
    """Speichert eine Analyse nachträglich"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Keine Daten erhalten'}), 400
        
        analysis_id = db_manager.save_analysis(data)
        
        if analysis_id:
            return jsonify({
                'success': True,
                'message': 'Analyse erfolgreich gespeichert',
                'id': analysis_id
            })
        else:
            return jsonify({'error': 'Fehler beim Speichern'}), 500
    
    except Exception as e:
        logger.error(f"❌ Speicherfehler: {str(e)}")
        return jsonify({'error': f'Speicherfehler: {str(e)}'}), 500

@app.route('/api/example/<example_type>')
def get_example_text(example_type):
    """Beispiel-Text für die aktuelle Sprache abrufen"""
    lang = get_language()
    examples = EXAMPLE_TEXTS.get(lang, EXAMPLE_TEXTS['de'])
    
    if example_type in examples:
        return jsonify({'text': examples[example_type]})
    else:
        return jsonify({'error': 'Beispiel-Text nicht gefunden'}), 404

@app.route('/api/statistics')
def get_statistics():
    """Holt aktuelle Datenbank-Statistiken"""
    try:
        stats = db_manager.get_statistics()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"❌ Statistikfehler: {str(e)}")
        return jsonify({'error': f'Statistikfehler: {str(e)}'}), 500

@app.route('/api/export')
def export_data():
    """Exportiert alle Analysen als JSON"""
    try:
        all_analyses = db_manager.search_texts('', {})
        
        return jsonify({
            'export_date': datetime.now().isoformat(),
            'total_records': len(all_analyses),
            'data': all_analyses
        })
    
    except Exception as e:
        logger.error(f"❌ Exportfehler: {str(e)}")
        return jsonify({'error': f'Exportfehler: {str(e)}'}), 500

@app.route('/api/health')
def health_check():
    """System-Health Check"""
    stats = db_manager.get_statistics()
    
    return jsonify({
        'status': 'healthy',
        'analyzers_available': {
            'vader': sentiment_analyzer.nltk_analyzer is not None,
            'bert': sentiment_analyzer.bert_analyzer is not None
        },
        'database_status': 'connected',
        'total_analyses': stats.get('total_analyses', 0),
        'supported_languages': list(LANGUAGES.keys()),
        'timestamp': datetime.now().isoformat(),
        'version': '1.0-sentiment-analysis'
    })

if __name__ == '__main__':
    logger.info("🚀 Starte SentimentGuard Flask App mit 4-Sprachen-Support...")
    
    # Erstelle notwendige Verzeichnisse
    Path("templates").mkdir(exist_ok=True)
    Path("static").mkdir(exist_ok=True)
    
    # Informationen über Features
    logger.info("📋 SENTIMENTGUARD FEATURES:")
    logger.info("  • 4 Sprachen: Deutsch, English, Français, Español")
    logger.info("  • Mehrfach-Sentiment-Analyse: VADER + BERT + TextBlob")
    logger.info("  • SQLite-Datenbank mit Volltext-Suche")
    logger.info("  • Erweiterte Suchfunktionen")
    logger.info("  • Export-Funktionalität")
    logger.info("  • Responsive Web-Interface")
    
    logger.info("🌍 Server verfügbar auf:")
    logger.info("  • http://localhost:5000 (alle Sprachen)")
    logger.info("  • http://localhost:5000/set_language/de (Deutsch)")
    logger.info("  • http://localhost:5000/set_language/en (English)")
    logger.info("  • http://localhost:5000/set_language/fr (Français)")
    logger.info("  • http://localhost:5000/set_language/es (Español)")
    
    app.run(host='0.0.0.0', port=5000, debug=True)