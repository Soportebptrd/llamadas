import os
import re
import json
import time
import zipfile
import pandas as pd
import streamlit as st
from datetime import datetime
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import whisper
from transformers.pipelines import pipeline
from sentence_transformers import SentenceTransformer, util
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import numpy as np
from urllib.parse import quote
import matplotlib.ticker as ticker
from matplotlib import rcParams

# Configuración de estilo para gráficos
try:
    plt.style.use('seaborn-v0_8')  # Para versiones recientes
except:
    sns.set_theme(style="whitegrid")  # Fallback a Seaborn moderno

rcParams['font.family'] = 'DejaVu Sans'
rcParams['axes.grid'] = True
rcParams['grid.alpha'] = 0.3
rcParams['axes.titlesize'] = 14
rcParams['axes.titleweight'] = 'bold'

# Configuración de estilo preferida (al inicio de tu código)
sns.set_theme(
    style="whitegrid",
    font="DejaVu Sans",
    rc={
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'grid.alpha': 0.3
    }
)

class Config:
    def __init__(self):
        self.DRIVE_FOLDER_ID = '16ZPWdgiTTtiSO5n5_uFmHeTCLlbg5DPu'
        self.WHISPER_MODEL = 'medium'
        self.SENTIMENT_MODEL = 'distilbert-base-uncased-finetuned-sst-2-english'
        self.SIMILARITY_MODEL = 'all-MiniLM-L6-v2'
        self.SCRIPT_FILE = 'guion_oficial.txt'
        self.REQUIRED_PHRASES = ['saludo', 'presentación', 'oferta', 'cierre']
        self.OUTPUT_FOLDER = 'results'
        self.AUDIO_FOLDER = 'downloaded_audios'
        self.MODEL_FOLDER = 'models'
        self.BATCH_SIZE = 3
        self.THRESHOLDS = {
            'adherence': 0.7,
            'sentiment': 0.7,
            'interruptions': 3
        }
        self.SHEET_NAME = "CallMonitoringResults"
        self.DRIVE_BASE_URL = "https://drive.google.com/file/d/"

class GoogleSheetsManager:
    def __init__(self, config):
        self.config = config
        self.scope = ['https://spreadsheets.google.com/feeds',
                     'https://www.googleapis.com/auth/drive']
        self.credentials = ServiceAccountCredentials.from_json_keyfile_name(
            'service_account.json', self.scope)
        self.client = gspread.authorize(self.credentials)
        
    def get_sheet(self):
        try:
            return self.client.open(self.config.SHEET_NAME).sheet1
        except gspread.SpreadsheetNotFound:
            return self.client.create(self.config.SHEET_NAME).sheet1
    
    def initialize_sheet(self):
        sheet = self.get_sheet()
        if len(sheet.get_all_values()) == 0:
            sheet.append_row([
                'Vendedor', 'Audio', 'Enlace Audio', 'Fecha', 'Apego (%)', 'Duración',
                'Sentimiento (%)', 'Nuevas Frases Aprendidas', 'Interrupciones',
                'Frases Coincidentes', 'Frases Faltantes', 'Duración (s)',
                'Cambios de Hablante', 'Segmentos Vendedor', 'Segmentos Cliente'
            ])
        return sheet

class DriveConnector:
    def __init__(self):
        self.gauth = GoogleAuth()
        self._setup_auth()
        self.drive = GoogleDrive(self.gauth)
        self.config = Config()
    
    def _setup_auth(self):
        if not os.path.exists('settings.yaml'):
            with open('settings.yaml', 'w') as f:
                f.write("""
client_config_backend: file
client_config_file: client_secrets.json
save_credentials: true
save_credentials_backend: file
save_credentials_file: credentials.json
get_refresh_token: true
access_type: offline
""")
        try:
            if os.path.exists('credentials.json'):
                self.gauth.LoadCredentialsFile('credentials.json')
            
            if self.gauth.credentials is None:
                self.gauth.LocalWebserverAuth()
            elif self.gauth.access_token_expired:
                self.gauth.Refresh()
            else:
                self.gauth.Authorize()
            
            self.gauth.SaveCredentialsFile('credentials.json')
        except Exception as e:
            st.error(f"Error de autenticación: {str(e)}")
            st.stop()
    
    def get_audio_url(self, file_id):
        return f"{self.config.DRIVE_BASE_URL}{file_id}/view?usp=drivesdk"
    
    def get_sellers_and_audios(self, folder_id):
        sellers = {}
        seller_folders = self.drive.ListFile({
            'q': f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        }).GetList()
        
        for seller_folder in seller_folders:
            seller_name = seller_folder['title']
            audio_files = self.drive.ListFile({
                'q': f"'{seller_folder['id']}' in parents and (mimeType='audio/mpeg' or mimeType='audio/wav') and trashed=false"
            }).GetList()
            
            sellers[seller_name] = {
                'folder_id': seller_folder['id'],
                'audios': [{
                    'id': audio['id'],
                    'title': audio['title'],
                    'url': self.get_audio_url(audio['id']),
                    'createdDate': audio['createdDate'],
                    'modifiedDate': audio['modifiedDate']
                } for audio in audio_files]
            }
        return sellers
    
    def download_audio(self, file_id, file_name):
        os.makedirs(self.config.AUDIO_FOLDER, exist_ok=True)
        file_path = os.path.join(self.config.AUDIO_FOLDER, file_name)
        
        if not os.path.exists(file_path):
            file = self.drive.CreateFile({'id': file_id})
            file.GetContentFile(file_path)
        return file_path
    
def _apply_color(self, value, threshold=60):
    """Aplica color rojo/verde según el valor"""
    color = "red" if value < threshold else "green"
    return f"color: {color}; font-weight: bold;"

class AudioProcessor:
    def __init__(self, config):
        self.config = config
        self.model = whisper.load_model(self.config.WHISPER_MODEL)
        self.transcriptions_folder = os.path.join(config.OUTPUT_FOLDER, 'transcriptions')
        os.makedirs(self.transcriptions_folder, exist_ok=True)
    
    def transcribe_audio(self, audio_path):
        result = self.model.transcribe(audio_path)
        
        # Agregar metadatos adicionales
        result['metadata'] = {
            'audio_file': os.path.basename(audio_path),
            'processing_date': datetime.now().isoformat(),
            'model': self.config.WHISPER_MODEL
        }
        return result
    
    def analyze_audio_features(self, audio_path):
        y, sr = librosa.load(audio_path)
        duration = librosa.get_duration(y=y, sr=sr)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        rms = librosa.feature.rms(y=y)[0]
        
        return {
            'duration': duration,
            'interruptions': len(frames),
            'speech_energy': np.mean(rms),
            'sample_rate': sr,
            'audio_array': y
        }

class TextAnalyzer:
    def __init__(self, config):
        self.config = config
        print(f"Ruta del guión: {os.path.abspath(self.config.SCRIPT_FILE)}")
        
        # Cargar modelos primero
        self.sentiment_analyzer = pipeline("sentiment-analysis", 
                                         model=self.config.SENTIMENT_MODEL)
        self.similarity_model = SentenceTransformer(self.config.SIMILARITY_MODEL)
        
        # Cargar frases del guión con manejo de errores
        self.script_phrases = self._load_script_phrases()
        
        if not self.script_phrases:
            st.warning("⚠️ El guión oficial está vacío o no se cargó correctamente")
        else:
            print(f"✅ {len(self.script_phrases)} frases cargadas del guión")
    
    def _load_script_phrases(self):
        encodings_to_try = ['utf-8', 'latin-1', 'utf-16', 'iso-8859-1']
        
        if os.path.exists(self.config.SCRIPT_FILE):
            for encoding in encodings_to_try:
                try:
                    with open(self.config.SCRIPT_FILE, 'r', encoding=encoding) as f:
                        phrases = [line.strip() for line in f if line.strip()]
                        print(f"Guion cargado con codificación: {encoding}")
                        return phrases
                except UnicodeDecodeError:
                    continue
            
            st.error(f"No se pudo leer el archivo {self.config.SCRIPT_FILE} con ninguna codificación conocida")
            return []
        else:
            st.error(f"Archivo de guión no encontrado: {self.config.SCRIPT_FILE}")
            return []
    
    def classify_role(self, text):
        text_lower = text.lower()
        if any(kw in text_lower for kw in ['hola', 'buenos días', 'gracias por llamar', 'mi nombre es', 
                                          'le habla', 'permítame', 'ofrecer', 'promoción', 
                                          'gracias por su tiempo', 'alguna otra consulta']):
            return 0  # vendedor
        elif any(kw in text_lower for kw in ['quiero', 'necesito', 'cuánto cuesta', 'tienen',
                                          'disculpe', 'pero', 'problema', 'queja',
                                          'no me', 'no está', 'no funciona']):
            return 1  # cliente
        return 0  # por defecto asumimos vendedor
    
    def classify_phrase_type(self, text):
        text_lower = text.lower()
        if any(w in text_lower for w in ['hola', 'buenos días', 'buenas tardes']):
            return 'saludo'
        elif any(w in text_lower for w in ['oferta', 'promoción', 'descuento', 'producto']):
            return 'oferta'
        elif any(w in text_lower for w in ['objeción', 'preocupación', 'problema', 'caro']):
            return 'objeción'
        elif any(w in text_lower for w in ['gracias', 'adiós', 'hasta luego', 'chao']):
            return 'cierre'
        elif '?' in text:
            return 'pregunta'
        return 'otro'
    
    def calculate_similarity(self, text1, text2):
        embedding1 = self.similarity_model.encode(text1, convert_to_tensor=True)
        embedding2 = self.similarity_model.encode(text2, convert_to_tensor=True)
        return util.pytorch_cos_sim(embedding1, embedding2).item()
    
    def analyze_sentiment(self, text):
        result = self.sentiment_analyzer(text)
        if isinstance(result, list) and isinstance(result[0], dict):
            return result[0]
        return {"label": "NEUTRAL", "score": 0.5}

class MetricsGenerator:
    def __init__(self, config):
        self.config = config
        self.text_analyzer = TextAnalyzer(config)
    
    def classify_speakers(self, transcription):
        """Clasifica cada segmento como vendedor (0) o cliente (1)"""
        speakers = []
        for segment in transcription['segments']:
            text = segment['text'].strip()
            speakers.append(self.text_analyzer.classify_role(text))
        
        # Post-procesamiento para corregir cambios improbables
        speakers = self._smooth_speaker_changes(speakers)
        return speakers
    
    def _smooth_speaker_changes(self, speakers):
        """Suaviza cambios bruscos de speaker"""
        if len(speakers) < 3:
            return speakers
            
        for i in range(1, len(speakers)-1):
            # Si hay un cambio aislado (ej. [0,1,0] -> [0,0,0])
            if speakers[i-1] == speakers[i+1] and speakers[i] != speakers[i-1]:
                speakers[i] = speakers[i-1]
        
        return speakers
    
    def calculate_script_adherence(self, transcription, speakers):
        total_similarity = 0
        count = 0
        matched_phrases = []
        positive_matches = 0
        negative_matches = 0
        
        for seg, speaker in zip(transcription['segments'], speakers):
            if speaker == 0:  # Solo segmentos del vendedor
                text = seg['text'].lower().strip()
                best_match = ("", 0)
                
                for phrase in self.text_analyzer.script_phrases:
                    similarity = self.text_analyzer.calculate_similarity(text, phrase)
                    if similarity > best_match[1]:
                        best_match = (phrase, similarity)
                
                if best_match[1] > 0.4:  # Umbral de coincidencia
                    matched_phrases.append(best_match[0])
                    total_similarity += best_match[1]
                    count += 1
                    
                    # Clasificar como positivo/negativo
                    sentiment = self.text_analyzer.analyze_sentiment(text)
                    if sentiment['label'] == 'POSITIVE':
                        positive_matches += 1
                    else:
                        negative_matches += 1
        
        adherence = total_similarity / count if count > 0 else 0
        return {
            'adherence': adherence,
            'matched_phrases': [str(phrase) for phrase in set(matched_phrases)],  # Convertir a string
            'positive_matches': positive_matches,
            'negative_matches': negative_matches
        }
    
    def generate_metrics(self, transcription, audio_features):
        speakers = self.classify_speakers(transcription)
        
        adherence_results = self.calculate_script_adherence(transcription, speakers)
        
        metrics = {
            'total_duration': audio_features['duration'],
            'speaker_ratio': sum(1 for s in speakers if s == 0) / len(speakers),
            'interruptions': audio_features['interruptions'],
            'script_adherence': adherence_results['adherence'],
            'matched_phrases': adherence_results['matched_phrases'],
            'required_phrases_missing': self._check_required_phrases(transcription, speakers),
            'speaker_changes': self._count_speaker_changes(speakers),
            'seller_segments': sum(1 for s in speakers if s == 0),
            'client_segments': sum(1 for s in speakers if s == 1),
            'positive_adherence': adherence_results['positive_matches'],
            'negative_adherence': adherence_results['negative_matches']
        }
        
        sentiments = [self.text_analyzer.analyze_sentiment(seg['text']) 
                    for seg in transcription['segments']]
        metrics['sentiment_score'] = sum(1 for s in sentiments if s['label'] == 'POSITIVE') / len(sentiments)
        
        phrase_types = [self.text_analyzer.classify_phrase_type(seg['text']) 
                       for seg, speaker in zip(transcription['segments'], speakers) 
                       if speaker == 0]
        for phrase_type in ['saludo', 'oferta', 'objeción', 'cierre', 'pregunta']:
            metrics[f'phrase_{phrase_type}'] = phrase_types.count(phrase_type)
        
        return metrics, speakers
    
    def _check_required_phrases(self, transcription, speakers):
        missing = []
        vendedor_texts = [seg['text'] for seg, speaker in zip(transcription['segments'], speakers) if speaker == 0]
        
        for phrase in self.config.REQUIRED_PHRASES:
            if not any(
                self.text_analyzer.calculate_similarity(
                    re.sub(r'[^\w\s]', '', text.lower().strip()),
                    re.sub(r'[^\w\s]', '', phrase.lower().strip())
                ) > 0.65
                for text in vendedor_texts
            ):
                missing.append(phrase)
        
        return missing
    
    def _count_speaker_changes(self, speakers):
        if len(speakers) < 2:
            return 0
        return sum(1 for i in range(1, len(speakers)) if speakers[i] != speakers[i-1])

class ContinuousLearner:
    def __init__(self, config):
        self.config = config
        self.learned_phrases_file = os.path.join(config.OUTPUT_FOLDER, 'learned_phrases.csv')
        self.learned_phrases = self._load_learned_phrases()
        self.text_analyzer = TextAnalyzer(config)
    
    def _load_learned_phrases(self):
        if os.path.exists(self.learned_phrases_file):
            return pd.read_csv(self.learned_phrases_file)['phrase'].tolist()
        return []
    
    def analyze_new_phrases(self, transcription, speakers):
        new_phrases = []
        for seg, speaker in zip(transcription['segments'], speakers):
            if speaker == 0:
                text = seg['text']
                max_sim = max([self.text_analyzer.calculate_similarity(text, phrase) 
                             for phrase in self.learned_phrases + self.text_analyzer.script_phrases], default=0)
                if max_sim < 0.6:
                    sentiment = self.text_analyzer.analyze_sentiment(text)
                    if sentiment['label'] == 'POSITIVE' and sentiment['score'] > 0.8:
                        new_phrases.append(text)
        
        if new_phrases:
            new_df = pd.DataFrame({'phrase': new_phrases, 'date_added': datetime.now().strftime('%Y-%m-%d')})
            if os.path.exists(self.learned_phrases_file):
                existing_df = pd.read_csv(self.learned_phrases_file)
                updated_df = pd.concat([existing_df, new_df]).drop_duplicates()
            else:
                updated_df = new_df
            
            os.makedirs(self.config.OUTPUT_FOLDER, exist_ok=True)
            updated_df.to_csv(self.learned_phrases_file, index=False)
            self.learned_phrases = updated_df['phrase'].tolist()
        
        return new_phrases

class TranscriptAnalyzer:
    def __init__(self, config):
        self.config = config
        self.transcriptions_folder = os.path.join(config.OUTPUT_FOLDER, 'transcriptions')
        self.script_phrases = self._load_script_phrases()
        self.positive_phrases = ["gracias", "excelente", "perfecto", "beneficio", "oferta"]
        self.negative_phrases = ["problema", "error", "queja", "reclamo", "no puedo"]
    
    def _load_script_phrases(self):
        """Carga las frases del guión (mismo método que en TextAnalyzer)"""
        if os.path.exists(self.config.SCRIPT_FILE):
            with open(self.config.SCRIPT_FILE, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
        return []
    
    def _load_transcription(self, audio_filename):
        base_name = os.path.splitext(audio_filename)[0]
        transcription_path = os.path.join(self.transcriptions_folder, f"{base_name}.json")
        
        if not os.path.exists(transcription_path):
            return None

        try:
            with open(transcription_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Validación de la estructura esperada
                if not isinstance(data, dict) or 'segments' not in data:
                    st.warning(f"Estructura inválida en {transcription_path}")
                    return None
                    
                return data
                
        except json.JSONDecodeError:
            st.warning(f"Archivo JSON corrupto: {transcription_path}")
            return None
        except Exception as e:
            st.warning(f"Error leyendo {transcription_path}: {str(e)}")
            return None
    
    def _estimate_speakers(self, transcription):
        """Estima speakers si no están en los datos (para compatibilidad)"""
        text_analyzer = TextAnalyzer(self.config)
        return [text_analyzer.classify_role(seg['text']) for seg in transcription['segments']]
    
    def align_with_script(self, transcription, speaker_tags):
        aligned_results = []
        
        for segment, speaker in zip(transcription['segments'], speaker_tags):
            if speaker == 0:  # Solo segmentos del vendedor
                segment_text = segment['text'].strip()
                best_match = None
                max_similarity = 0
                
                for phrase in self.script_phrases:
                    similarity = self._calculate_similarity(segment_text, phrase)
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_match = phrase
                
                # Clasificar como positiva/negativa
                phrase_type = "neutral"
                if best_match:
                    if any(p in segment_text.lower() for p in self.positive_phrases):
                        phrase_type = "positive"
                    elif any(p in segment_text.lower() for p in self.negative_phrases):
                        phrase_type = "negative"
                
                aligned_results.append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'transcribed': segment_text,
                    'script_phrase': best_match if max_similarity > 0.6 else None,
                    'similarity': max_similarity,
                    'type': 'match' if max_similarity > 0.6 else 'extra',
                    'phrase_type': phrase_type
                })
        
        return aligned_results
    
    def _calculate_similarity(self, text1, text2):
        model = SentenceTransformer(self.config.SIMILARITY_MODEL)
        embedding1 = model.encode(text1, convert_to_tensor=True)
        embedding2 = model.encode(text2, convert_to_tensor=True)
        return util.pytorch_cos_sim(embedding1, embedding2).item()
    
    def get_word_frequency(self, aligned_results):
        """Analiza frecuencia de palabras y las clasifica"""
        all_text = ' '.join([r['transcribed'] for r in aligned_results]).lower()
        words = re.findall(r'\w+', all_text)
        
        if not words:
            return pd.DataFrame()
        
        word_freq = pd.Series(words).value_counts().head(20)
        
        # Clasificar palabras
        word_types = []
        for word in word_freq.index:
            if word in self.positive_phrases:
                word_types.append("Positiva")
            elif word in self.negative_phrases:
                word_types.append("Negativa")
            else:
                word_types.append("Neutral")
        
        return pd.DataFrame({
            'Palabra': word_freq.index,
            'Frecuencia': word_freq.values,
            'Tipo': word_types
        })

class Dashboard:
    def __init__(self, config):
        self.config = config
        self.sheets_manager = GoogleSheetsManager(config)
        self.positive_phrases = ["gracias", "excelente", "perfecto", "beneficio", "oferta"]
        self.negative_phrases = ["problema", "error", "queja", "reclamo", "no puedo"]
    
    def _format_number(self, value, is_percent=False, decimals=2):
        """Formatea números con separadores de miles y decimales"""
        if pd.isna(value):
            return "-"
        
        if is_percent:
            return f"{value:,.{decimals}f}%"
        else:
            return f"{value:,.{decimals}f}"
    
    def _apply_color(self, value, threshold=60):
        """Aplica color rojo/verde según el valor"""
        if value < threshold:
            return f"color: red; font-weight: bold;"
        return f"color: green; font-weight: bold;"
    
    def _get_time_filters(self, df):
        """Crea filtros de tiempo y devuelve DataFrame filtrado"""
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        df['Mes'] = df['Fecha'].dt.to_period('M')
        df['Semana'] = df['Fecha'].dt.to_period('W')
        df['Dia'] = df['Fecha'].dt.date
        
        st.sidebar.header("🔍 Filtros Temporales")
        time_filter = st.sidebar.radio(
            "Filtrar por:", 
            ['Día', 'Semana', 'Mes', 'Todo'], 
            horizontal=True
        )
        
        if time_filter == 'Día':
            unique_days = sorted(df['Dia'].unique(), reverse=True)
            selected_day = st.sidebar.selectbox("Seleccionar día:", unique_days)
            return df[df['Dia'] == selected_day]
        
        elif time_filter == 'Semana':
            unique_weeks = sorted(df['Semana'].unique(), reverse=True)
            selected_week = st.sidebar.selectbox("Seleccionar semana:", unique_weeks)
            return df[df['Semana'] == selected_week]
        
        elif time_filter == 'Mes':
            unique_months = sorted(df['Mes'].unique(), reverse=True)
            selected_month = st.sidebar.selectbox("Seleccionar mes:", unique_months)
            return df[df['Mes'] == selected_month]
        
        return df
    
    def _explain_metrics(self):
        """Explica las métricas clave"""
        with st.expander("📊 Explicación de Métricas"):
            st.markdown("""
            **Apego al Guión (%):**  
            Porcentaje de coincidencia entre lo dicho por el vendedor y el guión oficial.  
            * >75%: Excelente | 60-75%: Bueno | <60%: Necesita mejora*

            **Apego Positivo:**  
            Frases que coinciden con el guión y tienen sentimiento positivo.

            **Apego Negativo:**  
            Frases que coinciden con el guión pero tienen sentimiento negativo.

            **Sentimiento (%):**  
            Porcentaje de frases con tono positivo detectado.  
            * >70%: Muy positivo | 50-70%: Neutral | <50%: Negativo*

            **Interrupciones:**  
            Número de veces que el cliente interrumpe al vendedor.  
            * >5: Muchas interrupciones | 3-5: Normal | <3: Buen flujo*

            **Frases Coincidentes:**  
            Listado de frases del guión que fueron identificadas en la llamada.

            **Frases Faltantes:**  
            Frases clave del guión que no se mencionaron en la llamada.
            """)
    
    def _create_metric_card(self, title, value, delta=None, delta_color="normal"):
        """Crea una tarjeta de métrica con formato profesional"""
        st.metric(
            label=title,
            value=value,
            delta=delta,
            delta_color=delta_color
        )
    
    def _create_bar_chart(self, data, x, y, title, x_label=None, y_label=None, color=None, format_yaxis=False):
        """Crea un gráfico de barras profesional"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if color:
            sns.barplot(x=x, y=y, data=data, ax=ax, color=color)
        else:
            sns.barplot(x=x, y=y, data=data, ax=ax)
        
        ax.set_title(title, fontweight='bold', pad=20)
        if x_label:
            ax.set_xlabel(x_label)
        if y_label:
            ax.set_ylabel(y_label)
        
        # Formatear ejes
        if format_yaxis:
            ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
        
        # Rotar etiquetas si es necesario
        if len(data[x]) > 5:
            plt.xticks(rotation=45, ha='right')
        
        # Añadir valores en las barras
        for p in ax.patches:
            ax.annotate(
                f"{p.get_height():,.1f}", 
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), 
                textcoords='offset points'
            )
        
        plt.tight_layout()
        st.pyplot(fig)
    
    def _create_line_chart(self, data, x, y, title, x_label=None, y_label=None, format_yaxis=False):
        """Crea un gráfico de líneas profesional"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.lineplot(x=x, y=y, data=data, ax=ax, marker='o', linewidth=2.5)
        
        ax.set_title(title, fontweight='bold', pad=20)
        if x_label:
            ax.set_xlabel(x_label)
        if y_label:
            ax.set_ylabel(y_label)
        
        # Formatear ejes
        if format_yaxis:
            ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
        
        # Añadir valores en los puntos
        for line in ax.lines:
            for x_val, y_val in zip(line.get_xdata(), line.get_ydata()):
                ax.annotate(
                    f"{y_val:,.1f}", 
                    (x_val, y_val),
                    textcoords="offset points", xytext=(0,10), 
                    ha='center'
                )
        
        plt.tight_layout()
        st.pyplot(fig)
    
    def display_executive_summary(self, df):
        """Vista General para el supervisor"""
        st.title("📊 Resumen Ejecutivo")
        
        # Filtros
        df_filtered = self._get_time_filters(df)
        
        # KPI Cards
        total_calls = len(df_filtered)
        avg_adherence = df_filtered['Apego (%)'].mean()
        avg_sentiment = df_filtered['Sentimiento (%)'].mean()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Audios Procesados", f"{total_calls:,}")
        col2.metric("Apego Promedio", f"{avg_adherence:,.1f}%", 
                  delta=f"{(avg_adherence - df['Apego (%)'].mean()):.1f}% vs total")
        col3.metric("Sentimiento Promedio", f"{avg_sentiment:,.1f}%", 
                  delta=f"{(avg_sentiment - df['Sentimiento (%)'].mean()):.1f}% vs total")
        
        # Ranking de vendedores
        st.subheader("🏆 Ranking por Apego al Guión")
        seller_ranking = df_filtered.groupby('Vendedor')['Apego (%)'].mean().sort_values(ascending=False)
        
        # Convertir a DataFrame y aplicar estilo
        ranking_df = seller_ranking.reset_index(name='Apego (%)')
        ranking_df['Apego (%)'] = ranking_df['Apego (%)'].apply(lambda x: f"{x:,.1f}%")
        
        st.dataframe(
            ranking_df.style.apply(
                lambda x: [self._apply_color(float(v.replace('%', ''))) if i == 1 else '' for i, v in enumerate(x)],
                subset=['Apego (%)']
            ),
            use_container_width=True,
            column_config={
                "Vendedor": st.column_config.TextColumn("Vendedor"),
                "Apego (%)": st.column_config.TextColumn("Apego (%)", help="Porcentaje de apego al guión")
            }
        )
        
        # Alertas
        st.subheader("🚨 Alertas")
        low_performance = df_filtered[df_filtered['Apego (%)'] < 60]['Vendedor'].value_counts()
        if not low_performance.empty:
            st.warning(f"Vendedores con bajo apego (<60%): {', '.join(low_performance.index)}")
        
        missing_phrases = df_filtered[df_filtered['Frases Faltantes'] != 'Ninguna']['Frases Faltantes'].astype(str).value_counts()
        if not missing_phrases.empty:
            st.warning(f"Frases más omitidas: {', '.join(missing_phrases.head(3).index)}")
        
        # Gráficos
        tab1, tab2, tab3 = st.tabs(["Distribución", "Tendencia", "Análisis de Palabras"])
        
        with tab1:
            fig, ax = plt.subplots(1, 2, figsize=(14, 6))
            
            # Gráfico de caja
            sns.boxplot(y='Vendedor', x='Apego (%)', data=df_filtered, ax=ax[0], palette='viridis')
            ax[0].set_title("Distribución de Apego al Guión", fontweight='bold')
            ax[0].set_xlabel("Apego (%)")
            ax[0].set_ylabel("")
            ax[0].xaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
            
            # Gráfico de conteo
            seller_counts = df_filtered['Vendedor'].value_counts().reset_index()
            seller_counts.columns = ['Vendedor', 'Conteo']
            sns.barplot(y='Vendedor', x='Conteo', data=seller_counts, ax=ax[1], palette='viridis')
            ax[1].set_title("Número de Llamadas por Vendedor", fontweight='bold')
            ax[1].set_xlabel("Número de Llamadas")
            ax[1].set_ylabel("")
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with tab2:
            df_trend = df_filtered.set_index('Fecha').sort_index()
            df_trend_resampled = df_trend[['Apego (%)', 'Sentimiento (%)']].resample('W').mean()
            
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.lineplot(data=df_trend_resampled, dashes=False, markers=True, linewidth=2.5)
            ax.set_title("Tendencia Semanal de Apego y Sentimiento", fontweight='bold')
            ax.set_ylabel("Porcentaje")
            ax.set_xlabel("Fecha")
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
            ax.legend(title='Métrica', labels=['Apego (%)', 'Sentimiento (%)'])
            
            # Añadir valores en los puntos
            for line in ax.lines:
                for x_val, y_val in zip(line.get_xdata(), line.get_ydata()):
                    if not np.isnan(y_val):
                        ax.annotate(
                            f"{y_val:.1f}%", 
                            (x_val, y_val),
                            textcoords="offset points", xytext=(0,10), 
                            ha='center'
                        )
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with tab3:
            transcript_analyzer = TranscriptAnalyzer(self.config)
            aligned_results_total = []
            for audio_name in df_filtered['Audio']:
                transcription = transcript_analyzer._load_transcription(audio_name)
                if transcription and 'speaker_tags' in transcription:
                    alignment = transcript_analyzer.align_with_script(transcription, transcription['speaker_tags'])
                    aligned_results_total.extend(alignment)
            word_df = transcript_analyzer.get_word_frequency(aligned_results_total)
            if not word_df.empty:
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.barplot(
                    y='Palabra', x='Frecuencia', hue='Tipo', 
                    data=word_df, 
                    palette={'Positiva': '#4CAF50', 'Negativa': '#F44336', 'Neutral': '#2196F3'},
                    ax=ax
                )
                ax.set_title("Palabras Más Utilizadas", fontweight='bold')
                ax.set_xlabel("Frecuencia")
                ax.set_ylabel("")
                
                # Añadir valores en las barras
                for p in ax.patches:
                    width = p.get_width()
                    ax.annotate(
                        f"{width:,.0f}", 
                        (width, p.get_y() + p.get_height() / 2.),
                        ha='left', va='center', xytext=(5, 0), 
                        textcoords='offset points'
                    )
                
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("No hay datos suficientes para análisis de palabras")
        
        self._explain_metrics()
    
    def display_seller_view(self, df):
        """Vista detallada por vendedor"""
        st.title("👤 Vista por Vendedor")
        
        selected_seller = st.selectbox(
            "Seleccionar Vendedor", 
            options=sorted(df['Vendedor'].unique()))
        
        df_seller = df[df['Vendedor'] == selected_seller]
        df_filtered = self._get_time_filters(df_seller)
        
        # Resumen individual
        st.subheader(f"📈 Rendimiento de {selected_seller}")
        
        latest_call = df_filtered.sort_values('Fecha', ascending=False).iloc[0]
        avg_adherence = df_filtered['Apego (%)'].mean()
        trend = df_filtered.sort_values('Fecha')['Apego (%)'].pct_change().iloc[-1] * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Último Apego", f"{latest_call['Apego (%)']:,.1f}%", 
                   delta=f"{trend:.1f}% vs anterior", delta_color="inverse")
        col2.metric("Apego Promedio", f"{avg_adherence:,.1f}%")
        col3.metric("Llamadas Analizadas", f"{len(df_filtered):,}")
        
        # Gráfico de evolución
        st.subheader("📅 Evolución Temporal")
        df_trend = df_filtered.set_index('Fecha').sort_index()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(
            data=df_trend, 
            x=df_trend.index, 
            y='Apego (%)', 
            marker='o', 
            linewidth=2.5,
            color='#4CAF50'
        )
        ax.set_title(f"Evolución de Apego al Guión - {selected_seller}", fontweight='bold')
        ax.set_ylabel("Apego (%)")
        ax.set_xlabel("Fecha")
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
        
        # Añadir valores en los puntos
        for x_val, y_val in zip(df_trend.index, df_trend['Apego (%)']):
            ax.annotate(
                f"{y_val:.1f}%", 
                (x_val, y_val),
                textcoords="offset points", xytext=(0,10), 
                ha='center'
            )
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Análisis de frases
        st.subheader("📝 Análisis de Frases")
        
        tab1, tab2, tab3 = st.tabs(["Faltantes", "Frecuentes", "Detalle por Llamada"])
        
        with tab1:
            missing_phrases = df_filtered[df_filtered['Frases Faltantes'] != 'Ninguna']['Frases Faltantes']
            if not missing_phrases.empty:
                st.write("Frases que faltan con frecuencia:")
                missing_counts = missing_phrases.value_counts().reset_index()
                missing_counts.columns = ['Frase', 'Veces omitidas']
                st.dataframe(missing_counts)
            else:
                st.success("✅ No se omiten frases clave frecuentemente")
        
        with tab2:
            transcript_analyzer = TranscriptAnalyzer(self.config)
            word_df = transcript_analyzer.get_word_frequency(df_filtered)
            if not word_df.empty:
                st.dataframe(word_df.style.format({'Frecuencia': '{:,.0f}'}))
            else:
                st.info("No hay datos suficientes para análisis de palabras")
        
        with tab3:
            selected_call = st.selectbox(
                "Seleccionar llamada", 
                df_filtered.sort_values('Fecha', ascending=False)['Audio'])
            
            call_data = df_filtered[df_filtered['Audio'] == selected_call].iloc[0]
            
            st.markdown(f"**Fecha:** {call_data['Fecha'].strftime('%Y-%m-%d %H:%M')}")
            st.markdown(f"**Duración:** {call_data['Duración (s)']:,.0f} segundos")
            
            cols = st.columns(2)
            cols[0].metric("Apego", f"{call_data['Apego (%)']:,.1f}%", 
                          delta_color="off", 
                          help="Porcentaje de coincidencia con el guión")
            cols[1].metric("Sentimiento", f"{call_data['Sentimiento (%)']:,.1f}%", 
                          delta_color="off", 
                          help="Porcentaje de frases positivas")
            
            st.markdown("**Frases coincidentes:**")
            if call_data['Frases Coincidentes'] != 'Ninguna':
                st.write(call_data['Frases Coincidentes'])
            else:
                st.warning("No se encontraron frases coincidentes")
            
            st.markdown("**Frases faltantes:**")
            if call_data['Frases Faltantes'] != 'Ninguna':
                st.write(call_data['Frases Faltantes'])
            else:
                st.success("Todas las frases clave fueron mencionadas")
    
    def display_call_review(self, df):
        """Revisión detallada de una llamada específica"""
        st.title("🎧 Revisión de Llamada")
        
        selected_seller = st.selectbox(
            "Seleccionar Vendedor", 
            options=sorted(df['Vendedor'].unique()))
        
        df_seller = df[df['Vendedor'] == selected_seller]
        selected_call = st.selectbox(
            "Seleccionar Llamada", 
            options=df_seller.sort_values('Fecha', ascending=False)['Audio'])
        
        call_data = df_seller[df_seller['Audio'] == selected_call].iloc[0]
        
        # Reproducción de audio
        st.audio(call_data['Enlace Audio'], format='audio/mp3')
        
        # Cargar transcripción
        transcript_analyzer = TranscriptAnalyzer(self.config)
        transcription = transcript_analyzer._load_transcription(selected_call)
        
        if not transcription:
            st.error("No se encontró la transcripción para esta llamada")
            return
        
        # Usar los tags guardados
        speakers = transcription.get('speaker_tags', [])
        
        # Visualización mejorada de speakers
        st.subheader("🗣️ Distribución de Hablantes")
        
        # Calcular estadísticas
        total_segments = len(speakers)
        seller_segments = sum(1 for s in speakers if s == 0)
        client_segments = total_segments - seller_segments
        
        col1, col2 = st.columns(2)
        col1.metric("Segmentos Vendedor", f"{seller_segments:,}", f"{seller_segments/total_segments:.0%}")
        col2.metric("Segmentos Cliente", f"{client_segments:,}", f"{client_segments/total_segments:.0%}")
        
        # Gráfico de distribución
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=['Vendedor', 'Cliente'], y=[seller_segments, client_segments], 
                    palette=['#4CAF50', '#2196F3'], ax=ax)
        ax.set_title("Distribución de Hablantes", fontweight='bold')
        ax.set_ylabel("Número de Segmentos")
        
        # Añadir valores en las barras
        for p in ax.patches:
            ax.annotate(
                f"{p.get_height():,.0f}", 
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), 
                textcoords='offset points'
            )
        
        st.pyplot(fig)
        
        # Alinear con el guión
        aligned_results = transcript_analyzer.align_with_script(transcription, speakers)
        
        # Mostrar resultados
        st.subheader("📄 Alineación con el Guión")
        
        # Filtros para la visualización
        col1, col2 = st.columns(2)
        show_matches = col1.checkbox("Mostrar coincidencias", value=True)
        show_extras = col2.checkbox("Mostrar frases adicionales", value=True)
        min_similarity = st.slider("Filtrar por similitud mínima", 0.0, 1.0, 0.6)
        
        # Mostrar la transcripción alineada
        for result in aligned_results:
            if ((result['type'] == 'match' and show_matches and result['similarity'] >= min_similarity) or
                (result['type'] == 'extra' and show_extras)):
                
                # Formatear según el tipo
                if result['type'] == 'match':
                    similarity_color = f"hsl({120 * result['similarity']}, 100%, 40%)"
                    border_color = similarity_color
                    icon = "✅" if result['similarity'] > 0.8 else "🔹"
                else:
                    border_color = "orange"
                    icon = "🔸"
                
                # Mostrar cada segmento como una tarjeta
                with st.container():
                    st.markdown(
                        f"""
                        <div style="
                            border-left: 4px solid {border_color};
                            padding: 0.5rem;
                            margin: 0.5rem 0;
                            background-color: #f8f9fa;
                            border-radius: 0.25rem;
                        ">
                            <div style="display: flex; justify-content: space-between;">
                                <small>{icon} [{(result['start']//60):02.0f}:{(result['start']%60):02.0f}-{(result['end']//60):02.0f}:{(result['end']%60):02.0f}]</small>
                                <small>Similitud: <strong>{result['similarity']:.0%}</strong></small>
                            </div>
                            <p><strong>Dicho:</strong> {result['transcribed']}</p>
                            {f'<p><strong>Guion:</strong> {result["script_phrase"]}</p>' if result["script_phrase"] else ''}
                            {f'<p><strong>Tipo:</strong> {result["phrase_type"].capitalize()}</p>' if "phrase_type" in result else ''}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        
        # Estadísticas de alineación
        total_segments = len(aligned_results)
        matched_segments = sum(1 for r in aligned_results if r['type'] == 'match')
        extra_segments = total_segments - matched_segments
        
        col1, col2 = st.columns(2)
        col1.metric("Coincidencias con guión", f"{matched_segments:,} ({matched_segments/total_segments:.0%})")
        col2.metric("Frases adicionales", f"{extra_segments:,} ({extra_segments/total_segments:.0%})")
        
        # Comentarios del supervisor
        st.subheader("💬 Comentarios y Ajustes")
        comment = st.text_area("Agregar comentario para el vendedor:")
        
        if st.button("💾 Guardar Comentario"):
            # Aquí iría la lógica para guardar los comentarios
            st.success("Comentario guardado exitosamente")
        
        # Opción para agregar nuevas frases al guión
        if st.checkbox("➕ Proponer nueva frase para el guión"):
            new_phrase = st.text_input("Frase propuesta:")
            phrase_type = st.radio("Tipo de frase:", ["Positiva", "Neutral", "Negativa"])
            
            if st.button("Enviar para aprobación"):
                # Lógica para guardar la nueva frase propuesta
                st.success("Frase enviada para aprobación del supervisor")
    
    def run_dashboard(self, df):
        """Ejecuta el dashboard completo"""
        st.sidebar.title("Navegación")
        app_mode = st.sidebar.radio(
            "Seleccionar Vista:",
            ["Resumen Ejecutivo", "Vista por Vendedor", "Revisión de Llamada"]
        )
        
        if app_mode == "Resumen Ejecutivo":
            self.display_executive_summary(df)
        elif app_mode == "Vista por Vendedor":
            self.display_seller_view(df)
        elif app_mode == "Revisión de Llamada":
            self.display_call_review(df)

class CallMonitoringSystem:
    def __init__(self):
        self.config = Config()
        self._setup_directories()
        self.drive_connector = DriveConnector()
        self.audio_processor = AudioProcessor(self.config)
        self.text_analyzer = TextAnalyzer(self.config)
        self.metrics_generator = MetricsGenerator(self.config)
        self.continuous_learner = ContinuousLearner(self.config)
        self.dashboard = Dashboard(self.config)
        self.sheets_manager = GoogleSheetsManager(self.config)
    
    def _setup_directories(self):
        os.makedirs(self.config.OUTPUT_FOLDER, exist_ok=True)
        os.makedirs(self.config.AUDIO_FOLDER, exist_ok=True)
        os.makedirs(self.config.MODEL_FOLDER, exist_ok=True)
    
    def process_new_audios(self):
        """Procesa nuevos audios desde Google Drive y guarda los resultados en Google Sheets"""
        st.header("🔄 Procesando Audios")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Inicializar conexión con Google Sheets
            sheet = self.sheets_manager.initialize_sheet()
            
            # Obtener listado de vendedores y audios desde Drive
            sellers = self.drive_connector.get_sellers_and_audios(self.config.DRIVE_FOLDER_ID)
            
            # Validar estructura de datos recibida
            if not isinstance(sellers, dict):
                st.error("Error: Formato inválido de datos de vendedores. Se esperaba un diccionario.")
                return

            total_audios = sum(len(seller_info['audios']) for seller_info in sellers.values() if 'audios' in seller_info)
            processed_count = 0
            success_count = 0
            
            # Procesar cada vendedor y sus audios
            for seller_name, seller_info in sellers.items():
                # Validar estructura por vendedor
                if not isinstance(seller_info, dict) or 'audios' not in seller_info:
                    st.warning(f"Estructura inválida para el vendedor {seller_name}. Se omite.")
                    continue

                for audio in seller_info['audios']:
                    processed_count += 1
                    progress = processed_count / total_audios
                    progress_bar.progress(progress)
                    
                    try:
                        # Validar estructura del audio
                        if not isinstance(audio, dict) or 'id' not in audio or 'title' not in audio:
                            st.warning(f"Audio inválido en {seller_name}. Se omite.")
                            continue

                        status_text.text(f"Procesando {processed_count}/{total_audios}: {audio['title']}")
                        
                        # 1. Descargar audio
                        audio_path = self.drive_connector.download_audio(audio['id'], audio['title'])
                        if not os.path.exists(audio_path):
                            st.warning(f"No se pudo descargar el audio: {audio['title']}")
                            continue

                        # 2. Transcribir audio
                        transcription = self.audio_processor.transcribe_audio(audio_path)
                        if not transcription or 'segments' not in transcription:
                            st.warning(f"Transcripción fallida para: {audio['title']}")
                            continue

                        # 3. Analizar características del audio
                        audio_features = self.audio_processor.analyze_audio_features(audio_path)
                        
                        # 4. Generar métricas
                        metrics, speakers = self.metrics_generator.generate_metrics(transcription, audio_features)
                        
                        # 5. Aprendizaje continuo (nuevas frases)
                        new_phrases = self.continuous_learner.analyze_new_phrases(transcription, speakers)
                        
                        # 6. Guardar transcripción completa
                        transcription['speaker_tags'] = speakers
                        audio_filename = os.path.splitext(audio['title'])[0]
                        transcription_path = os.path.join(
                            self.audio_processor.transcriptions_folder, 
                            f"{audio_filename}.json"
                        )
                        
                        with open(transcription_path, 'w', encoding='utf-8') as f:
                            json.dump(transcription, f, ensure_ascii=False, indent=2)

                        # 7. Preparar datos para Google Sheets
                        matched_phrases = ', '.join(str(phrase) for phrase in metrics['matched_phrases']) if metrics['matched_phrases'] else 'Ninguna'
                        missing_phrases = ', '.join(str(phrase) for phrase in metrics['required_phrases_missing']) if metrics['required_phrases_missing'] else 'Ninguna'
                        new_phrases_str = ', '.join(new_phrases) if new_phrases else 'Ninguna'

                        # 8. Guardar en Google Sheets
                        sheet.append_row([
                            seller_name,
                            audio['title'],
                            audio['url'],
                            audio['createdDate'],
                            round(metrics['script_adherence'] * 100, 2),
                            round(metrics['total_duration'], 2),
                            round(metrics['sentiment_score'] * 100, 2),
                            new_phrases_str,
                            metrics['interruptions'],
                            matched_phrases,
                            missing_phrases,
                            round(metrics['total_duration'], 2),
                            metrics['speaker_changes'],
                            metrics['seller_segments'],
                            metrics['client_segments']
                        ])
                        
                        success_count += 1

                    except whisper.WhisperException as e:
                        st.error(f"Error en Whisper al procesar {audio.get('title', 'audio desconocido')}: {str(e)}")
                    except librosa.LibrosaError as e:
                        st.error(f"Error en análisis de audio {audio.get('title', 'audio desconocido')}: {str(e)}")
                    except gspread.exceptions.APIError as e:
                        st.error(f"Error al guardar en Google Sheets: {str(e)}")
                        time.sleep(10)  # Esperar antes de reintentar
                    except Exception as e:
                        st.error(f"Error inesperado procesando {audio.get('title', 'audio desconocido')}: {str(e)}")
                        st.error(traceback.format_exc())  # Mostrar traceback completo

            # Resultado final
            if success_count > 0:
                st.success(f"✅ Procesamiento completo: {success_count}/{total_audios} audios procesados exitosamente")
            else:
                st.warning("⚠️ No se pudo procesar ningún audio. Verifica los logs para más detalles")
                
            # Forzar actualización del dashboard
            st.rerun()

        except Exception as e:
            st.error(f"Error crítico en process_new_audios: {str(e)}")
            st.error(traceback.format_exc())
    
    def run_streamlit_dashboard(self):
        st.title("📞 Sistema de Monitoreo Apego de Llamadas")
        
        # Verificación del guión
        if st.checkbox("🔍 Verificar contenido del guión"):
            try:
                with open(self.config.SCRIPT_FILE, 'r', encoding='utf-8') as f:
                    guion = f.read()
                    st.text_area("Contenido actual del guión oficial:", guion, height=200)
                    st.write(f"Total de frases en guión: {len(self.text_analyzer.script_phrases)}")
            except Exception as e:
                st.error(f"No se pudo cargar el guión: {str(e)}")
        
        with st.expander("🔧 Procesar nuevos audios", expanded=False):
            if st.button("🔄 Procesar audios ahora"):
                with st.spinner("Procesando y guardando en Google Sheets..."):
                    self.process_new_audios()
                st.rerun()
        
        try:
            sheet = self.sheets_manager.get_sheet()
            records = sheet.get_all_records()

            if not records:
                st.warning("No hay datos en Google Sheets. Procesa algunos audios primero.")
                return

            # Convertir a DataFrame
            df = pd.DataFrame(records)

            # Validar columnas obligatorias
            expected_cols = [
                'Vendedor', 'Audio', 'Enlace Audio', 'Fecha', 'Apego (%)', 'Duración',
                'Sentimiento (%)', 'Nuevas Frases Aprendidas', 'Interrupciones',
                'Frases Coincidentes', 'Frases Faltantes', 'Duración (s)',
                'Cambios de Hablante', 'Segmentos Vendedor', 'Segmentos Cliente'
            ]
            missing_cols = [col for col in expected_cols if col not in df.columns]
            if missing_cols:
                st.error(f"⚠️ Faltan columnas requeridas: {missing_cols}")
                st.stop()

            # Forzar conversiones de columnas numéricas
            for col in ['Apego (%)', 'Sentimiento (%)', 'Duración', 'Interrupciones', 'Duración (s)', 
                        'Cambios de Hablante', 'Segmentos Vendedor', 'Segmentos Cliente']:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            # Forzar columnas de frases a string
            for col in ['Frases Coincidentes', 'Frases Faltantes', 'Nuevas Frases Aprendidas']:
                df[col] = df[col].astype(str)

            # Reemplazar valores NaN en numéricos con 0
            numeric_cols = ['Apego (%)', 'Sentimiento (%)', 'Interrupciones',
                            'Duración (s)', 'Cambios de Hablante', 'Segmentos Vendedor',
                            'Segmentos Cliente']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            # Ejecutar el dashboard
            self.dashboard.run_dashboard(df)

        except Exception as e:
            st.error(f"Error cargando datos: {str(e)}")

if __name__ == "__main__":
    system = CallMonitoringSystem()
    system.run_streamlit_dashboard()  # ✅ Streamlit visualización, no procesamiento automático
