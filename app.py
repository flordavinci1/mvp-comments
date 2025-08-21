import streamlit as st
import pandas as pd
from googleapient.discovery import build
from googleapient.errors import HttpError
from urllib.parse import urlparse, parse_qs
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time

# Se descarga el léxico de VADER para el análisis de sentimientos
# Esto solo se necesita hacer una vez
try:
    _ = SentimentIntensityAnalyzer()
except LookupError:
    nltk.download('vader_lexicon')

# --- Configuración de la aplicación Streamlit ---
st.set_page_config(page_title="StreamLive Analytics", page_icon="📺")

st.title("StreamLive Analytics: Analizador de Livestream de YouTube")
st.markdown("""
    Esta herramienta te ayuda a analizar en tiempo real los comentarios de un livestream de YouTube.
    **Recuerda que debes tener una clave de API de YouTube válida.**
""")

# --- Obtención de la clave de API ---
st.info("Ingresa tu clave de API de YouTube Data v3. Puedes obtenerla en la Consola de Desarrolladores de Google.")
api_key = st.text_input("Clave de API de YouTube", type="password")

# --- Funciones de la aplicación ---

def get_video_id(url):
    """Extrae el ID del video de una URL de YouTube."""
    if url is None:
        return None
    
    match = re.search(r'(?:v=|\/live\/|\.be\/)([a-zA-Z0-9_-]{11})', url)
    if match:
        return match.group(1)
    
    # Intenta extraer de un formato de URL más corto (youtube.com/live/ID)
    match_short = re.search(r'youtube\.com\/live\/([a-zA-Z0-9_-]{11})', url)
    if match_short:
        return match_short.group(1)
    
    return None

def fetch_live_chat_comments(youtube_service, live_chat_id):
    """
    Obtiene todos los comentarios del chat en vivo de un livestream de YouTube.
    Maneja la paginación y espera un intervalo fijo entre solicitudes.
    """
    comments = []
    page_token = None
    
    while True:
        try:
            request = youtube_service.liveChatMessages().list(
                liveChatId=live_chat_id,
                part='authorDetails,snippet',
                pageToken=page_token
            )
            response = request.execute()
            
            for item in response.get('items', []):
                comment_text = item['snippet']['displayMessage']
                comment_time = item['snippet']['publishedAt']
                comments.append({
                    "text": comment_text,
                    "timestamp": comment_time
                })
            
            # Obtiene el tiempo de espera recomendado de la respuesta de la API (en milisegundos)
            polling_interval = response.get('pollingIntervalMillis', 10000) / 1000 # Convierte a segundos
            
            # Revisa si hay más páginas de comentarios
            page_token = response.get('nextPageToken')
            if not page_token:
                break
            
            # Espera el tiempo recomendado por la API antes de la próxima solicitud
            time.sleep(polling_interval)
        
        except HttpError as e:
            if e.resp.status == 400:
                # 400 Bad Request: puede ser por un chat en vivo que ya no está activo
                st.error("Error al obtener los comentarios: El livestream puede haber terminado o no tener chat habilitado.")
            else:
                st.error(f"Ocurrió un error con la API de YouTube: {e}")
            break
        
        except Exception as e:
            st.error(f"Ocurrió un error inesperado: {e}")
            break
            
    return comments

def perform_sentiment_analysis(df):
    """Realiza un análisis de sentimiento básico usando VADER."""
    if df.empty:
        return {}
    
    analyzer = SentimentIntensityAnalyzer()
    df['sentiment'] = df['text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    
    # Clasificar comentarios como Positivo, Negativo o Neutro
    def categorize_sentiment(score):
        if score >= 0.05:
            return 'Positivo'
        elif score <= -0.05:
            return 'Negativo'
        else:
            return 'Neutral'
    
    df['sentiment_category'] = df['sentiment'].apply(categorize_sentiment)
    
    # Contar la cantidad de comentarios por categoría
    sentiment_counts = df['sentiment_category'].value_counts(normalize=True).round(2)
    return sentiment_counts

# --- Interfaz de usuario principal ---
youtube_url = st.text_input("Ingresa la URL del livestream de YouTube")

# Botón para iniciar el análisis
if st.button("Iniciar Análisis"):
    if not api_key:
        st.error("Por favor, ingresa tu clave de API de YouTube para continuar.")
    elif not youtube_url:
        st.error("Por favor, ingresa una URL de YouTube válida.")
    else:
        video_id = get_video_id(youtube_url)
        if not video_id:
            st.error("URL de YouTube inválida. Por favor, asegúrate de que sea un enlace de livestream.")
            st.stop()
        
        # Conectarse a la API de YouTube
        try:
            youtube = build('youtube', 'v3', developerKey=api_key)
        except Exception as e:
            st.error(f"Error al conectar a la API. Verifica tu clave de API. Detalles: {e}")
            st.stop()

        try:
            # Obtener el chat ID del video
            with st.spinner('Obteniendo información del livestream...'):
                video_request = youtube.videos().list(
                    part='liveStreamingDetails',
                    id=video_id
                )
                video_response = video_request.execute()
                
                live_chat_id = None
                if video_response.get('items'):
                    live_chat_details = video_response['items'][0].get('liveStreamingDetails')
                    if live_chat_details:
                        live_chat_id = live_chat_details.get('activeLiveChatId')

            if not live_chat_id:
                st.error("No se pudo encontrar un chat en vivo activo para esta URL. ¿Quizás el stream ha terminado?")
                st.stop()
            
            # Extraer y analizar los comentarios
            with st.spinner(f'Extrayendo comentarios y realizando análisis. Esto puede tomar un momento...'):
                comments_list = fetch_live_chat_comments(youtube, live_chat_id)
                
            if comments_list:
                df_comments = pd.DataFrame(comments_list)
                df_comments['timestamp'] = pd.to_datetime(df_comments['timestamp'])
                
                # --- Visualización de resultados ---
                st.subheader("Resultados del Análisis")
                st.metric(label="Comentarios Totales Analizados", value=len(df_comments))
                
                # Gráfico de comentarios por hora
                df_comments['hour'] = df_comments['timestamp'].dt.hour
                comments_per_hour = df_comments.groupby('hour').size().reset_index(name='count')
                st.bar_chart(comments_per_hour.set_index('hour'))
                
                # Análisis de sentimiento
                sentiment_results = perform_sentiment_analysis(df_comments)
                
                st.subheader("Análisis de Sentimiento Básico")
                if not sentiment_results.empty:
                    # Crear columnas para mostrar métricas de sentimiento
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(label="Positivos", value=f"{sentiment_results.get('Positivo', 0) * 100:.0f}%")
                    with col2:
                        st.metric(label="Neutrales", value=f"{sentiment_results.get('Neutral', 0) * 100:.0f}%")
                    with col3:
                        st.metric(label="Negativos", value=f"{sentiment_results.get('Negativo', 0) * 100:.0f}%")
                    
                    st.dataframe(df_comments[['timestamp', 'text', 'sentiment_category']].tail(50))
                else:
                    st.warning("No hay suficientes comentarios para realizar un análisis de sentimiento.")

        except HttpError as e:
            st.error(f"Error en la API: {e}. Revisa si tu clave de API es válida y si tienes habilitada la 'YouTube Data API v3'.")
        except Exception as e:
            st.error(f"Ocurrió un error inesperado. Por favor, intenta con otra URL. Detalle del error: {e}")
