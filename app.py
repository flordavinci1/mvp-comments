import streamlit as st
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
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

# --- Inicialización del estado de la sesión ---
if 'comments' not in st.session_state:
    st.session_state.comments = []
if 'live_chat_id' not in st.session_state:
    st.session_state.live_chat_id = None
if 'next_page_token' not in st.session_state:
    st.session_state.next_page_token = None
if 'api_key' not in st.session_state:
    st.session_state.api_key = ''
if 'youtube_service' not in st.session_state:
    st.session_state.youtube_service = None

# --- Obtención de la clave de API ---
st.info("Ingresa tu clave de API de YouTube Data v3. Puedes obtenerla en la Consola de Desarrolladores de Google.")
st.session_state.api_key = st.text_input("Clave de API de YouTube", type="password", value=st.session_state.api_key)

# --- Funciones de la aplicación ---
def get_video_id(url):
    """Extrae el ID del video de una URL de YouTube."""
    if url is None:
        return None
    match = re.search(r'(?:v=|\/live\/|\.be\/)([a-zA-Z0-9_-]{11})', url)
    if match:
        return match.group(1)
    match_short = re.search(r'youtube\.com\/live\/([a-zA-Z0-9_-]{11})', url)
    if match_short:
        return match_short.group(1)
    return None

def fetch_live_chat_comments(youtube_service, live_chat_id, page_token):
    """
    Obtiene un conjunto de comentarios del chat en vivo de un livestream de YouTube.
    """
    comments = []
    next_page_token = None
    polling_interval = 10000 / 1000 # Valor por defecto de la API si no se especifica.
    
    try:
        request = youtube_service.liveChatMessages().list(
            liveChatId=live_chat_id,
            part='authorDetails,snippet',
            pageToken=page_token
        )
        response = request.execute()
        
        for item in response.get('items', []):
            comments.append({
                "text": item['snippet']['displayMessage'],
                "timestamp": item['snippet']['publishedAt']
            })
        
        next_page_token = response.get('nextPageToken')
        polling_interval = response.get('pollingIntervalMillis', 10000) / 1000
    
    except HttpError as e:
        if e.resp.status == 400:
            st.error("Error: El livestream puede haber terminado o no tener chat habilitado.")
        else:
            st.error(f"Error con la API de YouTube: {e}")
    except Exception as e:
        st.error(f"Ocurrió un error inesperado: {e}")
    
    return comments, next_page_token, polling_interval

def perform_sentiment_analysis(df):
    """Realiza un análisis de sentimiento básico usando VADER."""
    if df.empty:
        return {}
    analyzer = SentimentIntensityAnalyzer()
    df['sentiment'] = df['text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    def categorize_sentiment(score):
        if score >= 0.05: return 'Positivo'
        elif score <= -0.05: return 'Negativo'
        else: return 'Neutral'
    df['sentiment_category'] = df['sentiment'].apply(categorize_sentiment)
    sentiment_counts = df['sentiment_category'].value_counts(normalize=True).round(2)
    return sentiment_counts

# --- Interfaz de usuario principal ---
youtube_url = st.text_input("Ingresa la URL del livestream de YouTube")

# Botón para iniciar el análisis del stream
if st.button("Iniciar Nuevo Análisis"):
    if not st.session_state.api_key:
        st.error("Por favor, ingresa tu clave de API de YouTube.")
    elif not youtube_url:
        st.error("Por favor, ingresa una URL de YouTube válida.")
    else:
        video_id = get_video_id(youtube_url)
        if not video_id:
            st.error("URL de YouTube inválida. Asegúrate de que sea un enlace de livestream.")
        else:
            try:
                st.session_state.youtube_service = build('youtube', 'v3', developerKey=st.session_state.api_key)
                with st.spinner('Obteniendo información del livestream...'):
                    video_request = st.session_state.youtube_service.videos().list(
                        part='liveStreamingDetails', id=video_id
                    )
                    video_response = video_request.execute()
                    
                    live_chat_id = None
                    if video_response.get('items'):
                        live_chat_details = video_response['items'][0].get('liveStreamingDetails')
                        if live_chat_details:
                            live_chat_id = live_chat_details.get('activeLiveChatId')

                if not live_chat_id:
                    st.error("No se encontró un chat en vivo activo para esta URL. ¿Quizás el stream ha terminado?")
                else:
                    st.session_state.live_chat_id = live_chat_id
                    st.session_state.comments = []
                    st.session_state.next_page_token = None
                    st.success("¡Análisis iniciado! Haz clic en 'Actualizar Comentarios' para obtener datos.")
            except Exception as e:
                st.error(f"Error al conectar a la API: {e}. Verifica tu clave de API.")

# Botón para actualizar los comentarios
if st.session_state.live_chat_id and st.button("Actualizar Comentarios"):
    with st.spinner('Actualizando comentarios...'):
        new_comments, next_page_token, polling_interval = fetch_live_chat_comments(
            st.session_state.youtube_service,
            st.session_state.live_chat_id,
            st.session_state.next_page_token
        )
        st.session_state.comments.extend(new_comments)
        st.session_state.next_page_token = next_page_token
    
    st.success(f"Comentarios actualizados. La API recomienda esperar {polling_interval:.0f} segundos antes de la próxima solicitud.")
    time.sleep(1) # Pequeña pausa para que el mensaje de éxito se vea

# --- Visualización de resultados ---
if st.session_state.comments:
    df_comments = pd.DataFrame(st.session_state.comments)
    df_comments['timestamp'] = pd.to_datetime(df_comments['timestamp'])
    
    st.subheader("Resultados del Análisis")
    st.metric(label="Comentarios Totales Analizados", value=len(df_comments))
    
    # Gráfico de comentarios por minuto (gráfico de líneas)
    comments_per_minute = df_comments.groupby(pd.Grouper(key='timestamp', freq='T')).size().reset_index(name='count')
    comments_per_minute['timestamp'] = comments_per_minute['timestamp'].dt.strftime('%H:%M')
    st.line_chart(comments_per_minute.set_index('timestamp'))
    
    sentiment_results = perform_sentiment_analysis(df_comments)
    st.subheader("Análisis de Sentimiento Básico")
    if not sentiment_results.empty:
        col1, col2, col3 = st.columns(3)
        with col1: st.metric(label="Positivos", value=f"{sentiment_results.get('Positivo', 0) * 100:.0f}%")
        with col2: st.metric(label="Neutrales", value=f"{sentiment_results.get('Neutral', 0) * 100:.0f}%")
        with col3: st.metric(label="Negativos", value=f"{sentiment_results.get('Negativo', 0) * 100:.0f}%")
        st.dataframe(df_comments[['timestamp', 'text', 'sentiment_category']].tail(50))
    else:
        st.warning("No hay suficientes comentarios para el análisis de sentimiento.")
