import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from collections import Counter
import re

# =====================
# CONFIGURACIÓN INICIAL
# =====================
st.set_page_config(page_title="Panel Evaluacion llamadas", layout="wide")
st.title("📞 Panel Evaluacion llamadas")
# =====================
# CARGA DE DATOS
# =====================
@st.cache_data
def cargar_datos():
    url_csv = "https://drive.google.com/uc?export=download&id=19Bl3fDoM4jBoB6ZU-hQ7_-YVT2BhiQly"
    df = pd.read_csv(url_csv)
    
    df.columns = df.columns.str.strip()
    df["Duración (min)"] = df["Duración (seg)"] / 60
    
    try:
        df["Fecha"] = pd.to_datetime(df["Archivo"].str.extract(r"(\d{4}-\d{2}-\d{2})")[0], errors="coerce")
    except:
        df["Fecha"] = pd.NaT
    
    mapa_fluidez = {"Excelente": 10, "Buena": 8, "Regular": 6, "Deficiente": 4, "Malo": 2}
    df["Puntaje Fluidez"] = df["Evaluación Fluidez"].map(mapa_fluidez).fillna(0)
    
    df["Puntaje Calidad"] = (df["% Apego al guion"] * 0.5) + \
                            (df["% Sentimiento"] * 0.3) + \
                            (df["Puntaje Fluidez"] * 10 * 0.2)
    
    def clasificar(p):
        if p >= 85: return "🏆 Ejemplar"
        elif p >= 70: return "✅ Satisfactorio"
        elif p >= 50: return "⚠️ Necesita Mejora"
        else: return "🔴 Crítico"
    
    df["Clasificación"] = df["Puntaje Calidad"].apply(clasificar)
    
    return df

df = cargar_datos()

# =====================
# FILTROS GLOBALES
# =====================
st.sidebar.header("Filtros")
vendedores = st.sidebar.multiselect("Seleccionar vendedor", options=df["Vendedor"].unique(), default=df["Vendedor"].unique())
fecha_min = st.sidebar.date_input("Fecha mínima", value=df["Fecha"].min() if df["Fecha"].notna().any() else None)
fecha_max = st.sidebar.date_input("Fecha máxima", value=df["Fecha"].max() if df["Fecha"].notna().any() else None)

df_filtrado = df.copy()
if vendedores:
    df_filtrado = df_filtrado[df_filtrado["Vendedor"].isin(vendedores)]
if fecha_min and fecha_max:
    df_filtrado = df_filtrado[(df_filtrado["Fecha"] >= pd.to_datetime(fecha_min)) & 
                              (df_filtrado["Fecha"] <= pd.to_datetime(fecha_max))]

# =====================
# AYUDA - SIGNIFICADO DE MÉTRICAS
# =====================
with st.expander("ℹ️ Significado y correlación de métricas"):
    st.markdown("""
    **% Apego al guion** → Porcentaje de frases clave del guion mencionadas.  
    **% Sentimiento** → Emoción global (positivo o negativo).  
    **KPI Fluidez** → Velocidad y naturalidad en la conversación.  
    **Evaluación Fluidez** → Versión cualitativa de KPI Fluidez.  
    **Evaluación interrupciones** → Nivel de interrupciones durante la llamada.  
    **Diversidad léxica** → Riqueza del vocabulario usado.  
    **Long. prom. oraciones** → Tamaño promedio de las frases.  
    **Palabras de relleno** → "eh", "este", "o sea", etc.  
    **Interrupciones** → Número de interrupciones.  
    **Silencios largos** → Pausas extendidas.  
    **Energía de voz** → Intensidad vocal.  
    **Tasa de habla** → Palabras por minuto.  
    **Tono promedio** → Promedio de frecuencia tonal.  
    """)

# =====================
# PESTAÑAS
# =====================
tab1, tab2, tab3 = st.tabs(["📊 Resumen general", "👤 Vista por vendedor", "🗣 Análisis de lenguaje"])

# ---------------------
# TAB 1: RESUMEN GENERAL
# ---------------------
with tab1:
    st.subheader("Resumen general del equipo")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Promedio Apego (%)", f"{df_filtrado['% Apego al guion'].mean():.1f}%")
    col2.metric("Promedio Sentimiento (%)", f"{df_filtrado['% Sentimiento'].mean():.1f}%")
    col3.metric("Llamadas analizadas", len(df_filtrado))
    col4.metric("Duración promedio (min)", f"{df_filtrado['Duración (min)'].mean():.1f}")

    # Comparativa por vendedor
    fig_bar = px.bar(df_filtrado.groupby("Vendedor")["% Apego al guion"].mean().reset_index(),
                     x="Vendedor", y="% Apego al guion", title="Apego al guion por vendedor",
                     color="% Apego al guion", color_continuous_scale="RdYlGn")
    st.plotly_chart(fig_bar, use_container_width=True)

    # Tabla de calidad promedio por vendedor
    st.markdown("### 🏅 Calidad promedio por vendedor")
    calidad_vendedores = df_filtrado.groupby("Vendedor")[["Puntaje Calidad"]].mean().reset_index()
    calidad_vendedores["Clasificación"] = calidad_vendedores["Puntaje Calidad"].apply(lambda x: df["Clasificación"].iloc[0] if len(df) else "")
    st.dataframe(calidad_vendedores)

# ---------------------
# TAB 2: VISTA POR VENDEDOR
# ---------------------
with tab2:
    vendedor_sel = st.selectbox("Seleccionar vendedor", options=df_filtrado["Vendedor"].unique())
    df_vend = df_filtrado[df_filtrado["Vendedor"] == vendedor_sel]

    st.subheader(f"Desempeño de {vendedor_sel}")

    # Gauge Apego
    if not df_vend.empty:
        apego = df_vend["% Apego al guion"].mean()
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=apego,
            title={'text': "Apego al guion (%)"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "green" if apego >= 80 else "orange" if apego >= 50 else "red"}}
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # Tabla con clasificación por llamada
    st.markdown("### 📋 Calidad de llamadas")
    st.dataframe(df_vend[["Archivo", "Duración (min)", "% Apego al guion", "% Sentimiento", "Puntaje Calidad", "Clasificación"]])

# ---------------------
# TAB 3: ANÁLISIS DE LENGUAJE
# ---------------------
with tab3:
    st.subheader("Análisis de lenguaje")

    # Nube de palabras
    texto_completo = " ".join(df_filtrado["Transcripción completa"].dropna().astype(str))
    if texto_completo:
        wc = WordCloud(width=800, height=400, background_color="white").generate(texto_completo)
        st.image(wc.to_array(), caption="Nube de palabras", use_container_width=True)

    # Palabras de relleno
    if "Transcripción completa" in df_filtrado.columns:
        palabras_relleno = re.findall(r"\b(eh|este|o sea|mmm|ah)\b", texto_completo.lower())
        conteo_relleno = Counter(palabras_relleno)
        df_relleno = pd.DataFrame(conteo_relleno.items(), columns=["Palabra", "Frecuencia"]).sort_values(by="Frecuencia", ascending=False)
        
        st.markdown("### 📊 Distribución de palabras de relleno")
        fig_fill = px.bar(df_relleno, x="Palabra", y="Frecuencia", title="Frecuencia de palabras de relleno")
        st.plotly_chart(fig_fill, use_container_width=True)
        st.dataframe(df_relleno)

    # Comparativa de energía y tono
    if "Energía de voz" in df_filtrado.columns and "Tono promedio" in df_filtrado.columns:
        fig_scatter = px.scatter(df_filtrado, x="Tono promedio", y="Energía de voz", color="% Apego al guion",
                                 size="Tasa de habla", title="Tono vs Energía")
        st.plotly_chart(fig_scatter, use_container_width=True)





