"""
InterOrdra - Interfaz Web
Aplicación Streamlit para análisis de desacoplamientos semánticos
"""

import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import sys
import os

# Agregar backend al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.analyzer import InterOrdraAnalyzer

# Google Analytics
def inject_ga():
    """Inyecta código de Google Analytics"""
    ga_code = """
    <!-- Google tag (gtag.js) -->
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-BZPFS4HWR2"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){dataLayer.push(arguments);}
      gtag('js', new Date());
      gtag('config', 'G-BZPFS4HWR2');
    </script>
    """
    components.html(ga_code, height=0)

# Configuración de la página
st.set_page_config(
    page_title="InterOrdra - Detector of Semantic Gaps",
    page_icon="🌉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Activar Google Analytics
inject_ga()

st.set_page_config(
    page_title="InterOrdra - Detector de Gaps Semánticos",
    page_icon="🌉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-good {
        color: #28a745;
        font-weight: bold;
    }
    .metric-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .metric-bad {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Título
st.markdown('<div class="main-title">🌉 InterOrdra</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Hace visible la estructura del malentendido</div>', unsafe_allow_html=True)

# Sidebar con información
with st.sidebar:
    st.header("⚙️ Configuración")
    
    threshold = st.slider(
        "Umbral de acoplamiento",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Similaridad mínima para considerar dos conceptos 'acoplados'"
    )
    
    language = st.selectbox(
        "Idioma",
        options=['es', 'en'],
        index=0,
        format_func=lambda x: "🇪🇸 Español" if x == 'es' else "🇬🇧 English"
    )
    
    st.markdown("---")
    
    st.markdown("""
    ### 💡 ¿Cómo funciona?
    
    1. **Ingresá dos textos** que querés comparar
    2. **InterOrdra analiza** su topología semántica
    3. **Visualizá** gaps y desacoplamientos
    4. **Recibí recomendaciones** para mejorar
    
    ### 📖 Casos de uso
    
    - Documentación técnica vs pregunta de usuario
    - Prompt de IA vs respuesta esperada
    - Artículo científico vs divulgación
    - Manual vs FAQ
    
    ### 🧠 Tecnología
    
    InterOrdra usa:
    - Sentence Transformers para embeddings
    - DBSCAN para clustering
    - Análisis de similaridad coseno
    - Visualización 3D con PCA
    
    ---
    
    *Creado por [Rosibis](https://github.com/rosibis)*  
    *Arquitecta de puentes semánticos*
    """)

# Inputs principales
st.markdown("## 📝 Textos a Analizar")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Texto A")
    label_a = st.text_input(
        "Etiqueta para Texto A",
        value="Documentación Técnica",
        key="label_a"
    )
    text_a = st.text_area(
        "Ingresá el primer texto",
        height=250,
        placeholder="Ej: Manual técnico, artículo científico, documentación...",
        key="text_a"
    )

with col2:
    st.subheader("📄 Texto B")
    label_b = st.text_input(
        "Etiqueta para Texto B",
        value="Pregunta de Usuario",
        key="label_b"
    )
    text_b = st.text_area(
        "Ingresá el segundo texto",
        height=250,
        placeholder="Ej: Pregunta, explicación simplificada, feedback...",
        key="text_b"
    )

# Ejemplos precargados
with st.expander("📚 Cargar ejemplo"):
    ejemplo = st.selectbox(
        "Seleccioná un ejemplo",
        options=[
            "Ninguno",
            "Documentación Técnica vs Usuario",
            "Artículo Científico vs Divulgación",
            "Prompt IA - Confuso vs Claro"
        ]
    )
    
    if ejemplo == "Documentación Técnica vs Usuario":
        if st.button("Cargar este ejemplo"):
            st.session_state.text_a = """El sistema utiliza embeddings vectoriales de alta dimensión para representar información semántica. Los transformers pre-entrenados generan representaciones contextuales mediante mecanismos de atención multi-cabeza. La arquitectura implementa capas de normalización y conexiones residuales para estabilidad."""
            st.session_state.text_b = """¿Cómo funciona esto? No entiendo qué significa que use vectores. ¿Qué es un transformer? ¿Por qué necesito saber de matemáticas avanzadas? Solo quiero que mi aplicación entienda texto en español."""
            st.rerun()

# Botón de análisis
st.markdown("---")

analyze_button = st.button(
    "🔍 Analizar Desacoplamiento",
    type="primary",
    use_container_width=True
)

# Análisis
if analyze_button:
    if not text_a or not text_b:
        st.error("⚠️ Por favor ingresá ambos textos antes de analizar")
    else:
        with st.spinner("🔄 Analizando topología semántica..."):
            # Inicializar analyzer
            analyzer = InterOrdraAnalyzer(language=language, threshold=threshold)
            
            # Ejecutar análisis
            results = analyzer.analyze(text_a, text_b, label_a, label_b)
        
        # Mostrar resultados
        st.markdown("---")
        st.markdown("## 📊 Resultados del Análisis")
        
        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        
        sim = results['gaps']['global_similarity']
        gap_severity = results['gaps']['gaps']['gap_severity']
        vocab_overlap = results['gaps']['vocabulary_analysis']['vocabulary_overlap']
        
        with col1:
            color_class = "metric-good" if sim > 0.7 else "metric-warning" if sim > 0.5 else "metric-bad"
            st.metric(
                "Similaridad Global",
                f"{sim:.1%}",
                delta="Acoplado" if results['gaps']['is_coupled'] else "Desacoplado"
            )
        
        with col2:
            st.metric(
                "Severidad del Gap",
                f"{gap_severity:.1%}"
            )
        
        with col3:
            st.metric(
                "Overlap de Vocabulario",
                f"{vocab_overlap:.1%}"
            )
        
        with col4:
            total_orphans = results['gaps']['gaps']['orphan_count_a'] + results['gaps']['gaps']['orphan_count_b']
            st.metric(
                "Conceptos Huérfanos",
                total_orphans
            )
        
        # Visualización 3D
        st.markdown("---")
        st.markdown("### 🗺️ Topología Semántica 3D")
        st.markdown("*Cada punto es una oración. Distancia = diferencia semántica*")
        
        fig = go.Figure()
        
        # Puntos de texto A
        coords_a = results['text_a']['coords_3d']
        fig.add_trace(go.Scatter3d(
            x=[c[0] for c in coords_a],
            y=[c[1] for c in coords_a],
            z=[c[2] for c in coords_a],
            mode='markers',
            name=label_a,
            marker=dict(
                size=10,
                color='#3b82f6',
                opacity=0.8,
                line=dict(color='white', width=1)
            ),
            text=results['text_a']['sentences'],
            hovertemplate='<b>%{text}</b><br>Coordenadas: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
        ))
        
        # Puntos de texto B
        coords_b = results['text_b']['coords_3d']
        fig.add_trace(go.Scatter3d(
            x=[c[0] for c in coords_b],
            y=[c[1] for c in coords_b],
            z=[c[2] for c in coords_b],
            mode='markers',
            name=label_b,
            marker=dict(
                size=10,
                color='#ef4444',
                opacity=0.8,
                line=dict(color='white', width=1)
            ),
            text=results['text_b']['sentences'],
            hovertemplate='<b>%{text}</b><br>Coordenadas: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
        ))
        
        fig.update_layout(
            scene=dict(
                xaxis_title='Dimensión 1',
                yaxis_title='Dimensión 2',
                zaxis_title='Dimensión 3',
                bgcolor='rgba(0,0,0,0)'
            ),
            height=600,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Gaps detectados
        st.markdown("---")
        st.markdown("### 🔍 Desacoplamientos Detectados")
        
        tab1, tab2, tab3 = st.tabs([
            f"🔸 Huérfanos en {label_a}",
            f"🔸 Huérfanos en {label_b}",
            "📚 Vocabulario"
        ])
        
        with tab1:
            orphans_a = results['gaps']['gaps']['text_a_orphans']
            if orphans_a:
                st.markdown(f"**{len(orphans_a)} concepto(s) sin equivalente cercano en {label_b}:**")
                for i, orphan in enumerate(orphans_a, 1):
                    with st.expander(f"{i}. {orphan['sentence'][:80]}..."):
                        st.markdown(f"**Oración completa:**  \n{orphan['sentence']}")
                        st.markdown(f"**Mejor match en {label_b}:**  \n{orphan['best_match_in_b']}")
                        st.progress(orphan['best_match_similarity'])
                        st.caption(f"Similaridad: {orphan['best_match_similarity']:.1%}")
            else:
                st.success(f"✅ Todos los conceptos de {label_a} tienen equivalente en {label_b}")
        
        with tab2:
            orphans_b = results['gaps']['gaps']['text_b_orphans']
            if orphans_b:
                st.markdown(f"**{len(orphans_b)} concepto(s) sin equivalente cercano en {label_a}:**")
                for i, orphan in enumerate(orphans_b, 1):
                    with st.expander(f"{i}. {orphan['sentence'][:80]}..."):
                        st.markdown(f"**Oración completa:**  \n{orphan['sentence']}")
                        st.markdown(f"**Mejor match en {label_a}:**  \n{orphan['best_match_in_a']}")
                        st.progress(orphan['best_match_similarity'])
                        st.caption(f"Similaridad: {orphan['best_match_similarity']:.1%}")
            else:
                st.success(f"✅ Todos los conceptos de {label_b} tienen equivalente en {label_a}")
        
        with tab3:
            vocab = results['gaps']['vocabulary_analysis']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Único en {label_a}:**")
                if vocab['unique_to_a']:
                    st.write(", ".join(vocab['unique_to_a']))
                else:
                    st.write("*(ninguno)*")
            
            with col2:
                st.markdown(f"**Único en {label_b}:**")
                if vocab['unique_to_b']:
                    st.write(", ".join(vocab['unique_to_b']))
                else:
                    st.write("*(ninguno)*")
            
            st.markdown("**Vocabulario Compartido:**")
            if vocab['shared']:
                st.write(", ".join(vocab['shared']))
            else:
                st.write("*(ninguno)*")
        
        # Recomendaciones
        st.markdown("---")
        st.markdown("### 💡 Recomendaciones")
        
        for rec in results['gaps']['recommendations']:
            severity_emoji = {
                'high': '🔴',
                'medium': '🟡',
                'low': '🟢',
                'info': 'ℹ️'
            }
            emoji = severity_emoji.get(rec['severity'], '•')
            
            if rec['severity'] == 'high':
                st.error(f"{emoji} **{rec['message']}**  \n{rec['suggestion']}")
            elif rec['severity'] == 'medium':
                st.warning(f"{emoji} **{rec['message']}**  \n{rec['suggestion']}")
            elif rec['severity'] == 'low':
                st.success(f"{emoji} **{rec['message']}**  \n{rec['suggestion']}")
            else:
                st.info(f"{emoji} **{rec['message']}**  \n{rec['suggestion']}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p><strong>InterOrdra v0.1.0</strong> • Construyendo puentes entre sistemas diversos</p>
    <p>Creado por <a href='#'>Rosibis</a> • Arquitecta de interfaces inter-sistémicas</p>
</div>
""", unsafe_allow_html=True)