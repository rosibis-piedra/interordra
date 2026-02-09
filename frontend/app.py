"""
InterOrdra - Interfaz Web / Web Interface
Aplicación Streamlit para análisis de desacoplamientos semánticos
"""

import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import sys
import os
import io

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

# --- Funciones de extracción de texto ---
def extract_text_from_pdf(uploaded_file):
    import pdfplumber
    with pdfplumber.open(uploaded_file) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n".join(pages).strip()

def extract_text_from_txt(uploaded_file):
    return uploaded_file.read().decode('utf-8')

def extract_text_from_docx(uploaded_file):
    from docx import Document
    doc = Document(uploaded_file)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

def extract_text(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith('.pdf'):
        return extract_text_from_pdf(uploaded_file)
    elif name.endswith('.txt'):
        return extract_text_from_txt(uploaded_file)
    elif name.endswith('.docx'):
        return extract_text_from_docx(uploaded_file)
    return ""

# Configuración de la página
st.set_page_config(
    page_title="InterOrdra - Semantic Gap Detector",
    page_icon="🌉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Activar Google Analytics
inject_ga()

# --- TRADUCCIONES / TRANSLATIONS ---
TRANSLATIONS = {
    'es': {
        'config': '⚙️ Configuración',
        'threshold': 'Umbral de acoplamiento',
        'threshold_help': 'Similaridad mínima para considerar dos conceptos "acoplados"',
        'how_it_works_title': '### 💡 ¿Cómo funciona?',
        'how_it_works': """
1. **Ingresá dos textos** que querés comparar
2. **InterOrdra analiza** su topología semántica
3. **Visualizá** gaps y desacoplamientos
4. **Recibí recomendaciones** para mejorar
""",
        'use_cases_title': '### 📖 Casos de uso',
        'use_cases': """
- Documentación técnica vs pregunta de usuario
- Prompt de IA vs respuesta esperada
- Artículo científico vs divulgación
- Manual vs FAQ
""",
        'tech_title': '### 🧠 Tecnología',
        'tech': """
InterOrdra usa:
- Sentence Transformers para embeddings
- DBSCAN para clustering
- Análisis de similaridad coseno
- Visualización 3D con PCA
""",
        'created_by': '*Creado por [Rosibis](https://github.com/rosibis)*',
        'architect': '*Arquitecta de puentes semánticos*',
        'subtitle': 'Hace visible la estructura del malentendido',
        'texts_to_analyze': '## 📝 Textos a Analizar',
        'text_a_header': '📄 Texto A',
        'text_b_header': '📄 Texto B',
        'label_a': 'Etiqueta para Texto A',
        'label_b': 'Etiqueta para Texto B',
        'default_label_a': 'Documentación Técnica',
        'default_label_b': 'Pregunta de Usuario',
        'input_a': 'Ingresá el primer texto',
        'input_b': 'Ingresá el segundo texto',
        'placeholder_a': 'Ej: Manual técnico, artículo científico, documentación...',
        'placeholder_b': 'Ej: Pregunta, explicación simplificada, feedback...',
        'load_example': '📚 Cargar ejemplo',
        'select_example': 'Seleccioná un ejemplo',
        'example_none': 'Ninguno',
        'example_tech': 'Documentación Técnica vs Usuario',
        'example_science': 'Artículo Científico vs Divulgación',
        'example_prompt': 'Prompt IA - Confuso vs Claro',
        'load_this': 'Cargar este ejemplo',
        'example_tech_a': """El sistema utiliza embeddings vectoriales de alta dimensión para representar información semántica. Los transformers pre-entrenados generan representaciones contextuales mediante mecanismos de atención multi-cabeza. La arquitectura implementa capas de normalización y conexiones residuales para estabilidad.""",
        'example_tech_b': """¿Cómo funciona esto? No entiendo qué significa que use vectores. ¿Qué es un transformer? ¿Por qué necesito saber de matemáticas avanzadas? Solo quiero que mi aplicación entienda texto en español.""",
        'analyze_button': '🔍 Analizar Desacoplamiento',
        'error_no_text': '⚠️ Por favor ingresá ambos textos antes de analizar',
        'spinner': '🔄 Analizando topología semántica...',
        'results_title': '## 📊 Resultados del Análisis',
        'metric_similarity': 'Similaridad Global',
        'metric_coupled': 'Acoplado',
        'metric_decoupled': 'Desacoplado',
        'metric_gap_severity': 'Severidad del Gap',
        'metric_vocab_overlap': 'Overlap de Vocabulario',
        'metric_orphans': 'Conceptos Huérfanos',
        'topology_title': '### 🗺️ Topología Semántica 3D',
        'topology_subtitle': '*Cada punto es una oración. Distancia = diferencia semántica*',
        'dim1': 'Dimensión 1',
        'dim2': 'Dimensión 2',
        'dim3': 'Dimensión 3',
        'coordinates': 'Coordenadas',
        'gaps_title': '### 🔍 Desacoplamientos Detectados',
        'orphans_in': '🔸 Huérfanos en',
        'vocabulary_tab': '📚 Vocabulario',
        'orphan_count': 'concepto(s) sin equivalente cercano en',
        'full_sentence': 'Oración completa',
        'best_match_in': 'Mejor match en',
        'similarity_label': 'Similaridad',
        'all_concepts_matched': '✅ Todos los conceptos de {a} tienen equivalente en {b}',
        'unique_in': 'Único en',
        'shared_vocab': 'Vocabulario Compartido',
        'none_text': '*(ninguno)*',
        'recommendations_title': '### 💡 Recomendaciones',
        'footer_tagline': 'Construyendo puentes entre sistemas diversos',
        'footer_role': 'Arquitecta de interfaces inter-sistémicas',
        'input_mode': 'Método de entrada',
        'write_text': '✍️ Escribir texto',
        'upload_file': '📁 Subir archivo',
        'upload_label': 'Subir archivo (PDF, TXT, DOCX)',
        'file_preview': 'Vista previa del texto extraído',
        'chars_extracted': 'caracteres extraídos',
        'extract_error': 'Error al extraer texto del archivo',
    },
    'en': {
        'config': '⚙️ Settings',
        'threshold': 'Coupling threshold',
        'threshold_help': 'Minimum similarity to consider two concepts "coupled"',
        'how_it_works_title': '### 💡 How does it work?',
        'how_it_works': """
1. **Enter two texts** you want to compare
2. **InterOrdra analyzes** their semantic topology
3. **Visualize** gaps and decouplings
4. **Get recommendations** to improve
""",
        'use_cases_title': '### 📖 Use Cases',
        'use_cases': """
- Technical documentation vs user question
- AI prompt vs expected response
- Scientific article vs popular science
- Manual vs FAQ
""",
        'tech_title': '### 🧠 Technology',
        'tech': """
InterOrdra uses:
- Sentence Transformers for embeddings
- DBSCAN for clustering
- Cosine similarity analysis
- 3D visualization with PCA
""",
        'created_by': '*Created by [Rosibis](https://github.com/rosibis)*',
        'architect': '*Semantic bridge architect*',
        'subtitle': 'Making the structure of misunderstanding visible',
        'texts_to_analyze': '## 📝 Texts to Analyze',
        'text_a_header': '📄 Text A',
        'text_b_header': '📄 Text B',
        'label_a': 'Label for Text A',
        'label_b': 'Label for Text B',
        'default_label_a': 'Technical Documentation',
        'default_label_b': 'User Question',
        'input_a': 'Enter the first text',
        'input_b': 'Enter the second text',
        'placeholder_a': 'E.g.: Technical manual, scientific article, documentation...',
        'placeholder_b': 'E.g.: Question, simplified explanation, feedback...',
        'load_example': '📚 Load Example',
        'select_example': 'Select an example',
        'example_none': 'None',
        'example_tech': 'Technical Documentation vs User',
        'example_science': 'Scientific Article vs Popular Science',
        'example_prompt': 'AI Prompt - Confusing vs Clear',
        'load_this': 'Load this example',
        'example_tech_a': """The system uses high-dimensional vector embeddings to represent semantic information. Pre-trained transformers generate contextual representations through multi-head attention mechanisms. The architecture implements normalization layers and residual connections for stability.""",
        'example_tech_b': """How does this work? I don't understand what it means to use vectors. What is a transformer? Why do I need to know advanced mathematics? I just want my application to understand text in English.""",
        'analyze_button': '🔍 Analyze Decoupling',
        'error_no_text': '⚠️ Please enter both texts before analyzing',
        'spinner': '🔄 Analyzing semantic topology...',
        'results_title': '## 📊 Analysis Results',
        'metric_similarity': 'Global Similarity',
        'metric_coupled': 'Coupled',
        'metric_decoupled': 'Decoupled',
        'metric_gap_severity': 'Gap Severity',
        'metric_vocab_overlap': 'Vocabulary Overlap',
        'metric_orphans': 'Orphan Concepts',
        'topology_title': '### 🗺️ 3D Semantic Topology',
        'topology_subtitle': '*Each point is a sentence. Distance = semantic difference*',
        'dim1': 'Dimension 1',
        'dim2': 'Dimension 2',
        'dim3': 'Dimension 3',
        'coordinates': 'Coordinates',
        'gaps_title': '### 🔍 Detected Decouplings',
        'orphans_in': '🔸 Orphans in',
        'vocabulary_tab': '📚 Vocabulary',
        'orphan_count': 'concept(s) with no close equivalent in',
        'full_sentence': 'Full sentence',
        'best_match_in': 'Best match in',
        'similarity_label': 'Similarity',
        'all_concepts_matched': '✅ All concepts from {a} have an equivalent in {b}',
        'unique_in': 'Unique to',
        'shared_vocab': 'Shared Vocabulary',
        'none_text': '*(none)*',
        'recommendations_title': '### 💡 Recommendations',
        'footer_tagline': 'Building bridges between diverse systems',
        'footer_role': 'Inter-systemic interface architect',
        'input_mode': 'Input method',
        'write_text': '✍️ Write text',
        'upload_file': '📁 Upload file',
        'upload_label': 'Upload file (PDF, TXT, DOCX)',
        'file_preview': 'Preview of extracted text',
        'chars_extracted': 'characters extracted',
        'extract_error': 'Error extracting text from file',
    }
}

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

# Sidebar con selector de idioma y configuración
with st.sidebar:
    lang = st.selectbox(
        "🌐",
        options=['es', 'en'],
        index=0,
        format_func=lambda x: "🇪🇸 Español" if x == 'es' else "🇬🇧 English",
        key="lang"
    )

def t(key):
    """Devuelve el texto traducido según el idioma seleccionado."""
    return TRANSLATIONS[lang][key]

# Sidebar - resto de configuración
with st.sidebar:
    st.header(t('config'))

    threshold = st.slider(
        t('threshold'),
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help=t('threshold_help')
    )

    st.markdown("---")

    st.markdown(t('how_it_works_title'))
    st.markdown(t('how_it_works'))

    st.markdown(t('use_cases_title'))
    st.markdown(t('use_cases'))

    st.markdown(t('tech_title'))
    st.markdown(t('tech'))

    st.markdown("---")

    st.markdown(t('created_by'))
    st.markdown(t('architect'))

# Título
st.markdown('<div class="main-title">🌉 InterOrdra</div>', unsafe_allow_html=True)
st.markdown(f'<div class="subtitle">{t("subtitle")}</div>', unsafe_allow_html=True)

# Inputs principales
st.markdown(t('texts_to_analyze'))

col1, col2 = st.columns(2)

with col1:
    st.subheader(t('text_a_header'))
    label_a = st.text_input(
        t('label_a'),
        value=t('default_label_a'),
        key="label_a"
    )
    mode_a = st.radio(
        t('input_mode'),
        [t('write_text'), t('upload_file')],
        key="mode_a",
        horizontal=True
    )
    if mode_a == t('write_text'):
        text_a = st.text_area(
            t('input_a'),
            height=250,
            placeholder=t('placeholder_a'),
            key="text_a"
        )
    else:
        file_a = st.file_uploader(
            t('upload_label'),
            type=['pdf', 'txt', 'docx'],
            key="file_a"
        )
        text_a = ""
        if file_a:
            try:
                text_a = extract_text(file_a)
                st.success(f"✅ {len(text_a)} {t('chars_extracted')}")
                with st.expander(t('file_preview')):
                    st.text(text_a[:1000] + ("..." if len(text_a) > 1000 else ""))
            except Exception as e:
                st.error(f"{t('extract_error')}: {e}")

with col2:
    st.subheader(t('text_b_header'))
    label_b = st.text_input(
        t('label_b'),
        value=t('default_label_b'),
        key="label_b"
    )
    mode_b = st.radio(
        t('input_mode'),
        [t('write_text'), t('upload_file')],
        key="mode_b",
        horizontal=True
    )
    if mode_b == t('write_text'):
        text_b = st.text_area(
            t('input_b'),
            height=250,
            placeholder=t('placeholder_b'),
            key="text_b"
        )
    else:
        file_b = st.file_uploader(
            t('upload_label'),
            type=['pdf', 'txt', 'docx'],
            key="file_b"
        )
        text_b = ""
        if file_b:
            try:
                text_b = extract_text(file_b)
                st.success(f"✅ {len(text_b)} {t('chars_extracted')}")
                with st.expander(t('file_preview')):
                    st.text(text_b[:1000] + ("..." if len(text_b) > 1000 else ""))
            except Exception as e:
                st.error(f"{t('extract_error')}: {e}")

# Ejemplos precargados
with st.expander(t('load_example')):
    ejemplo = st.selectbox(
        t('select_example'),
        options=[
            t('example_none'),
            t('example_tech'),
            t('example_science'),
            t('example_prompt')
        ]
    )

    if ejemplo == t('example_tech'):
        if st.button(t('load_this')):
            st.session_state.text_a = t('example_tech_a')
            st.session_state.text_b = t('example_tech_b')
            st.rerun()

# Botón de análisis
st.markdown("---")

analyze_button = st.button(
    t('analyze_button'),
    type="primary",
    use_container_width=True
)

# Análisis
if analyze_button:
    if not text_a or not text_b:
        st.error(t('error_no_text'))
    else:
        with st.spinner(t('spinner')):
            # Inicializar analyzer
            analyzer = InterOrdraAnalyzer(language=lang, threshold=threshold)

            # Ejecutar análisis
            results = analyzer.analyze(text_a, text_b, label_a, label_b)

        # Mostrar resultados
        st.markdown("---")
        st.markdown(t('results_title'))

        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)

        sim = results['gaps']['global_similarity']
        gap_severity = results['gaps']['gaps']['gap_severity']
        vocab_overlap = results['gaps']['vocabulary_analysis']['vocabulary_overlap']

        with col1:
            st.metric(
                t('metric_similarity'),
                f"{sim:.1%}",
                delta=t('metric_coupled') if results['gaps']['is_coupled'] else t('metric_decoupled')
            )

        with col2:
            st.metric(
                t('metric_gap_severity'),
                f"{gap_severity:.1%}"
            )

        with col3:
            st.metric(
                t('metric_vocab_overlap'),
                f"{vocab_overlap:.1%}"
            )

        with col4:
            total_orphans = results['gaps']['gaps']['orphan_count_a'] + results['gaps']['gaps']['orphan_count_b']
            st.metric(
                t('metric_orphans'),
                total_orphans
            )

        # Visualización 3D
        st.markdown("---")
        st.markdown(t('topology_title'))
        st.markdown(t('topology_subtitle'))

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
            hovertemplate='<b>%{text}</b><br>' + t('coordinates') + ': (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
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
            hovertemplate='<b>%{text}</b><br>' + t('coordinates') + ': (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
        ))

        fig.update_layout(
            scene=dict(
                xaxis_title=t('dim1'),
                yaxis_title=t('dim2'),
                zaxis_title=t('dim3'),
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
        st.markdown(t('gaps_title'))

        tab1, tab2, tab3 = st.tabs([
            f"{t('orphans_in')} {label_a}",
            f"{t('orphans_in')} {label_b}",
            t('vocabulary_tab')
        ])

        with tab1:
            orphans_a = results['gaps']['gaps']['text_a_orphans']
            if orphans_a:
                st.markdown(f"**{len(orphans_a)} {t('orphan_count')} {label_b}:**")
                for i, orphan in enumerate(orphans_a, 1):
                    with st.expander(f"{i}. {orphan['sentence'][:80]}..."):
                        st.markdown(f"**{t('full_sentence')}:**  \n{orphan['sentence']}")
                        st.markdown(f"**{t('best_match_in')} {label_b}:**  \n{orphan['best_match_in_b']}")
                        st.progress(orphan['best_match_similarity'])
                        st.caption(f"{t('similarity_label')}: {orphan['best_match_similarity']:.1%}")
            else:
                st.success(t('all_concepts_matched').format(a=label_a, b=label_b))

        with tab2:
            orphans_b = results['gaps']['gaps']['text_b_orphans']
            if orphans_b:
                st.markdown(f"**{len(orphans_b)} {t('orphan_count')} {label_a}:**")
                for i, orphan in enumerate(orphans_b, 1):
                    with st.expander(f"{i}. {orphan['sentence'][:80]}..."):
                        st.markdown(f"**{t('full_sentence')}:**  \n{orphan['sentence']}")
                        st.markdown(f"**{t('best_match_in')} {label_a}:**  \n{orphan['best_match_in_a']}")
                        st.progress(orphan['best_match_similarity'])
                        st.caption(f"{t('similarity_label')}: {orphan['best_match_similarity']:.1%}")
            else:
                st.success(t('all_concepts_matched').format(a=label_b, b=label_a))

        with tab3:
            vocab = results['gaps']['vocabulary_analysis']

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**{t('unique_in')} {label_a}:**")
                if vocab['unique_to_a']:
                    st.write(", ".join(vocab['unique_to_a']))
                else:
                    st.write(t('none_text'))

            with col2:
                st.markdown(f"**{t('unique_in')} {label_b}:**")
                if vocab['unique_to_b']:
                    st.write(", ".join(vocab['unique_to_b']))
                else:
                    st.write(t('none_text'))

            st.markdown(f"**{t('shared_vocab')}:**")
            if vocab['shared']:
                st.write(", ".join(vocab['shared']))
            else:
                st.write(t('none_text'))

        # Recomendaciones
        st.markdown("---")
        st.markdown(t('recommendations_title'))

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
st.markdown(f"""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p><strong>InterOrdra v0.1.0</strong> • {t('footer_tagline')}</p>
    <p>{'Creado por' if lang == 'es' else 'Created by'} <a href='#'>Rosibis</a> • {t('footer_role')}</p>
</div>
""", unsafe_allow_html=True)
