# -*- coding: utf-8 -*-
# =====================================================
# Universidad del Quindío - Ingeniería de Sistemas
# Proyecto: Análisis de Algoritmos en el Contexto de la Bibliometría
# Requerimientos 2, 3 y Seguimiento 2 — Similitud, Frecuencia y Grafos
# Autor: Yeison Álvarez
# =====================================================

import streamlit as st
import pandas as pd
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sentence_transformers import SentenceTransformer
import textdistance
import re
import json
from pathlib import Path
import chardet
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import tempfile
import os

# =====================================================
# CONFIGURACIÓN GENERAL
# =====================================================
st.set_page_config(layout="wide", page_title="Análisis Bibliométrico con IA")
st.title("🔬 Análisis Bibliométrico con Inteligencia Artificial")
st.markdown("""
Aplicación para los **Requerimientos 2, 3 y Seguimiento 2** del proyecto de Análisis de Algoritmos — Universidad del Quindío.
""")

# =====================================================
# CONFIGURAR API GEMINI
# =====================================================
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    gemini_is_configured = True
except (KeyError, AttributeError):
    gemini_is_configured = False
    st.warning("⚠️ No se encontró la API Key de Google Gemini. Las funciones IA serán limitadas.")

# =====================================================
# LISTA DE PALABRAS CLAVE PREDEFINIDAS
# =====================================================
PREDEFINED_KEYWORDS = [
    "Generative models", "Prompting", "Machine learning", "Multimodality",
    "Fine-tuning", "Training data", "Algorithmic bias", "Explainability",
    "Transparency", "Ethics", "Privacy", "Personalization",
    "Human-AI interaction", "AI literacy", "Co-creation"
]

# =====================================================
# FUNCIONES AUXILIARES
# =====================================================
@st.cache_data
def load_data(file_path):
    """Carga CSV detectando automáticamente encoding y separador."""
    try:
        with open(file_path, 'rb') as f:
            enc = chardet.detect(f.read())['encoding']
        for sep in [',', ';', '\t']:
            try:
                df = pd.read_csv(file_path, encoding=enc or 'latin1', sep=sep, on_bad_lines='skip', engine='python')
                if len(df.columns) > 1:
                    break
            except Exception:
                continue
        df.columns = df.columns.str.strip().str.lower()
        if not all(col in df.columns for col in ["title", "abstract"]):
            st.error("❌ El archivo debe contener las columnas: 'title' y 'abstract'.")
            return None
        df = df.dropna(subset=["title", "abstract"])
        st.success(f"✅ Archivo cargado correctamente con codificación {enc} y separador '{sep}'")
        return df
    except Exception as e:
        st.error(f"⚠️ Error al leer el archivo: {e}")
        return None


def preprocess_text(text):
    """Limpieza básica del texto."""
    stopwords = {'a','un','una','el','la','los','las','de','del','en','y','o','con','por','para','su',
                 'the','an','and','in','of','to','is','for','on','with','as','by','this','that','it','are','be'}
    text = text.lower()
    text = re.sub(r'[^a-záéíóúüñ\s]', ' ', text)
    tokens = [w.strip() for w in text.split() if len(w) > 2 and w not in stopwords]
    return tokens

# =====================================================
# ALGORITMOS DE SIMILITUD
# =====================================================
def jaccard_similarity(t1, t2):
    s1, s2 = set(preprocess_text(t1)), set(preprocess_text(t2))
    return len(s1 & s2) / len(s1 | s2) if s1 | s2 else 0

def cosine_tfidf(t1, t2):
    v = TfidfVectorizer(tokenizer=preprocess_text, token_pattern=None)
    X = v.fit_transform([t1, t2])
    return cosine_similarity(X[0:1], X[1:2])[0][0]

def levenshtein_sim(t1, t2):
    return textdistance.levenshtein.normalized_similarity(t1, t2)

def euclidean_sim(t1, t2):
    v = TfidfVectorizer(tokenizer=preprocess_text, token_pattern=None)
    X = v.fit_transform([t1, t2])
    dist = euclidean_distances(X[0:1], X[1:2])[0][0]
    return 1 / (1 + dist)

@st.cache_data
def embedding_sim(t1, t2):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    emb = model.encode([t1, t2])
    return cosine_similarity([emb[0]], [emb[1]])[0][0]

def gemini_similarity_analysis(t1, t2):
    """Usa Gemini para analizar similitud conceptual."""
    if not gemini_is_configured:
        return "⚠️ Gemini no configurado."
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = f"Compara los siguientes textos y explica su similitud conceptual en 5 líneas.\n\nTexto 1: {t1}\n\nTexto 2: {t2}"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error en Gemini: {e}"

# =====================================================
# REQ 3 — FRECUENCIAS Y IA DE PALABRAS NUEVAS
# =====================================================
def predefined_frequencies(df):
    text = " ".join(df['abstract'].str.lower())
    data = []
    for kw in PREDEFINED_KEYWORDS:
        count = len(re.findall(r'\b' + re.escape(kw.lower()) + r'\b', text))
        data.append({'name': kw, 'frequency': count})
    return sorted(data, key=lambda x: x['frequency'], reverse=True)

@st.cache_data
def generate_new_keywords_with_ai(text):
    """Genera nuevas keywords con Gemini."""
    if not gemini_is_configured:
        return {"newKeywords": [], "analysis": "Gemini no configurado."}
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = f"""
        Analiza los siguientes abstracts y propone hasta 15 palabras clave relacionadas con 'Generative AI in Education'.
        Devuelve JSON: {{ "newKeywords":[{{"keyword":"", "frequency":int}}], "analysis":"..." }}
        Textos: {text}
        """
        resp = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(response_mime_type="application/json"))
        return json.loads(resp.text)
    except Exception as e:
        return {"newKeywords": [], "analysis": str(e)}

# =====================================================
# CARGAR DATOS
# =====================================================
DATA_FILE_PATH = Path("DataFinal/unificado.csv")
df_articles = load_data(DATA_FILE_PATH)

if df_articles is not None:
    st.success(f"✅ {len(df_articles)} artículos cargados correctamente.")

    # ===========================================
    # MENÚ SUPERIOR CON TODOS LOS TABS
    # ===========================================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Req 2: Similitud Textual",
        "📈 Req 3: Frecuencias y Keywords IA",
        "🔗 Seguimiento 2: Redes Bibliométricas",
        "🔎 Req 4: Análisis de Keywords",
        "💾 Req 5: Exportación Resultados"
    ])

    # -------------------------------
    # TAB 1 — REQUERIMIENTO 2
    # -------------------------------
    with tab1:
        st.header("📊 Requerimiento 2 — Similitud Textual entre Artículos")
        col1, col2 = st.columns(2)
        with col1:
            a1_title = st.selectbox("Primer artículo:", df_articles['title'])
        with col2:
            a2_title = st.selectbox("Segundo artículo:", df_articles[df_articles['title'] != a1_title]['title'])

        if st.button("Calcular Similitud", type="primary"):
            t1 = df_articles[df_articles['title'] == a1_title].iloc[0]['abstract']
            t2 = df_articles[df_articles['title'] == a2_title].iloc[0]['abstract']

            jaccard = jaccard_similarity(t1, t2)
            cosine = cosine_tfidf(t1, t2)
            lev = levenshtein_sim(t1, t2)
            eucl = euclidean_sim(t1, t2)
            embed = embedding_sim(t1, t2)
            gem_analysis = gemini_similarity_analysis(t1, t2)

            st.subheader("📈 Resultados de Similitud")
            colA, colB = st.columns(2)
            with colA:
                st.metric("Jaccard", f"{jaccard:.4f}")
                st.metric("Cosine TF-IDF", f"{cosine:.4f}")
                st.metric("Levenshtein", f"{lev:.4f}")
                st.metric("Euclidiana", f"{eucl:.4f}")
            with colB:
                st.metric("MiniLM (Embeddings IA)", f"{embed:.4f}")
                st.markdown("#### 🤖 Gemini — Análisis Conceptual")
                st.info(gem_analysis)

        with st.expander("📘 Explicación Técnica de Algoritmos"):
            st.markdown("""
            - **Jaccard:** mide coincidencia de términos (|A∩B|/|A∪B|).  
            - **Cosine TF-IDF:** compara vectores de frecuencia ponderada.  
            - **Levenshtein:** mide distancia de edición carácter a carácter.  
            - **Euclidiana:** distancia geométrica entre vectores TF-IDF.  
            - **MiniLM (IA):** embeddings semánticos entrenados (Sentence-BERT).  
            - **Gemini (IA):** análisis conceptual generativo del significado textual.
            """)

# -------------------------------
# TAB 2 — REQUERIMIENTO 3
# -------------------------------
    with tab2:
        st.header("📈 Requerimiento 3 — Frecuencia y Generación de Keywords IA")
        freq = predefined_frequencies(df_articles)
        df_freq = pd.DataFrame(freq).set_index('name')
        st.bar_chart(df_freq)

        if st.button("Generar nuevas keywords con IA"):
            corpus = " ".join(df_articles['abstract'].astype(str).tolist())
            ai_result = generate_new_keywords_with_ai(corpus)
            st.session_state['ai_keywords'] = ai_result

        if 'ai_keywords' in st.session_state:
            res = st.session_state['ai_keywords']
            if res.get('newKeywords'):
                df_ai = pd.DataFrame(res['newKeywords'])
                if 'keyword' in df_ai.columns:
                    df_ai = df_ai.rename(columns={'keyword':'name'})
                df_ai = df_ai.set_index('name')
                st.markdown("#### 🔍 Nuevas palabras IA generadas")
                st.dataframe(df_ai)
                predefined = {k.lower() for k in PREDEFINED_KEYWORDS}
                ai_set = {n.lower() for n in df_ai.index}
                precision = len(predefined & ai_set) / len(ai_set) if ai_set else 0
                st.success(f"**Precisión respecto a palabras originales:** {precision:.2%}")
            else:
                st.warning(f"No se generaron nuevas palabras. Detalle: {res.get('analysis')}")

# -------------------------------
# TAB 3 — SEGUIMIENTO 2
# -------------------------------
    with tab3:
        st.header("🔗 Seguimiento 2 — Redes de Citaciones y Coocurrencia")
        subtab1, subtab2 = st.tabs(["📘 Red de Citaciones", "🧩 Red de Coocurrencia"])

        # RED DE CITACIONES
        with subtab1:
            st.subheader("📘 Grafo de Citaciones — con Caminos Mínimos y Componentes")
            selected = st.multiselect(
                "Selecciona los artículos que deseas incluir en el grafo de citaciones:",
                df_articles['title'].tolist(),
                default=[]
            )
            threshold = st.slider("Umbral de similitud", 0.0, 1.0, 0.25, 0.05)
            use_emb = st.checkbox("Usar embeddings IA (MiniLM)", value=True)

            if st.button("Construir Grafo de Citaciones", type="primary"):
                if not selected:
                    st.warning("⚠️ Debes seleccionar al menos un artículo para construir el grafo.")
                else:
                    subset = df_articles[df_articles['title'].isin(selected)]
                    abstracts = subset['abstract'].tolist()
                    titles = subset['title'].tolist()

                    if use_emb:
                        model = SentenceTransformer('all-MiniLM-L6-v2')
                        emb = model.encode(abstracts, show_progress_bar=True)
                        sim_matrix = cosine_similarity(emb)
                    else:
                        v = TfidfVectorizer(tokenizer=preprocess_text, token_pattern=None)
                        tfidf = v.fit_transform(abstracts)
                        sim_matrix = cosine_similarity(tfidf)

                    st.session_state["grafo_titles"] = titles
                    st.session_state["grafo_matrix"] = sim_matrix
                    st.session_state["grafo_threshold"] = threshold

            if "grafo_matrix" in st.session_state:
                titles = st.session_state["grafo_titles"]
                sim_matrix = st.session_state["grafo_matrix"]
                threshold = st.session_state["grafo_threshold"]

                df_sim = pd.DataFrame(sim_matrix, index=titles, columns=titles)
                st.markdown("### 🧮 Matriz de Similitud")
                st.dataframe(df_sim.style.background_gradient(cmap="Blues").format("{:.2f}"))

                G = nx.DiGraph()
                for i, t1 in enumerate(titles):
                    for j, t2 in enumerate(titles):
                        if i != j and sim_matrix[i][j] >= threshold:
                            G.add_edge(t1, t2, weight=round(float(sim_matrix[i][j]), 2))

                if len(G.nodes()) == 0:
                    st.warning("⚠️ Sin conexiones, baja el umbral o elige más artículos.")
                else:
                    id_map = {title: f"A{i+1}" for i, title in enumerate(titles)}
                    G_short = nx.relabel_nodes(G, id_map)
                    fig, ax = plt.subplots(figsize=(10, 8))
                    pos = nx.spring_layout(G_short, k=0.6, seed=42)
                    nx.draw(G_short, pos,
                            with_labels=True, node_color="skyblue", node_size=900,
                            arrows=True, font_size=9, font_weight='bold')
                    nx.draw_networkx_edge_labels(
                        G_short, pos,
                        edge_labels={(u, v): d['weight'] for u, v, d in G_short.edges(data=True)},
                        font_size=7
                    )
                    plt.title(f"Grafo de Citaciones — {len(G.nodes())} nodos / {len(G.edges())} aristas", fontsize=11)
                    st.pyplot(fig)

                    st.markdown("### 🗂️ Leyenda de artículos")
                    df_leg = pd.DataFrame({"ID": list(id_map.values()), "Título": list(id_map.keys())})
                    st.dataframe(df_leg)

                    densidad = nx.density(G)
                    grado_prom = sum(dict(G.degree()).values()) / len(G)
                    st.markdown(f"**Densidad:** {densidad:.4f} | **Grado promedio:** {grado_prom:.2f}")

                    st.markdown("#### 🚀 Caminos Mínimos (Dijkstra)")
                    origen = st.selectbox("Selecciona artículo origen:", titles, key="origen_dijkstra")
                    destino = st.selectbox("Selecciona artículo destino:", titles, key="destino_dijkstra")

                    if st.button("Calcular camino mínimo"):
                        try:
                            path = nx.shortest_path(G, source=origen, target=destino, weight='weight', method='dijkstra')
                            distance = nx.shortest_path_length(G, source=origen, target=destino, weight='weight', method='dijkstra')
                            st.session_state["dijkstra_path"] = path
                            st.session_state["dijkstra_distance"] = distance
                        except nx.NetworkXNoPath:
                            st.session_state["dijkstra_path"] = None
                            st.session_state["dijkstra_distance"] = None

                    if "dijkstra_path" in st.session_state:
                        if st.session_state["dijkstra_path"]:
                            st.success(f"Camino más corto: {' ➜ '.join(st.session_state['dijkstra_path'])} (Distancia: {st.session_state['dijkstra_distance']:.2f})")
                        else:
                            st.warning("⚠️ No hay camino entre esos nodos.")

                    st.markdown("#### 🔗 Componentes Fuertemente Conexas")
                    comp = list(nx.strongly_connected_components(G))
                    st.write(f"Se detectaron **{len(comp)}** componentes:")
                    for i, c in enumerate(comp, 1):
                        st.write(f"Componente {i}: {', '.join(c)}")

# -------------------------------
# RED DE COOCURRENCIA
# -------------------------------
        with subtab2:
            st.subheader("🧩 Grafo de Coocurrencia de Palabras Clave")
            window = st.slider("Ventana de palabras", 5, 40, 10, 5)
            threshold = st.slider("Umbral mínimo de coocurrencia", 1, 10, 2, 1)

            terms = [re.sub(r'[-\s]+', ' ', kw.lower().strip()) for kw in PREDEFINED_KEYWORDS]
            G2 = nx.Graph()
            for t in terms:
                G2.add_node(t)

            for abs_ in df_articles['abstract']:
                tokens = preprocess_text(abs_)
                for i in range(len(tokens)):
                    window_tokens = tokens[i:i+window]
                    presentes = [t for t in terms if any(t in w for w in window_tokens)]
                    for a in range(len(presentes)):
                        for b in range(a+1, len(presentes)):
                            x, y = presentes[a], presentes[b]
                            if G2.has_edge(x, y):
                                G2[x][y]['weight'] += 1
                            else:
                                G2.add_edge(x, y, weight=1)

            edges_validas = [(u, v, d) for u, v, d in G2.edges(data=True) if d['weight'] >= threshold]
            if not edges_validas:
                st.warning("⚠️ No se encontraron coocurrencias con ese umbral.")
            else:
                G3 = nx.Graph()
                G3.add_weighted_edges_from([(u, v, d['weight']) for u, v, d in edges_validas])
                fig, ax = plt.subplots(figsize=(9, 7))
                pos = nx.spring_layout(G3, k=0.6, seed=42)
                nx.draw(G3, pos, with_labels=True, node_color="lightgreen", node_size=800, edge_color="gray", font_size=8)
                nx.draw_networkx_edge_labels(G3, pos, edge_labels={(u, v): d['weight'] for u, v, d in G3.edges(data=True)}, font_size=7)
                plt.title(f"Grafo de Coocurrencia — {len(G3.nodes())} nodos / {len(G3.edges())} aristas", fontsize=10)
                st.pyplot(fig)

                st.markdown("### 📊 Análisis del Grafo de Coocurrencia")
                grados = dict(G3.degree())
                df_grado = pd.DataFrame(list(grados.items()), columns=["Término", "Grado"]).sort_values("Grado", ascending=False)
                st.dataframe(df_grado)

                st.markdown("#### 🔗 Componentes Conexas")
                componentes = list(nx.connected_components(G3))
                st.write(f"Se detectaron **{len(componentes)}** componentes:")
                for i, c in enumerate(componentes, 1):
                    st.write(f"Componente {i}: {', '.join(c)}")

                st.success("✅ Análisis completado correctamente.")

# =====================================================
# TAB 4 — REQUERIMIENTO 4: Clustering Jerárquico
# =====================================================
with tab4:
    st.header("🔎 Requerimiento 4 — Clustering Jerárquico de Abstracts")

    # --- Selección de artículos ---
    selected_articles = st.multiselect(
        "Selecciona los artículos a analizar (clustering):",
        df_articles['title'].tolist()
    )
    subset_df = df_articles[df_articles['title'].isin(selected_articles)]
    st.info(f"✅ {len(subset_df)} artículos seleccionados para clustering")

    if len(subset_df) == 0:
        st.warning("⚠️ Debes seleccionar al menos un artículo.")
    else:
        # --- Embeddings (pesado, cacheado) ---
        @st.cache_data
        def compute_embeddings(texts):
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            return model.encode(texts, show_progress_bar=True)

        with st.spinner("Calculando embeddings..."):
            embeddings = compute_embeddings(subset_df['abstract'].tolist())

        # --- Clustering jerárquico ---
        from scipy.cluster.hierarchy import linkage, dendrogram
        import matplotlib.pyplot as plt

        with st.spinner("Construyendo dendrogramas con 3 algoritmos..."):
            methods = ["ward", "complete", "average"]
            for method in methods:
                st.markdown(f"### 📌 Algoritmo: {method}")
                Z = linkage(embeddings, method=method)
                fig, ax = plt.subplots(figsize=(12, 6))
                dendrogram(Z, labels=subset_df['title'].tolist(), leaf_rotation=90)
                plt.title(f"Dendrograma ({method})")
                st.pyplot(fig)

# =====================================================
# TAB 5 — REQUERIMIENTO 5: Análisis Visual de Producción Científica
# =====================================================
with tab5:
    st.header("💾 Requerimiento 5 — Análisis Visual de Producción Científica")

    # --- Selección de artículos ---
    selected_articles_r5 = st.multiselect(
        "Selecciona los artículos a analizar (REQ 5):",
        df_articles['title'].tolist()
    )
    subset_r5 = df_articles[df_articles['title'].isin(selected_articles_r5)]
    st.info(f"✅ {len(subset_r5)} artículos seleccionados para visualización")

    if len(subset_r5) == 0:
        st.warning("⚠️ Debes seleccionar al menos un artículo.")
    else:
        # -----------------------------
        # 1️⃣ Mapa de calor geográfico (primer autor)
        # -----------------------------
        st.subheader("🌍 Mapa de calor geográfico por primer autor")
        if 'affiliation' in subset_r5.columns:
            import plotly.express as px
            df_geo = subset_r5['affiliation'].value_counts().reset_index()
            df_geo.columns = ['affiliation', 'count']
            fig = px.choropleth(df_geo,
                                locations="affiliation",
                                locationmode='country names',
                                color="count",
                                title="Distribución geográfica de artículos")
            st.plotly_chart(fig)
        else:
            st.warning("❌ No se encuentra columna 'affiliation' para mapa geográfico")

        # -----------------------------
        # 2️⃣ Nube de palabras dinámica
        # -----------------------------
        st.subheader("☁️ Nube de palabras (abstracts y keywords)")
        from wordcloud import WordCloud

        text_corpus = " ".join(subset_r5['abstract'].astype(str).tolist())
        if 'keywords' in subset_r5.columns:
            text_corpus += " " + " ".join(subset_r5['keywords'].astype(str).tolist())

        wc = WordCloud(width=800, height=400, background_color='white').generate(text_corpus)
        fig_wc, ax_wc = plt.subplots(figsize=(12, 6))
        ax_wc.imshow(wc, interpolation='bilinear')
        ax_wc.axis("off")
        st.pyplot(fig_wc)

        # -----------------------------
        # 3️⃣ Línea temporal de publicaciones por año
        # -----------------------------
        st.subheader("📈 Línea temporal de publicaciones por año")
        if 'year' in subset_r5.columns:
            subset_r5['year'] = pd.to_numeric(subset_r5['year'], errors='coerce')
            subset_r5 = subset_r5.dropna(subset=['year'])

            timeline = subset_r5.groupby('year').size().reset_index(name='count')

            import plotly.express as px
            fig_time = px.line(
                timeline,
                x='year',
                y='count',
                title="Publicaciones por año"
            )
            st.plotly_chart(fig_time)
        else:
            st.warning("❌ No se encuentra columna 'year' en los datos")

        # -----------------------------
        # 4️⃣ Exportación a PDF
        # -----------------------------
        st.subheader("📄 Exportar visualizaciones a PDF")
        if st.button("Exportar PDF"):
            from fpdf import FPDF
            import tempfile

            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "Análisis Visual de Producción Científica", ln=True, align="C")

            # Guardar figuras temporalmente y agregarlas al PDF
            figs = [fig_wc, fig_time]
            for fig in figs:
                tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                fig.savefig(tmpfile.name)
                pdf.image(tmpfile.name, x=10, y=30, w=190)
                tmpfile.close()
            pdf.output("Analisis_Visual.pdf")
            st.success("✅ PDF generado: 'Analisis_Visual.pdf'")
