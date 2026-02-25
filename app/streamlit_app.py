#Setting Libraries

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Videogame Recommender", page_icon="🎮", layout="wide")

DATA_URL = "https://huggingface.co/datasets/pabloramcos/Videogame-Recommender-Final-Project/resolve/main/games.parquet"

@st.cache_data(show_spinner=False)
def load_data(url: str, sample_n: int = 5000) -> pd.DataFrame:
    df = pd.read_parquet(url)

    if sample_n and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=42)

    for c in ["name", "release_date", "short_description", "detailed_description"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)

    if "app_id" in df.columns:
        df["app_id"] = pd.to_numeric(df["app_id"], errors="coerce").astype("Int64")
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
    if "required_age" in df.columns:
        df["required_age"] = pd.to_numeric(df["required_age"], errors="coerce").fillna(0).astype(int)

    return df

st.title("🎮 Videogame Recommender")

# --- Lazy load: NO cargamos al inicio ---
if "df" not in st.session_state:
    st.session_state["df"] = None

sample_n = st.sidebar.slider("Tamaño de carga (rápido)", 1000, 30000, 5000, step=1000)

if st.session_state["df"] is None:
    st.info("Pulsa para cargar el dataset.")
    if st.button("Cargar dataset"):
        with st.spinner("Cargando…"):
            st.session_state["df"] = load_data(DATA_URL, sample_n=sample_n)
        st.success("Dataset cargado ✅")
    st.stop()

df = st.session_state["df"]

# Sidebar filtros
st.sidebar.header("Filtros")
q = st.sidebar.text_input("Buscar por nombre", "")

# Precio
price_range = None
if "price" in df.columns and df["price"].notna().any():
    pmin = float(df["price"].min())
    pmax = float(df["price"].max())
    price_range = st.sidebar.slider("Precio", min_value=pmin, max_value=pmax, value=(pmin, min(pmax, 30.0)))

# Edad
age_range = None
if "required_age" in df.columns:
    amin = int(df["required_age"].min())
    amax = int(df["required_age"].max())
    age_range = st.sidebar.slider("Edad requerida", min_value=amin, max_value=amax, value=(amin, amax))

# Año (best-effort usando release_date)
year_range = st.sidebar.slider("Año (aprox)", min_value=1980, max_value=2026, value=(2005, 2026))

work = df.copy()

if q.strip():
    work = work[work["name"].str.contains(q, case=False, na=False)]

if price_range and "price" in work.columns:
    work = work[(work["price"] >= price_range[0]) & (work["price"] <= price_range[1])]

if age_range and "required_age" in work.columns:
    work = work[(work["required_age"] >= age_range[0]) & (work["required_age"] <= age_range[1])]

if "release_date" in work.columns:
    years = work["release_date"].str.extract(r"(\d{4})")[0]
    work["_year"] = pd.to_numeric(years, errors="coerce")
    work = work[
        (work["_year"].fillna(year_range[0]) >= year_range[0]) &
        (work["_year"].fillna(year_range[1]) <= year_range[1])
    ]

st.write(f"Resultados: **{len(work):,}**")

cols_show = [c for c in ["app_id", "name", "release_date", "required_age", "price"] if c in work.columns]
show = work[cols_show].head(1000)

left, right = st.columns([1, 1])

row = None  # 👈 importante: por defecto no hay juego seleccionado

with left:
    st.subheader("Lista")
    st.dataframe(show, use_container_width=True, height=540)

with right:
    st.subheader("Ficha del juego")
    if len(show) == 0:
        st.info("No hay resultados con esos filtros.")
    else:
        selected_idx = st.selectbox(
            "Selecciona un juego",
            options=show.index.tolist(),
            format_func=lambda i: str(show.loc[i, "name"])[:80]
        )
        row = work.loc[selected_idx]

        st.markdown(f"### {row.get('name','(sin nombre)')}")
        meta_cols = [c for c in ["app_id", "release_date", "required_age", "price"] if c in row.index]
        st.json({c: row.get(c) for c in meta_cols})

        desc = row.get("detailed_description", "")
        if desc:
            st.markdown("**Descripción**")
            st.write(desc[:2000] + ("…" if len(desc) > 2000 else ""))

        if pd.notna(row.get("app_id", None)):
            st.markdown(f"**Steam:** https://store.steampowered.com/app/{int(row['app_id'])}/")

# -------------------------
# Recomendador simple
# -------------------------

@st.cache_resource
def build_tfidf(text_series: pd.Series, max_features: int = 20000):
    vec = TfidfVectorizer(stop_words="english", max_features=max_features)
    X = vec.fit_transform(text_series)
    return vec, X

st.markdown("### 🤝 Recomendador simple (similares por descripción)")

# Si todavía no hay juego seleccionado, no intentamos recomendar
if row is None:
    st.info("Selecciona un juego arriba para ver recomendaciones.")
    st.stop()

# Ajustes en sidebar
sample_n_rec = st.sidebar.slider("Tamaño muestra para recomendador", 2000, 50000, 15000, step=1000)
top_k = st.sidebar.slider("Número de recomendaciones", 3, 20, 10)

text_col = "detailed_description" if "detailed_description" in work.columns else "short_description"

base = work.copy()
if len(base) > sample_n_rec:
    base = base.sample(n=sample_n_rec, random_state=42)

base[text_col] = base[text_col].fillna("").astype(str)

if len(base) < 5 or base[text_col].str.len().sum() == 0:
    st.info("No hay texto suficiente para recomendar.")
else:
    vec, X = build_tfidf(base[text_col])

    # buscar el juego seleccionado dentro del subset (mejor por app_id)
    target_idx = None
    if "app_id" in base.columns and pd.notna(row.get("app_id", None)):
        matches = base.index[base["app_id"] == row["app_id"]].tolist()
        if matches:
            target_idx = matches[0]

    if target_idx is None:
        matches = base.index[base["name"] == row["name"]].tolist()
        if matches:
            target_idx = matches[0]

    if target_idx is None:
        st.info("El juego seleccionado no está en la muestra del recomendador. Sube el tamaño de muestra o cambia filtros.")
    else:
        i = base.index.get_loc(target_idx)
        sims = cosine_similarity(X[i], X).flatten()
        order = sims.argsort()[::-1]

        # quitar el propio juego (el más similar es él mismo)
        order = [j for j in order if j != i][:top_k]

        recs = base.iloc[order][["app_id", "name", "price", "release_date"]].copy()
        st.dataframe(recs, use_container_width=True)