import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import time
import os
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote, unquote


# ==========================================
# PART 1: TMDB API se movies fetch karna
# ==========================================

# API_KEY ab hardcoded nahi hai — Streamlit secrets ya environment variable se aayega.
# Local mein chalane ke liye: .streamlit/secrets.toml file banao (isi folder ke andar) aur likho:
#   TMDB_API_KEY = "apni_key_yahan"
# Streamlit Cloud pe deploy karte waqt: app settings > Secrets mein wahi line paste kar dena.
try:
    API_KEY = st.secrets["TMDB_API_KEY"]
except Exception:
    API_KEY = os.getenv("TMDB_API_KEY", "PASTE_YOUR_TMDB_API_KEY_HERE")

BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w300"   # poster image ka base URL

# Kitne pages fetch karne hain (har page mein ~20 movies hoti hain)
NUM_PAGES_HOLLYWOOD = 60
NUM_PAGES_BOLLYWOOD = 40
NUM_PAGES_SOUTH = 20   # Telugu + Tamil ke beech split hoga (10-10)


def safe_get(url, params, retries=3, delay=1):
    """Requests.get jo network error pe khud retry karta hai, warna None deta hai."""
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException:
            time.sleep(delay)
    return None


def get_genre_mapping():
    """Genre ID -> Genre Name ka mapping banata hai (e.g. 28 -> Action)"""
    url = f"{BASE_URL}/genre/movie/list"
    params = {"api_key": API_KEY, "language": "en-US"}
    response = safe_get(url, params)
    if response is None:
        return {}
    genres = response.json()["genres"]
    return {g["id"]: g["name"] for g in genres}


def get_keywords_and_business(movie_id):
    """Ek hi call mein keywords + budget + revenue fetch karta hai (append_to_response se)."""
    url = f"{BASE_URL}/movie/{movie_id}"
    params = {"api_key": API_KEY, "append_to_response": "keywords"}
    response = safe_get(url, params)
    if response is None:
        return "", 0, 0

    data = response.json()
    keywords_list = data.get("keywords", {}).get("keywords", [])
    keywords = " ".join([k["name"] for k in keywords_list])
    budget = data.get("budget", 0) or 0
    revenue = data.get("revenue", 0) or 0

    return keywords, budget, revenue


def get_verdict(budget, revenue):
    """Budget vs revenue ke hisaab se Hit/Superhit/Flop verdict nikalta hai."""
    if not budget or not revenue:
        return "N/A"

    ratio = revenue / budget

    if ratio >= 4:
        return "🔥 Superhit"
    elif ratio >= 2:
        return "✅ Hit"
    elif ratio >= 1:
        return "➖ Average"
    else:
        return "📉 Flop"


@st.cache_data(ttl=3600)
def get_usd_to_inr_rate():
    """Live USD->INR rate fetch karta hai (1 ghante cache), fail hone pe fixed rate deta hai."""
    try:
        response = requests.get("https://api.exchangerate-api.com/v4/latest/USD", timeout=5)
        response.raise_for_status()
        return response.json()["rates"]["INR"]
    except Exception:
        return 83.0   # fallback approx rate


def format_inr(usd_amount):
    """USD amount ko INR Crore/Lakh format mein badalta hai (e.g. ₹1,234 Cr)."""
    rate = get_usd_to_inr_rate()
    inr_amount = usd_amount * rate

    if inr_amount >= 1_00_00_000:  # 1 crore
        return f"₹{inr_amount / 1_00_00_000:,.2f} Cr"
    elif inr_amount >= 1_00_000:   # 1 lakh
        return f"₹{inr_amount / 1_00_000:,.2f} Lakh"
    else:
        return f"₹{inr_amount:,.0f}"


def fetch_popular_page(page):
    """Hollywood ke liye — TMDB ki global popular list (mostly English movies)."""
    url = f"{BASE_URL}/movie/popular"
    params = {"api_key": API_KEY, "language": "en-US", "page": page}
    response = safe_get(url, params)
    if response is None:
        return []
    return response.json().get("results", [])


def fetch_discover_page(language_code, page):
    """Discover endpoint se ek specific language ki movies fetch karta hai, popularity sorted."""
    url = f"{BASE_URL}/discover/movie"
    params = {
        "api_key": API_KEY,
        "with_original_language": language_code,
        "sort_by": "popularity.desc",
        "page": page
    }
    response = safe_get(url, params)
    if response is None:
        return []
    return response.json().get("results", [])


def fetch_movies():
    genre_map = get_genre_mapping()

    # Step 1: Hollywood + Bollywood + South (Telugu/Tamil) — sab PARALLEL mein fetch karo
    basic_movies = []
    seen_ids = set()

    south_pages_each = NUM_PAGES_SOUTH // 2  # Telugu aur Tamil ke beech split

    with ThreadPoolExecutor(max_workers=25) as executor:
        futures_with_industry = []

        for page in range(1, NUM_PAGES_HOLLYWOOD + 1):
            futures_with_industry.append((executor.submit(fetch_popular_page, page), "Hollywood"))

        for page in range(1, NUM_PAGES_BOLLYWOOD + 1):
            futures_with_industry.append((executor.submit(fetch_discover_page, "hi", page), "Bollywood"))

        for page in range(1, south_pages_each + 1):
            futures_with_industry.append((executor.submit(fetch_discover_page, "te", page), "South"))
            futures_with_industry.append((executor.submit(fetch_discover_page, "ta", page), "South"))

        for future, industry in futures_with_industry:
            for movie in future.result():
                movie_id = movie.get("id")
                if movie_id in seen_ids:
                    continue
                seen_ids.add(movie_id)

                genre_ids = movie.get("genre_ids", [])
                poster_path = movie.get("poster_path", "")

                basic_movies.append({
                    "title": movie.get("title", ""),
                    "movie_id": movie_id,
                    "genre": " ".join([genre_map.get(gid, "") for gid in genre_ids]),
                    "poster_url": f"{IMAGE_BASE_URL}{poster_path}" if poster_path else "",
                    "overview": movie.get("overview", ""),
                    "release_date": movie.get("release_date", ""),
                    "rating": movie.get("vote_average", ""),
                    "industry": industry
                })

    # Step 2: Keywords + budget + revenue ko PARALLEL mein fetch karo (ab ek hi call mein)
    all_movies = [None] * len(basic_movies)

    with ThreadPoolExecutor(max_workers=30) as executor:
        future_to_index = {
            executor.submit(get_keywords_and_business, m["movie_id"]): i
            for i, m in enumerate(basic_movies)
        }

        for future in as_completed(future_to_index):
            i = future_to_index[future]
            keywords, budget, revenue = future.result()
            movie_info = basic_movies[i]
            all_movies[i] = {
                "title": movie_info["title"],
                "genre": movie_info["genre"],
                "keywords": keywords,
                "poster_url": movie_info["poster_url"],
                "overview": movie_info["overview"],
                "release_date": movie_info["release_date"],
                "rating": movie_info["rating"],
                "budget": budget,
                "revenue": revenue,
                "verdict": get_verdict(budget, revenue),
                "industry": movie_info["industry"]
            }

    return pd.DataFrame(all_movies)


def ensure_dataset():
    """Agar movies.csv nahi hai to fetch karke banata hai. Sirf ek baar chalta hai."""
    if not os.path.exists("movies.csv"):
        with st.spinner("Movies fetch ho rahi hain TMDB se (pehli baar thoda time lagega)..."):
            df = fetch_movies()
            df.drop_duplicates(subset="title", inplace=True)
            df.to_csv("movies.csv", index=False)


# ==========================================
# PART 2: Data load + model taiyar karna
# (cache karte hain taaki har rerun pe dubara na ho)
# ==========================================

@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    movies["genre"] = movies["genre"].fillna("")
    movies["keywords"] = movies["keywords"].fillna("")
    movies["poster_url"] = movies["poster_url"].fillna("")
    movies["overview"] = movies["overview"].fillna("") if "overview" in movies.columns else ""
    if "budget" in movies.columns:
        movies["budget"] = movies["budget"].fillna(0)
    if "revenue" in movies.columns:
        movies["revenue"] = movies["revenue"].fillna(0)
    if "verdict" in movies.columns:
        movies["verdict"] = movies["verdict"].fillna("N/A")
    if "industry" in movies.columns:
        movies["industry"] = movies["industry"].fillna("Hollywood")
    else:
        movies["industry"] = "Hollywood"

    movies["tags"] = (
        movies["genre"].astype(str)
        + " "
        + movies["keywords"].astype(str)
    )

    cv = CountVectorizer()
    vectors = cv.fit_transform(movies["tags"])
    similarity = cosine_similarity(vectors)

    return movies, similarity


def get_recommendations(movie, movies, similarity):
    """Movie ka naam leke top-5 recommended movies (rows) return karta hai."""
    movie = movie.lower()
    titles = movies["title"].str.lower()

    if movie not in titles.values:
        return None

    index = titles[titles == movie].index[0]
    distances = similarity[index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    return [movies.iloc[i] for i, _ in movie_list]


# ==========================================
# PART 3: Streamlit UI
# ==========================================

st.set_page_config(page_title="Movie Recommender", layout="wide")

# Netflix-jaisa dark theme + red accents ke liye custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #000000;
    }
    h1, h2, h3, .stMarkdown, label, p {
        color: #e5e5e5 !important;
    }
    h1 {
        color: #e50914 !important;
        font-weight: 800 !important;
    }
    div[data-testid="stSelectbox"] > div > div {
        background-color: #2b2b2b;
        color: white;
        border: 1px solid #3d3d3d;
    }
    .stButton > button {
        background-color: #e50914;
        color: white;
        border: none;
        border-radius: 4px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
    }
    .stButton > button:hover {
        background-color: #f6121d;
        color: white;
    }
    div[data-testid="stImage"] img {
        border-radius: 6px;
        transition: transform 0.2s ease;
    }
    div[data-testid="stImage"] img:hover {
        transform: scale(1.05);
    }
    div[data-testid="stCaptionContainer"] {
        color: #e5e5e5 !important;
        text-align: center;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎬 Movie Recommender")

ensure_dataset()
movies, similarity = load_data()


def render_movie_card(col, movie_row, key_prefix, idx):
    """Ek movie card banata hai — poster pe click karne se details khulti hain."""
    with col:
        title = movie_row.get("title", "")
        poster_url = movie_row.get("poster_url", "")

        if poster_url:
            st.markdown(
                f"""
                <a href="?selected={quote(title)}" target="_self" style="text-decoration: none;">
                    <img src="{poster_url}" style="width:100%; border-radius:6px; transition: transform 0.2s ease; cursor: pointer;"
                         onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'">
                </a>
                """,
                unsafe_allow_html=True
            )
        st.caption(title)


# ---------- Details panel (jab kisi movie ke poster pe click kiya jaaye) ----------
selected_title = st.query_params.get("selected")

if selected_title:
    selected_title = unquote(selected_title)
    match = movies[movies["title"] == selected_title]

    if not match.empty:
        m = match.iloc[0]
        with st.container():
            st.markdown("### 🎬 " + m.get("title", ""))
            detail_cols = st.columns([1, 2])
            with detail_cols[0]:
                if m.get("poster_url"):
                    st.image(m["poster_url"], use_container_width=True)
            with detail_cols[1]:
                if m.get("genre"):
                    st.markdown(f"**Genre:** {m['genre']}")
                if m.get("release_date"):
                    st.markdown(f"**Release Date:** {m['release_date']}")
                if m.get("rating"):
                    st.markdown(f"**Rating:** ⭐ {m['rating']}/10")
                if m.get("revenue"):
                    st.markdown(f"**Box Office Collection:** {format_inr(m['revenue'])}")
                if m.get("verdict") and m.get("verdict") != "N/A":
                    st.markdown(f"**Verdict:** {m['verdict']}")
                if m.get("overview"):
                    st.markdown(f"**Overview:** {m['overview']}")
            if st.button("✕ Close"):
                st.query_params.clear()
                st.rerun()
        st.markdown("---")

# ---------- Search + Recommend (ab sabse upar) ----------
movie_name = st.selectbox(
    "Ek movie choose karo:",
    options=sorted(movies["title"].dropna().unique())
)

searched = st.button("Recommend")

if searched:
    results = get_recommendations(movie_name, movies, similarity)

    if results is None:
        st.error("Ye movie dataset mein nahi mili!")
    else:
        st.subheader("Recommended Movies")
        cols = st.columns(5)

        for idx, (col, movie_row) in enumerate(zip(cols, results)):
            render_movie_card(col, movie_row, "rec", idx)

else:
    # ---------- Trending Now — Bollywood, Hollywood, South alag-alag ----------
    st.subheader("🔥 Trending Now")

    for industry_label, emoji in [("Bollywood", "🇮🇳"), ("Hollywood", "🎥"), ("South", "🎬")]:
        industry_movies = movies[movies["industry"] == industry_label].head(5)

        if industry_movies.empty:
            continue

        st.markdown(f"#### {emoji} {industry_label}")
        row_cols = st.columns(5)

        for i, (_, movie_row) in enumerate(industry_movies.iterrows()):
            render_movie_card(row_cols[i], movie_row, f"trend_{industry_label}", i)