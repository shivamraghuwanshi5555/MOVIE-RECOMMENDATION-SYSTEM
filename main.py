import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import time
import os
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote, unquote
import html as html_lib
import base64


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


@st.cache_data(ttl=86400)
def get_trailer_url(movie_id):
    """Movie ka YouTube trailer (ya na mile to teaser) URL TMDB se nikalta hai."""
    if not movie_id:
        return None
    url = f"{BASE_URL}/movie/{int(movie_id)}/videos"
    params = {"api_key": API_KEY, "language": "en-US"}
    response = safe_get(url, params)
    if response is None:
        return None

    videos = response.json().get("results", [])
    # Pehle official "Trailer" dhundo, warna "Teaser" chalega
    for video_type in ["Trailer", "Teaser"]:
        for v in videos:
            if v.get("site") == "YouTube" and v.get("type") == video_type:
                return f"https://www.youtube.com/watch?v={v['key']}"
    return None


@st.cache_data(ttl=86400)
def get_movie_extras(movie_id):
    """Cast + Where-to-Watch (India) ek hi TMDB call mein nikalta hai."""
    empty = {"cast": [], "providers": [], "provider_type": "", "link": ""}
    if not movie_id:
        return empty
    url = f"{BASE_URL}/movie/{int(movie_id)}"
    params = {"api_key": API_KEY, "append_to_response": "credits,watch/providers"}
    response = safe_get(url, params)
    if response is None:
        return empty

    data = response.json()

    cast = [
        {
            "name": c.get("name", ""),
            "character": c.get("character", ""),
            "photo": f"https://image.tmdb.org/t/p/w185{c['profile_path']}" if c.get("profile_path") else "",
        }
        for c in data.get("credits", {}).get("cast", [])[:10]
    ]

    india = data.get("watch/providers", {}).get("results", {}).get("IN", {})
    providers, provider_type = [], ""
    # Pehle stream (subscription), fir rent, fir buy
    for key, label in [("flatrate", "Stream"), ("rent", "Rent"), ("buy", "Buy")]:
        if india.get(key):
            providers = [
                {"name": p.get("provider_name", ""),
                 "logo": f"https://image.tmdb.org/t/p/w92{p['logo_path']}" if p.get("logo_path") else ""}
                for p in india[key]
            ]
            provider_type = label
            break

    return {"cast": cast, "providers": providers, "provider_type": provider_type, "link": india.get("link", "")}


def get_verdict(budget, revenue, release_date=None):
    """Budget vs revenue se verdict. Data bharosemand na ho to 'N/A' deta hai (galat label se behtar)."""
    try:
        budget = float(budget)
        revenue = float(revenue)
    except (TypeError, ValueError):
        return "N/A"

    # TMDB mein kayi movies ka budget/collection missing ya placeholder hota hai (jaise $1000)
    if budget < 500_000 or revenue < 500_000:
        return "N/A"

    # Abhi theatre mein chal rahi (ya release nahi hui) movie ka collection adhoora hota hai
    release_dt = pd.to_datetime(release_date, errors="coerce")
    if pd.notna(release_dt) and release_dt > pd.Timestamp.today() - pd.Timedelta(days=60):
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
                "movie_id": movie_info["movie_id"],
                "genre": movie_info["genre"],
                "keywords": keywords,
                "poster_url": movie_info["poster_url"],
                "overview": movie_info["overview"],
                "release_date": movie_info["release_date"],
                "rating": movie_info["rating"],
                "budget": budget,
                "revenue": revenue,
                "verdict": get_verdict(budget, revenue, movie_info["release_date"]),
                "industry": movie_info["industry"]
            }

    return pd.DataFrame(all_movies)


def find_movie_id_by_title(title, release_date):
    """Title (aur agar mile to release year) se TMDB search karke best-match movie_id deta hai."""
    url = f"{BASE_URL}/search/movie"
    params = {"api_key": API_KEY, "query": title, "language": "en-US"}
    response = safe_get(url, params)
    if response is None:
        return None

    results = response.json().get("results", [])
    if not results:
        return None

    year = None
    if isinstance(release_date, str) and len(release_date) >= 4:
        year = release_date[:4]

    if year:
        for r in results:
            if r.get("release_date", "").startswith(year):
                return r.get("id")

    return results[0].get("id")


def migrate_movie_id_if_needed():
    """Purani movies.csv mein movie_id missing ho to ek baar backfill kar deta hai (trailer feature ke liye)."""
    if not os.path.exists("movies.csv"):
        return

    df = pd.read_csv("movies.csv")

    if "movie_id" not in df.columns:
        df["movie_id"] = None

    missing_mask = df["movie_id"].isna()
    missing_rows = df[missing_mask]

    if missing_rows.empty:
        return

    with st.spinner(f"Trailers ke liye {len(missing_rows)} movies update ho rahi hain (ek hi baar hoga)..."):
        with ThreadPoolExecutor(max_workers=20) as executor:
            future_to_index = {
                executor.submit(find_movie_id_by_title, row["title"], row.get("release_date", "")): idx
                for idx, row in missing_rows.iterrows()
            }
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                found_id = future.result()
                # Agar match nahi mila to 0 save karo (NaN nahi) - warna ye row har
                # rerun pe "missing" dikhti rahegi aur baar-baar retry hoti rahegi.
                df.at[idx, "movie_id"] = found_id if found_id else 0

        df["movie_id"] = df["movie_id"].astype("Int64")
        df.to_csv("movies.csv", index=False)


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
    if "budget" in movies.columns and "revenue" in movies.columns:
        # CSV ka purana verdict ignore karke naye rules se dobara nikalo
        movies["verdict"] = movies.apply(
            lambda r: get_verdict(r["budget"], r["revenue"], r.get("release_date", "")), axis=1
        )
    else:
        movies["verdict"] = "N/A"
    if "industry" in movies.columns:
        movies["industry"] = movies["industry"].fillna("Hollywood")
    else:
        movies["industry"] = "Hollywood"

    movies["tags"] = (
        movies["genre"].astype(str)
        + " "
        + movies["keywords"].astype(str)
    )

    tfidf = TfidfVectorizer(stop_words="english")
    vectors = tfidf.fit_transform(movies["tags"])
    similarity = cosine_similarity(vectors)

    return movies, similarity


def get_recommendations(movie, movies, similarity, top_n=15):
    """Movie ka naam leke searched movie + top-N similar movies (rows) return karta hai."""
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
    )[1:top_n + 1]

    searched_movie = movies.iloc[index]
    recommended = [movies.iloc[i] for i, _ in movie_list]

    return searched_movie, recommended


# ==========================================
# PART 3: Streamlit UI
# ==========================================

st.set_page_config(
    page_title="Movie Recommender - Bollywood, Hollywood & South Movies",
    page_icon="🎬",
    layout="wide"
)

# Netflix-jaisa dark theme + red accents + mobile polish + custom fonts ke liye CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Poppins:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background-color: #000000;
    }
    h1, h2, h3, .stMarkdown, label, p {
        color: #e5e5e5 !important;
    }
    h1 {
        color: #e50914 !important;
        font-family: 'Bebas Neue', sans-serif !important;
        font-weight: 400 !important;
        font-size: 3rem !important;
        letter-spacing: 2px;
    }
    h3 {
        font-family: 'Bebas Neue', sans-serif !important;
        letter-spacing: 1px;
        color: #ffffff !important;
    }

    /* Mobile ke liye title aur spacing chhota kar do */
    @media (max-width: 640px) {
        h1 {
            font-size: 1.6rem !important;
        }
        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            padding-top: 1.5rem !important;
        }
        h3 {
            font-size: 1.1rem !important;
        }

        /* Details panel (poster + info) mobile pe thoda chhota */
        .st-key-movie-detail div[data-testid="stHorizontalBlock"] {
            gap: 10px !important;
        }

        .movie-card {
            width: 100px !important;
        }

        .app-logo {
            width: 140px !important;
        }
    }

    div[data-testid="stSelectbox"] > div > div {
        background-color: #2b2b2b;
        color: white;
        border: 1px solid #3d3d3d;
        border-radius: 6px;
    }

    .stButton > button {
        background-color: #e50914;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        padding: 0.6rem 1.5rem;
        width: 100%;
        min-height: 44px;   /* touch-friendly size mobile ke liye */
    }
    .stButton > button:hover {
        background-color: #f6121d;
        color: white;
    }

    div[data-testid="stImage"] img {
        border-radius: 8px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.5);
        transition: transform 0.2s ease;
    }
    div[data-testid="stImage"] img:hover {
        transform: scale(1.05);
    }

    /* ---------- Horizontal-scroll poster row (OTT app jaisi) ---------- */
    .movie-row {
        display: flex;
        overflow-x: auto;
        gap: 10px;
        padding: 4px 2px 14px 2px;
        -webkit-overflow-scrolling: touch;
        scrollbar-width: none;
    }
    .movie-row::-webkit-scrollbar {
        display: none;
    }
    .movie-card {
        flex: 0 0 auto;
        width: 130px;
        cursor: pointer;
    }
    .movie-card a {
        text-decoration: none;
    }
    .movie-card img {
        width: 100%;
        display: block;
        border-radius: 8px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.5);
        transition: transform 0.15s ease;
    }
    .movie-card img:hover {
        transform: scale(1.04);
    }
    .movie-card-title {
        color: #e5e5e5;
        text-align: center;
        font-weight: 500;
        margin-top: 4px;
        font-size: 12px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .rating-badge {
        position: absolute;
        top: 6px;
        right: 6px;
        background: rgba(0,0,0,0.75);
        color: #f5c518;
        font-size: 11px;
        font-weight: 600;
        padding: 2px 6px;
        border-radius: 4px;
    }

    /* ---------- Cast row ---------- */
    .cast-card {
        flex: 0 0 auto;
        width: 84px;
        text-align: center;
    }
    .cast-card img, .cast-card .cast-ph {
        width: 72px;
        height: 72px;
        border-radius: 50%;
        object-fit: cover;
        margin: 0 auto;
        display: block;
        background: #2b2b2b;
    }
    .cast-name {
        color: #e5e5e5;
        font-size: 11px;
        font-weight: 600;
        margin-top: 4px;
        line-height: 1.2;
    }
    .cast-role {
        color: #9a9a9a;
        font-size: 10px;
        line-height: 1.2;
    }

    /* ---------- Where to watch logos ---------- */
    .provider-logo {
        width: 44px;
        height: 44px;
        border-radius: 10px;
        margin-right: 8px;
    }

    /* ---------- Logo: laptop pe bada, mobile pe chhota ---------- */
    .app-logo {
        width: 260px;
        max-width: 100%;
        height: auto;
        display: block;
        margin-bottom: 0.5rem;
    }

    hr {
        border-color: #2b2b2b !important;
    }
</style>
""", unsafe_allow_html=True)

ensure_dataset()
migrate_movie_id_if_needed()
movies, similarity = load_data()

# ---------- Hero Banner (top posters ka collage) ----------
hero_posters = movies["poster_url"].dropna()
hero_posters = hero_posters[hero_posters != ""].head(8).tolist()

if hero_posters:
    hero_images_html = "".join(
        f'<img src="{url}" style="flex:1 1 0; min-width:0; height:100%; width:100%; '
        f'object-fit:cover; opacity:0.55;">'
        for url in hero_posters
    )
    hero_html = (
        f'<div style="position:relative; width:100%; height:150px; overflow:hidden; '
        f'border-radius:10px; margin-bottom:1.2rem;">'
        f'<div style="display:flex; gap:2px; height:100%; width:100%;">{hero_images_html}</div>'
        f'<div style="position:absolute; inset:0; '
        f'background:linear-gradient(180deg, rgba(0,0,0,0.1) 0%, rgba(0,0,0,0.85) 90%);"></div>'
        f'</div>'
    )
    st.markdown(hero_html, unsafe_allow_html=True)



def show_logo():
    """logo.svg dikhata hai (laptop/mobile size CSS se). File na mile to purana text title dikhata hai."""
    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.svg")
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        st.markdown(f'<img class="app-logo" src="data:image/svg+xml;base64,{b64}">', unsafe_allow_html=True)
    else:
        st.title("🎬 Movie Recommender")


show_logo()


def render_movie_row(movies_rows):
    """Poori row ek hi HTML block mein banata hai jo side-ways scroll hoti hai (OTT app jaisi)."""
    cards_html = []

    for movie_row in movies_rows:
        title = movie_row.get("title", "")
        poster_url = movie_row.get("poster_url", "")
        rating = movie_row.get("rating", "")

        if not poster_url:
            continue

        rating_badge = ""
        if rating and str(rating) != "nan":
            try:
                rating_val = float(rating)
                if rating_val > 0:
                    rating_badge = (
                        f'<div class="rating-badge">⭐ {rating_val:.1f}</div>'
                    )
            except (ValueError, TypeError):
                pass

        cards_html.append(
            f'<div class="movie-card">'
            f'<div style="position:relative;">'
            f'<a href="?selected={quote(title)}" target="_self">'
            f'<img src="{poster_url}">'
            f'</a>'
            f'{rating_badge}'
            f'</div>'
            f'<div class="movie-card-title">{title}</div>'
            f'</div>'
        )

    row_html = f'<div class="movie-row">{"".join(cards_html)}</div>'
    st.markdown(row_html, unsafe_allow_html=True)


# ---------- Details panel (jab kisi movie ke poster pe click kiya jaaye) ----------
selected_title = st.query_params.get("selected")

if selected_title:
    selected_title = unquote(selected_title)
    match = movies[movies["title"] == selected_title]

    if not match.empty:
        m = match.iloc[0]
        with st.container():
            st.markdown("### 🎬 " + m.get("title", ""))
            with st.container(key="movie-detail"):
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
                    st.caption("Verdict TMDB ke budget aur collection se nikala gaya hai, kabhi kabhi adhoora ho sakta hai.")
                if m.get("overview"):
                    st.markdown(f"**Overview:** {m['overview']}")

            if m.get("movie_id") and str(m.get("movie_id")) not in ("nan", "0"):
                trailer_url = get_trailer_url(m["movie_id"])
                st.markdown("**🎬 Trailer**")
                if trailer_url:
                    st.video(trailer_url)
                else:
                    st.caption("Is movie ka trailer TMDB pe available nahi hai.")

            extras = get_movie_extras(m["movie_id"]) if m.get("movie_id") and str(m.get("movie_id")) not in ("nan", "0") else None

            if extras:
                # ---------- Where to Watch ----------
                st.markdown("**📺 Where to Watch (India)**")
                if extras["providers"]:
                    logos = "".join(
                        f'<img class="provider-logo" src="{p["logo"]}" title="{html_lib.escape(p["name"])}">'
                        for p in extras["providers"] if p["logo"]
                    )
                    st.markdown(f'<div>{logos}</div>', unsafe_allow_html=True)
                    st.caption(f"{extras['provider_type']} options · Data by JustWatch")
                else:
                    st.caption("India mein abhi kisi platform pe available nahi hai. · Data by JustWatch")

                # ---------- Cast ----------
                if extras["cast"]:
                    st.markdown("**🎭 Cast**")
                    cast_html = ""
                    for c in extras["cast"]:
                        photo = (
                            f'<img src="{c["photo"]}">' if c["photo"]
                            else '<div class="cast-ph"></div>'
                        )
                        cast_html += (
                            f'<div class="cast-card">{photo}'
                            f'<div class="cast-name">{html_lib.escape(c["name"])}</div>'
                            f'<div class="cast-role">{html_lib.escape(c["character"])}</div></div>'
                        )
                    st.markdown(f'<div class="movie-row">{cast_html}</div>', unsafe_allow_html=True)

            if st.button("✕ Close"):
                st.query_params.clear()
                st.rerun()
        st.markdown("---")

# ---------- Search + Recommend (ab sabse upar) ----------
movie_name = st.selectbox(
    "🍿 What are you in the mood to watch? Type a movie below 🎥✨",
    options=sorted(movies["title"].dropna().unique())
)

searched = st.button("🔍 Find Similar Movies")

if searched:
    result = get_recommendations(movie_name, movies, similarity, top_n=15)

    if result is None:
        st.error("Ye movie dataset mein nahi mili!")
    else:
        searched_movie, recommended = result

        st.markdown("#### 🎯 Your Pick")
        render_movie_row([searched_movie])

        st.markdown("---")

        st.markdown("#### ✨ Because You Watched This")
        render_movie_row(recommended)

else:
    # ---------- Genre chips (All / Romance / Drama ...) ----------
    GENRE_CHIPS = ["All", "Action", "Comedy", "Drama", "Romance", "Thriller", "Horror",
                   "Animation", "Crime", "Family", "Fantasy", "Science Fiction"]

    if hasattr(st, "pills"):
        chosen = st.pills("Genre", GENRE_CHIPS, default="All", label_visibility="collapsed")
    else:
        chosen = st.radio("Genre", GENRE_CHIPS, horizontal=True, label_visibility="collapsed")
    chosen = chosen or "All"

    if chosen == "All":
        view = movies
    else:
        view = movies[movies["genre"].str.contains(chosen, case=False, na=False)]

    with_poster = view[view["poster_url"] != ""].copy()
    with_poster["rating_num"] = pd.to_numeric(with_poster["rating"], errors="coerce")
    with_poster["release_dt"] = pd.to_datetime(with_poster["release_date"], errors="coerce")

    # ---------- 1) Top Rated ----------
    top_rated = with_poster.sort_values("rating_num", ascending=False).head(15)

    if not top_rated.empty:
        st.markdown("#### ⭐ Top Rated")
        render_movie_row([row for _, row in top_rated.iterrows()])

    # ---------- 2) New Releases (aaj tak release hui, sabse nayi pehle) ----------
    new_releases = (
        with_poster[with_poster["release_dt"] <= pd.Timestamp.today()]
        .sort_values("release_dt", ascending=False)
        .head(15)
    )

    if not new_releases.empty:
        st.markdown("#### 🆕 New Releases")
        render_movie_row([row for _, row in new_releases.iterrows()])

    # ---------- 3) Bollywood, Hollywood, South ----------
    st.subheader("🔥 Trending Now")

    shown_any = False
    for industry_label, emoji in [("Bollywood", "🇮🇳"), ("Hollywood", "🎥"), ("South", "🎬")]:
        # Pehle poster wali movies chuno, phir top 10 lo (warna bina-poster movies row khali chhod deti hain)
        industry_movies = with_poster[with_poster["industry"] == industry_label].head(10)

        if industry_movies.empty:
            continue

        shown_any = True
        st.markdown(f"#### {emoji} {industry_label}")
        render_movie_row([row for _, row in industry_movies.iterrows()])

    if not shown_any:
        st.info("Is genre ki koi movie nahi mili.")
