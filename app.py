import streamlit as st
import pandas as pd
import pickle
import requests
import time
import concurrent.futures
from io import BytesIO
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# ─── Page Config ───────────────────────────────────────────────
st.set_page_config(page_title="Movie Recommender", layout="wide")

# ─── CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    div.stButton > button {
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%) !important;
        color: white !important; border: none !important;
        border-radius: 20px !important; padding: 0.5rem 2.2rem !important;
        font-weight: 600 !important; transition: all 0.3s ease !important;
    }
    div.stButton > button:hover { transform: scale(1.05) !important; }
    [data-testid="column"] {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 14px !important; padding: 14px !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25) !important;
    }
    [data-testid="column"] img { border-radius: 8px !important; }
    .movie-title {
        text-align: center; font-weight: 700; font-size: 0.9rem;
        height: 44px; display: flex; align-items: center;
        justify-content: center; margin: 6px 0 10px 0;
        color: #f0f0f0;
    }
</style>
""", unsafe_allow_html=True)

# ─── Reliable HTTP Session (retry + keep-alive) ─────────────────
API_KEY = '8265bd1679663a7ea12ac168da84d2e8'
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

def make_session():
    s = requests.Session()
    retry = Retry(total=4, backoff_factor=0.3,
                  status_forcelist=[429, 500, 502, 503, 504],
                  raise_on_status=False)
    adapter = HTTPAdapter(pool_connections=10, pool_maxsize=10, max_retries=retry)
    s.mount('https://', adapter)
    s.mount('http://', adapter)
    return s

api_session = make_session()    # for TMDB metadata calls
img_session = make_session()    # separate pool for image downloads

FALLBACK_POSTER = "https://images.unsplash.com/photo-1485846234645-a62644f84728?w=500&q=80"

# ─── Fetch image as BYTES (server-side) ─────────────────────────
# st.image() accepts raw bytes → browser never touches TMDB directly
@st.cache_data(show_spinner=False)
def fetch_poster_bytes(movie_id: int, index: int):
    """Return JPEG/PNG bytes of the poster, or a fallback URL string."""
    time.sleep(index * 0.12)   # gentle stagger to avoid burst resets

    # Step 1 – get poster path from TMDB API
    poster_path = None
    for attempt in range(3):
        try:
            r = api_session.get(
                f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}',
                headers=HEADERS, timeout=6
            )
            if r.status_code == 200:
                poster_path = r.json().get('poster_path')
                break
            if r.status_code == 401:
                break    # invalid key, no point retrying
        except Exception:
            time.sleep(0.3 * (attempt + 1))
            continue

    if not poster_path:
        return FALLBACK_POSTER

    # Step 2 – download the actual image bytes (server-side)
    # Try multiple image hosts so we're never stuck if one is blocked
    image_urls = [
        f"https://image.tmdb.org/t/p/w500{poster_path}",
        f"https://images.weserv.nl/?url=https://image.tmdb.org/t/p/w500{poster_path}",
        f"https://wsrv.nl/?url=https://image.tmdb.org/t/p/w500{poster_path}",
    ]

    for img_url in image_urls:
        for attempt in range(2):
            try:
                img_r = img_session.get(img_url, headers=HEADERS, timeout=8)
                if img_r.status_code == 200 and img_r.content:
                    return img_r.content   # raw bytes
            except Exception:
                time.sleep(0.2)
                continue

    # All image sources failed → return fallback URL
    return FALLBACK_POSTER


def fetch_poster_task(args):
    return fetch_poster_bytes(*args)


# ─── Load cached data ────────────────────────────────────────────
@st.cache_resource
def load_data():
    movies = pd.DataFrame(pickle.load(open('movie_dict.pkl', 'rb')))
    similarity = pickle.load(open('similarity.pkl', 'rb'))
    return movies, similarity

movies, similarity = load_data()


# ─── Recommend ───────────────────────────────────────────────────
@st.cache_data(show_spinner="Finding recommendations…")
def recommend(movie: str):
    idx = movies[movies['title'] == movie].index[0]
    top5 = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])[1:6]
    titles = [movies.iloc[i[0]].title for i in top5]
    ids    = [int(movies.iloc[i[0]].movie_id) for i in top5]
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as exe:
        posters = list(exe.map(fetch_poster_task, [(mid, pos) for pos, mid in enumerate(ids)]))
    return titles, posters


# ─── UI ──────────────────────────────────────────────────────────
st.title("🎬 Movie Recommender")
selected = st.selectbox("Select a movie you love:", movies['title'].values)

if st.button("Recommend"):
    names, posters = recommend(selected)
    st.write("### More Like This:")
    cols = st.columns(5)
    for i, col in enumerate(cols):
        with col:
            st.markdown(f'<div class="movie-title">{names[i]}</div>', unsafe_allow_html=True)
            st.image(posters[i], use_container_width=True)
