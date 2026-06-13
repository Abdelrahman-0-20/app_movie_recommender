"""
app_movie_recommender.py
========================
Content-Based Movie Recommender — a self-contained Streamlit application.

What this app does
------------------
1.  Procedurally generates a synthetic catalogue of ~200 movies, each with a
    title, genres, a plot summary, an audience rating and a release year.
2.  Cleans the catalogue with several clearly commented steps.
3.  Builds a content-based recommender: TF-IDF vectors over the combined
    plot + genre text, compared with cosine similarity (scikit-learn).
4.  Lets the user pick a movie in the sidebar and returns the 5 most similar
    titles, complete with placeholder poster art.
5.  Visualises the catalogue with Plotly (genre mix, rating spread, year
    histogram) and shows the similarity scores of the recommendations.

Run it with:
    streamlit run app_movie_recommender.py

Only the following libraries are used: streamlit, pandas, numpy, scikit-learn,
plotly.
"""

from __future__ import annotations

from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------------------------------- #
# Page configuration                                                          #
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
)

# Building blocks for procedurally generated, genre-consistent plot summaries.
GENRES = [
    "Action", "Comedy", "Drama", "Sci-Fi", "Horror",
    "Romance", "Thriller", "Fantasy", "Adventure", "Mystery",
]

# Genre-specific vocabulary makes similar movies share words -> higher TF-IDF
# similarity, which is exactly the signal the recommender exploits.
GENRE_PLOT_WORDS = {
    "Action": ["explosive chase", "elite soldier", "high-stakes mission", "revenge", "combat"],
    "Comedy": ["hilarious misunderstanding", "awkward romance", "wacky friends", "chaos", "laughter"],
    "Drama": ["family secret", "personal struggle", "emotional journey", "redemption", "loss"],
    "Sci-Fi": ["distant galaxy", "artificial intelligence", "time travel", "alien", "future"],
    "Horror": ["haunted house", "vengeful spirit", "isolated cabin", "nightmare", "survival"],
    "Romance": ["unexpected love", "second chance", "heartbreak", "wedding", "soulmate"],
    "Thriller": ["cat-and-mouse", "hidden conspiracy", "ticking clock", "betrayal", "manhunt"],
    "Fantasy": ["ancient prophecy", "magical kingdom", "dragon", "chosen hero", "spell"],
    "Adventure": ["lost treasure", "perilous expedition", "uncharted island", "quest", "explorer"],
    "Mystery": ["baffling murder", "clever detective", "hidden clue", "small town", "investigation"],
}

TITLE_ADJECTIVES = [
    "Last", "Silent", "Broken", "Eternal", "Hidden", "Crimson", "Lost", "Final",
    "Golden", "Frozen", "Burning", "Secret", "Dark", "Bright", "Wild", "Quiet",
]
TITLE_NOUNS = [
    "Horizon", "Promise", "Empire", "Shadow", "Dream", "Legacy", "Voyage",
    "Echo", "Kingdom", "Storm", "Mirror", "Journey", "Paradox", "Garden",
    "Throne", "Whisper", "Dawn", "Requiem",
]


# --------------------------------------------------------------------------- #
# 1. Synthetic data generation                                                #
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Generating synthetic movie catalogue…")
def generate_movie_data(n_movies: int = 200, seed: int = 21) -> pd.DataFrame:
    """Create a synthetic movie catalogue.

    Titles are assembled from adjective/noun pools; plot summaries are stitched
    together from genre-specific keyword banks so that movies sharing genres
    naturally share vocabulary (a clean signal for TF-IDF + cosine similarity).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_movies):
        # 1–2 genres per movie.
        n_g = rng.integers(1, 3)
        movie_genres = list(rng.choice(GENRES, size=n_g, replace=False))

        # Title (kept unique by appending the index when needed downstream).
        title = f"{rng.choice(TITLE_ADJECTIVES)} {rng.choice(TITLE_NOUNS)}"

        # Plot summary drawn from each chosen genre's vocabulary.
        plot_bits = []
        for g in movie_genres:
            plot_bits.extend(rng.choice(GENRE_PLOT_WORDS[g], size=2, replace=False))
        rng.shuffle(plot_bits)
        plot = (
            f"A story of {plot_bits[0]} where the protagonist faces "
            f"{plot_bits[1]} and must overcome {plot_bits[-1]}."
        )

        rows.append(
            {
                "movie_id": i,
                "title": title,
                "genres": ", ".join(movie_genres),
                "plot": plot,
                # Ratings cluster around 6.5 with realistic spread.
                "rating": round(float(np.clip(rng.normal(6.5, 1.3), 1.0, 10.0)), 1),
                "year": int(rng.integers(1980, 2025)),
                "runtime_min": int(rng.integers(80, 180)),
            }
        )

    data = pd.DataFrame(rows)

    # --- Inject "dirtiness" so cleaning steps matter ------------------------
    # (a) Duplicate a handful of movies outright.
    data = pd.concat([data, data.sample(12, random_state=seed)], ignore_index=True)
    # (b) Blank out a few plot summaries.
    blank_idx = rng.choice(len(data), size=8, replace=False)
    data.loc[blank_idx, "plot"] = ""
    # (c) Corrupt a few ratings with out-of-range values.
    bad_idx = rng.choice(len(data), size=6, replace=False)
    data.loc[bad_idx, "rating"] = rng.choice([-3.0, 99.0, 0.0], size=6)

    return data


# --------------------------------------------------------------------------- #
# 2. Data cleaning                                                            #
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Cleaning catalogue…")
def clean_movie_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean the movie catalogue with transparent, commented steps.

    Cleaning steps:
      1. Drop duplicate movies (same title + year).
      2. Remove rows with empty plot summaries (needed for content filtering).
      3. Clip ratings into the valid 1–10 range.
      4. Make every title unique so the sidebar selector is unambiguous.
    """
    df = raw.copy()

    # --- Cleaning step 1: remove duplicate titles ---------------------------
    df = df.drop_duplicates(subset=["title", "year"]).reset_index(drop=True)

    # --- Cleaning step 2: drop empty plots ----------------------------------
    # The recommender relies entirely on the plot text, so blanks are useless.
    df = df[df["plot"].str.strip().astype(bool)].reset_index(drop=True)

    # --- Cleaning step 3: clip out-of-range ratings -------------------------
    df["rating"] = df["rating"].clip(lower=1.0, upper=10.0)

    # --- Cleaning step 4: de-duplicate display titles -----------------------
    # Two different movies may have rolled the same title; append the year.
    dup_mask = df["title"].duplicated(keep=False)
    df.loc[dup_mask, "title"] = df.loc[dup_mask, "title"] + " (" + df.loc[dup_mask, "year"].astype(str) + ")"

    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 3. Recommender model                                                        #
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Building similarity matrix…")
def build_similarity(df: pd.DataFrame) -> np.ndarray:
    """Compute the pairwise cosine-similarity matrix from TF-IDF features.

    Genres are appended to the plot text so the model blends thematic words
    with genre membership. English stop-words are removed and unigrams plus
    bigrams are used to capture short phrases such as "time travel".
    """
    corpus = (df["plot"] + " " + df["genres"].str.replace(",", " ")).tolist()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Cosine similarity between every pair of movies.
    return cosine_similarity(tfidf_matrix)


def recommend(
    df: pd.DataFrame, similarity: np.ndarray, title: str, top_n: int = 5
) -> pd.DataFrame:
    """Return the ``top_n`` movies most similar to ``title``.

    The selected movie itself is excluded from its own recommendation list.
    """
    idx = df.index[df["title"] == title][0]
    scores = list(enumerate(similarity[idx]))
    # Sort by similarity descending, skip the movie itself (position 0).
    scores = sorted(scores, key=lambda pair: pair[1], reverse=True)
    top = [s for s in scores if s[0] != idx][:top_n]

    rec = df.iloc[[i for i, _ in top]].copy()
    rec["similarity"] = [round(score, 3) for _, score in top]
    return rec


# --------------------------------------------------------------------------- #
# 4. Plotly helpers                                                           #
# --------------------------------------------------------------------------- #
def plot_genre_distribution(df: pd.DataFrame) -> go.Figure:
    """Bar chart counting how often each genre appears across the catalogue."""
    exploded = (
        df["genres"].str.split(", ").explode().value_counts().reset_index()
    )
    exploded.columns = ["genre", "count"]
    fig = px.bar(
        exploded, x="genre", y="count", color="count",
        color_continuous_scale="Magma",
        labels={"genre": "Genre", "count": "Number of movies"},
    )
    fig.update_layout(title="Genre Distribution", coloraxis_showscale=False)
    return fig


def plot_rating_distribution(df: pd.DataFrame) -> go.Figure:
    """Histogram of audience ratings."""
    fig = px.histogram(
        df, x="rating", nbins=20, color_discrete_sequence=["#7c3aed"],
        labels={"rating": "Rating (1–10)"},
    )
    fig.update_layout(title="Rating Distribution", yaxis_title="Count")
    return fig


def plot_year_distribution(df: pd.DataFrame) -> go.Figure:
    """Histogram of release years (release decades become visually obvious)."""
    fig = px.histogram(
        df, x="year", nbins=20, color_discrete_sequence=["#0891b2"],
        labels={"year": "Release year"},
    )
    fig.update_layout(title="Movies by Release Year", yaxis_title="Count")
    return fig


def plot_similarity_scores(rec: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of the recommendation similarity scores."""
    ordered = rec.iloc[::-1]  # reverse so the best score sits at the top
    fig = px.bar(
        ordered, x="similarity", y="title", orientation="h",
        color="similarity", color_continuous_scale="Tealgrn",
        labels={"similarity": "Cosine similarity", "title": ""},
    )
    fig.update_layout(title="Recommendation Similarity Scores", coloraxis_showscale=False)
    return fig


def poster_url(title: str) -> str:
    """Build a placeholder poster URL for a movie title.

    Uses placehold.co (a reliable placeholder image service). The title is URL
    encoded so spaces and punctuation are handled correctly.
    """
    label = quote_plus(title)
    return f"https://placehold.co/200x300/1e293b/ffffff/png?text={label}"


# --------------------------------------------------------------------------- #
# 5. Sidebar controls                                                         #
# --------------------------------------------------------------------------- #
def build_sidebar(df: pd.DataFrame) -> dict:
    """Render the sidebar selectors and return the chosen values."""
    st.sidebar.header("🎯 Find Recommendations")
    selected_movie = st.sidebar.selectbox(
        "Pick a movie you like", sorted(df["title"].tolist())
    )
    top_n = st.sidebar.slider("Number of recommendations", 3, 10, 5)

    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Catalogue Filters")
    min_rating = st.sidebar.slider("Minimum rating to display", 1.0, 10.0, 1.0, 0.5)
    genre_filter = st.sidebar.multiselect(
        "Highlight genres", options=GENRES, default=[]
    )
    return {
        "selected_movie": selected_movie,
        "top_n": top_n,
        "min_rating": min_rating,
        "genre_filter": genre_filter,
    }


# --------------------------------------------------------------------------- #
# 6. Main application body                                                     #
# --------------------------------------------------------------------------- #
def main() -> None:
    """Assemble the recommender page."""
    st.title("🎬 Content-Based Movie Recommender")
    st.markdown(
        "Pick a movie and discover similar titles using **TF-IDF** text "
        "features and **cosine similarity** over synthetic plot summaries."
    )

    raw = generate_movie_data()
    df = clean_movie_data(raw)
    similarity = build_similarity(df)
    controls = build_sidebar(df)

    # --- Headline metrics ----------------------------------------------------
    c1, c2, c3 = st.columns(3)
    c1.metric("Movies in catalogue", f"{len(df):,}")
    c2.metric("Average rating", f"{df['rating'].mean():.1f}")
    c3.metric("Year span", f"{df['year'].min()}–{df['year'].max()}")

    st.markdown("---")

    # --- Recommendations -----------------------------------------------------
    selected = controls["selected_movie"]
    seed_row = df[df["title"] == selected].iloc[0]
    st.subheader(f"⭐ Because you picked *{selected}*")
    st.caption(
        f"Genres: {seed_row['genres']} · Rating: {seed_row['rating']} · "
        f"Year: {seed_row['year']}"
    )
    st.info(seed_row["plot"])

    rec = recommend(df, similarity, selected, controls["top_n"])

    st.subheader("🍿 You might also like")
    # Show the recommendations as a row of poster cards.
    poster_cols = st.columns(len(rec))
    for col, (_, movie) in zip(poster_cols, rec.iterrows()):
        with col:
            st.image(poster_url(movie["title"]), use_container_width=True)
            st.markdown(f"**{movie['title']}**")
            st.caption(
                f"{movie['genres']}\n\n⭐ {movie['rating']} · {movie['year']}\n\n"
                f"Similarity: {movie['similarity']:.2f}"
            )

    st.plotly_chart(plot_similarity_scores(rec), use_container_width=True)

    st.markdown("---")

    # --- Catalogue insights --------------------------------------------------
    st.subheader("📊 Catalogue Insights")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(plot_genre_distribution(df), use_container_width=True)
    with right:
        st.plotly_chart(plot_rating_distribution(df), use_container_width=True)
    st.plotly_chart(plot_year_distribution(df), use_container_width=True)

    # --- Filtered catalogue table -------------------------------------------
    with st.expander("🔎 Browse the catalogue"):
        view = df[df["rating"] >= controls["min_rating"]]
        if controls["genre_filter"]:
            pattern = "|".join(controls["genre_filter"])
            view = view[view["genres"].str.contains(pattern)]
        st.dataframe(
            view[["title", "genres", "rating", "year", "runtime_min"]]
            .sort_values("rating", ascending=False)
            .reset_index(drop=True),
            use_container_width=True,
        )
        st.caption(f"{len(view):,} movies match the current filters.")


if __name__ == "__main__":
    main()
