
import streamlit as st
from sentence_transformers import SentenceTransformer
from supabase import create_client

# Page configuration
st.set_page_config(
    page_title="College Notes AI Search",
    page_icon="📚",
    layout="wide"
)

# Load AI model and database connection
@st.cache_resource
def initialize():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    supabase = create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"]
    )

    return model, supabase


model, supabase = initialize()

# Semantic search function
def semantic_search(query, match_count=5):

    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    response = supabase.rpc(
        "match_notes",
        {
            "query_embedding": query_embedding,
            "match_count": match_count
        }
    ).execute()

    return response.data


# Website interface
st.title("📚 College Notes AI Search")

st.write(
    "Find relevant college notes using artificial intelligence."
)

query = st.text_input(
    "Enter your question",
    placeholder="Example: What is normalization in DBMS?"
)

if st.button("🔍 Search", type="primary"):

    if query.strip():

        with st.spinner("Searching your notes..."):

            try:
                results = semantic_search(query)

                if results:
                    st.success(f"Found {len(results)} relevant notes!")

                    for i, result in enumerate(results, start=1):

                        with st.container(border=True):

                            st.subheader(
                                f"{i}. {result['topic']}"
                            )

                            st.caption(
                                f"Subject: {result['subject']} | "
                                f"Note type: {result['note_type']}"
                            )

                            st.write(result["paragraph"])

                            st.progress(
                                max(0, min(1, result["similarity"]))
                            )

                            st.caption(
                                f"Similarity: {result['similarity']:.4f}"
                            )

                else:
                    st.info("No relevant notes found.")

            except Exception as e:
                st.error(f"Something went wrong: {e}")

    else:
        st.warning("Please enter a question.")
