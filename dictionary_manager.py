import streamlit as st
import pandas as pd
import os
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

# Set page config
st.set_page_config(
    page_title="Medumba-French Dictionary",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Medumba-French Dictionary Manager")
st.markdown("---")

# Define paths
output_dir = Path(__file__).parent / "output"
words_csv = output_dir / "translations_words.csv"
expressions_csv = output_dir / "translations_expressions.csv"

# Initialize session state
if 'words_df' not in st.session_state:
    st.session_state.words_df = pd.read_csv(words_csv, encoding='utf-8')
if 'expressions_df' not in st.session_state:
    st.session_state.expressions_df = pd.read_csv(expressions_csv, encoding='utf-8')

def save_dataframes():
    """Save dataframes back to CSV files"""
    st.session_state.words_df.to_csv(words_csv, index=False, encoding='utf-8')
    st.session_state.expressions_df.to_csv(expressions_csv, index=False, encoding='utf-8')

def search_translations(query, dict_type, search_in):
    """Search for translations in the specified dictionary"""
    if dict_type == "Words":
        df = st.session_state.words_df
    else:
        df = st.session_state.expressions_df
    
    query_lower = query.lower()
    
    if search_in == "Medumba":
        results = df[df['Medumba'].str.lower().str.contains(query_lower, na=False, regex=False)]
    elif search_in == "French":
        results = df[df['French'].str.lower().str.contains(query_lower, na=False, regex=False)]
    else:  # Both
        results = df[
            (df['Medumba'].str.lower().str.contains(query_lower, na=False, regex=False)) |
            (df['French'].str.lower().str.contains(query_lower, na=False, regex=False))
        ]
    
    return results

def delete_translation(medumba, french, dict_type):
    """Delete a translation from the dictionary"""
    if dict_type == "Words":
        df = st.session_state.words_df
    else:
        df = st.session_state.expressions_df
    
    # Find and remove the exact match
    mask = (df['Medumba'].str.lower() == medumba.lower()) & (df['French'].str.lower() == french.lower())
    df_updated = df[~mask]
    
    if len(df_updated) == len(df):
        return False, "❌ Translation not found"
    
    if dict_type == "Words":
        st.session_state.words_df = df_updated.reset_index(drop=True)
    else:
        st.session_state.expressions_df = df_updated.reset_index(drop=True)
    
    save_dataframes()
    return True, f"✅ Deleted: '{medumba}' → '{french}'"

def add_translation(medumba, french, dict_type):
    """Add a new translation with validation"""
    # Validate inputs
    if not medumba.strip():
        return False, "❌ Medumba field cannot be empty"
    if not french.strip():
        return False, "❌ French field cannot be empty"
    
    medumba = medumba.strip()
    french = french.strip()
    
    if dict_type == "Words":
        df = st.session_state.words_df
    else:
        df = st.session_state.expressions_df
    
    # Check if exact translation already exists
    exact_match = df[
        (df['Medumba'].str.lower() == medumba.lower()) &
        (df['French'].str.lower() == french.lower())
    ]
    
    if not exact_match.empty:
        return False, "⚠️ This translation already exists in the database"
    
    # Check if medumba exists with different french translations
    medumba_matches = df[df['Medumba'].str.lower() == medumba.lower()]
    
    if not medumba_matches.empty:
        # Add to existing medumba
        # We need to handle multiple translations - for now we'll add as a new row
        new_row = pd.DataFrame({'Medumba': [medumba], 'French': [french]})
        st.session_state.words_df if dict_type == "Words" else st.session_state.expressions_df
        df_to_update = st.session_state.words_df if dict_type == "Words" else st.session_state.expressions_df
        df_to_update = pd.concat([df_to_update, new_row], ignore_index=True)
        if dict_type == "Words":
            st.session_state.words_df = df_to_update
        else:
            st.session_state.expressions_df = df_to_update
        save_dataframes()
        return True, f"✅ Added new French translation '{french}' to existing Medumba '{medumba}'"
    
    # Check if french exists with different medumba translations
    french_matches = df[df['French'].str.lower() == french.lower()]
    
    if not french_matches.empty:
        # Add to existing french
        new_row = pd.DataFrame({'Medumba': [medumba], 'French': [french]})
        df_to_update = st.session_state.words_df if dict_type == "Words" else st.session_state.expressions_df
        df_to_update = pd.concat([df_to_update, new_row], ignore_index=True)
        if dict_type == "Words":
            st.session_state.words_df = df_to_update
        else:
            st.session_state.expressions_df = df_to_update
        save_dataframes()
        return True, f"✅ Added new Medumba translation '{medumba}' to existing French '{french}'"
    
    # New translation pair
    new_row = pd.DataFrame({'Medumba': [medumba], 'French': [french]})
    df_to_update = st.session_state.words_df if dict_type == "Words" else st.session_state.expressions_df
    df_to_update = pd.concat([df_to_update, new_row], ignore_index=True)
    if dict_type == "Words":
        st.session_state.words_df = df_to_update
    else:
        st.session_state.expressions_df = df_to_update
    save_dataframes()
    return True, f"✅ Successfully added new translation: '{medumba}' → '{french}'"

# Create tabs
@st.cache_resource
def load_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


@st.cache_data
def build_embeddings(words_df, expressions_df):
    model = load_model()
    med_words = list(words_df["Medumba"].astype(str)) + list(expressions_df["Medumba"].astype(str))
    fr_words = list(words_df["French"].astype(str)) + list(expressions_df["French"].astype(str))
    med_emb = model.encode(med_words, normalize_embeddings=True)
    fr_emb = model.encode(fr_words, normalize_embeddings=True)
    return med_words, fr_words, med_emb, fr_emb


def nearest(query_embedding, target_embeddings, threshold=0.6):
    scores = np.dot(target_embeddings, query_embedding)
    idx = int(np.argmax(scores))
    best = float(scores[idx])
    if best >= threshold:
        return idx, best
    return None, None


def translate_text(sentence, words_df, expressions_df, med_to_fr=True, threshold=0.6):
    med_words, fr_words, med_emb, fr_emb = build_embeddings(words_df, expressions_df)
    lookup_embeddings = med_emb if med_to_fr else fr_emb
    target_texts = fr_words if med_to_fr else med_words
    model = load_model()

    tokens = sentence.split()
    outputs = []
    scores = []
    for token in tokens:
        emb = model.encode(token, normalize_embeddings=True)
        idx, score = nearest(emb, lookup_embeddings, threshold=threshold)
        if idx is not None:
            outputs.append(target_texts[idx])
            scores.append(score)
        else:
            outputs.append("[UNK]")
            scores.append(0.0)

    return " ".join(outputs), list(zip(tokens, outputs, scores))


tab1, tab3, tab4 = st.tabs(["🔍 Search Dictionary", "📊 View Database", "🤖 Translation Model"])

# TAB 1: Search
with tab1:
    st.subheader("Search Translations")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        dict_type = st.radio("Select Dictionary", ["Words", "Expressions"])
    with col2:
        search_in = st.radio("Search in", ["Medumba", "French", "Both"])
    
    search_query = st.text_input("Enter your search query", placeholder="Type a word or expression...")
    
    if search_query:
        results = search_translations(search_query, dict_type, search_in)
        
        if not results.empty:
            st.success(f"Found {len(results)} result(s)")
            st.dataframe(results, use_container_width=True, hide_index=True)
            
            st.markdown("**Delete a result:**")
            col1, col2, col3 = st.columns([1, 1, 1])
            
            for idx, row in results.iterrows():
                with col1:
                    st.write(f"**{row['Medumba']}**")
                with col2:
                    st.write(f"→ {row['French']}")
                with col3:
                    if st.button("🗑️ Delete", key=f"del_{idx}_{row['Medumba']}_{row['French']}"):
                        success, message = delete_translation(row['Medumba'], row['French'], dict_type)
                        if success:
                            st.success(message)
                            # Reload dataframes
                            st.session_state.words_df = pd.read_csv(words_csv, encoding='utf-8')
                            st.session_state.expressions_df = pd.read_csv(expressions_csv, encoding='utf-8')
                            st.rerun()
                        else:
                            st.error(message)
        else:
            st.info("No translations found. Try adding it!")
    else:
        st.info(f"👆 Enter a search query to find translations in the {dict_type.lower()} dictionary")

# TAB 2: View Full Database
with tab3:
    st.subheader("Database Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Words Dictionary")
        st.write(f"Total entries: **{len(st.session_state.words_df)}**")
        with st.expander("View all words"):
            st.dataframe(st.session_state.words_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.write("### Expressions Dictionary")
        st.write(f"Total entries: **{len(st.session_state.expressions_df)}**")
        with st.expander("View all expressions"):
            st.dataframe(st.session_state.expressions_df, use_container_width=True, hide_index=True)

# TAB 3: Translation Model
with tab4:
    st.subheader("Try the Translation Model")

    med_to_fr = st.radio("Direction", ["Medumba → French", "French → Medumba"], horizontal=True)
    threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.6, 0.01)
    sentence = st.text_area("Enter a sentence", placeholder="Type a sentence to translate...", height=120)

    if st.button("Translate", type="primary"):
        if not sentence.strip():
            st.warning("Please enter a sentence first.")
        else:
            with st.spinner("Translating..."):
                output_text, details = translate_text(
                    sentence,
                    st.session_state.words_df,
                    st.session_state.expressions_df,
                    med_to_fr=(med_to_fr == "Medumba → French"),
                    threshold=threshold,
                )
            st.success("Translation complete")
            st.text_area("Output", value=output_text, height=100)

            if details:
                st.write("Token details (input → output, similarity score):")
                detail_df = pd.DataFrame(details, columns=["Input token", "Output", "Score"])
                st.dataframe(detail_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("Made with ❤️ for Medumba-French translations")
