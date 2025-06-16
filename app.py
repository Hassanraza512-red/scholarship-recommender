import streamlit as st
import pandas as pd
import pycountry
import nltk

from app.data_utils import load_data
from app.recommender import recommend_scholarships, semantic_recommend, query_ollama_cli

@st.cache_resource
def setup_nltk():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet',  quiet=True)
    return True

@st.cache_data
def get_data():
    return load_data("data/raw/scholarships_mock.csv")

@st.cache_resource
def get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('all-mpnet-base-v2')

_ = setup_nltk()
df = get_data()

st.title("🎓 AIScores Scholarship Recommender")
mode = st.radio("Choose mode:", ["Rule‑based", "Semantic (chatbot)"])

if mode == "Rule‑based":
    field   = st.text_input("Field of study (optional)")
    country = st.text_input("Preferred country (optional)")
    fee_opt = st.selectbox("No application fee?", ["No preference","Yes","No"])
    fee_flag= None if fee_opt=="No preference" else (fee_opt=="No")

    if st.button("Search"):
        results = recommend_scholarships(df, field, country, fee_flag)
        st.dataframe(results[["Title","Provider","Fields_of_Study","Country","Application_Fee"]])

else:
    st.header("Semantic (Chatbot‑style)")

    with st.form(key="quick_form"):
        original = st.text_input("Describe what you want",
                                 value=st.session_state.get("original","e.g. scholarships in the USA"),
                                 key="orig")
        submit_quick = st.form_submit_button("Quick Search")

    if submit_quick:
        st.session_state.original = original

        with st.spinner("Loading model & searching…"):
            _model = get_model()

            inferred = None
            for c in pycountry.countries:
                if c.name.lower() in original.lower():
                    inferred = c.name
                    break

            paraphrase = query_ollama_cli(f"Paraphrase exactly (no extra):\n{original}")

            quick = semantic_recommend(
                df,
                user_query    = paraphrase,
                top_n         = 5,
                user_country  = inferred,
                user_deadline = None,
                fee_pref      = None,
                user_degree   = None
            )

        st.session_state.inferred   = inferred
        st.session_state.paraphrase = paraphrase
        st.session_state.quick      = quick

    if st.session_state.get("quick") is not None:
        st.write("**Paraphrased:**", st.session_state.paraphrase)
        quick = st.session_state.quick
        if quick.empty:
            st.warning("No quick matches found.")
        else:
            st.subheader("Quick Results")
            st.dataframe(quick[["Title","Provider","Country","Similarity"]])

        with st.form(key="refine_form"):
            st.markdown("**Refine your search**")
            field2 = st.text_input("Field of study (optional)", key="f2")
            fee2   = st.selectbox("No application fee?", ["Skip","Yes","No"], key="fee2")
            dl2    = st.date_input("Deadline cutoff (optional)", key="dl2")
            deg2   = st.selectbox("Degree level", ["Skip","Undergraduate","Graduate"], key="deg2")
            submit_refine = st.form_submit_button("Refine")

        if submit_refine:
            inferred   = st.session_state.inferred
            paraphrase = st.session_state.paraphrase
            dl_val  = pd.to_datetime(dl2) if dl2 else None
            fee_val = None if fee2=="Skip" else (fee2=="Yes")
            deg_val = None if deg2=="Skip" else deg2.lower()

            final = semantic_recommend(
                df,
                user_query    = paraphrase,
                top_n         = 5,
                user_country  = inferred,
                user_deadline = dl_val,
                fee_pref      = fee_val,
                user_degree   = deg_val
            )

            if final.empty:
                st.warning("No scholarships found after refinement.")
            else:
                st.subheader("Refined Results")
                st.dataframe(final[["Title","Provider","Country","Similarity","Explanation"]])
