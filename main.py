import nltk
import pandas as pd
import subprocess
import pycountry
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import os

# Initialize NLP components
nltk.download('stopwords')
nltk.download('wordnet')
model = SentenceTransformer('all-mpnet-base-v2')
stop_words = set(stopwords.words('english'))
punct_table = str.maketrans('', '', string.punctuation)
lemmatizer = WordNetLemmatizer()


def query_ollama_cli(prompt: str, model: str = "llama3") -> str:
    try:
        cmd = ['ollama', 'run', model, '--message', '--prompt', prompt]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception as e:
        print(f"⚠️ Ollama CLI call failed: {e}")
        return prompt


def extract_keywords(text):
    text = text.lower().translate(punct_table)
    tokens = [w for w in text.split() if w not in stop_words and len(w) > 2]
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    return set(lemmas)


def parse_eligibility(text):
    text = text.lower()
    return {
        "is_female": any(word in text for word in ['female', 'woman', 'women']),
        "is_undergraduate": 'undergraduate' in text,
        "is_graduate": 'graduate' in text and 'undergraduate' not in text,
        "is_international": 'international' in text,
        "is_disabled": 'disab' in text,
        "is_minority": 'minority' in text,
    }


def expand_with_synonyms(keywords):
    expanded = set(keywords)
    for word in keywords:
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                expanded.add(lemma.name().lower().replace('_', ' '))
    return expanded


def generate_explanation(user_query, scholarship_row, user_country=None, user_deadline=None, fee_pref=None, user_degree=None):
    explanations = []

    # Degree match explanation
    if user_degree:
        degree_keywords = {
            'undergraduate': ['undergraduate', 'bachelor'],
            'graduate': ['graduate', 'master', 'phd', 'doctorate']
        }
        keywords = degree_keywords.get(user_degree.lower(), [])
        eligibility_text = str(scholarship_row.get('Eligibility_Criteria', '')).lower()
        if any(kw in eligibility_text for kw in keywords):
            explanations.append(f"matches your degree level: {user_degree}")

    # Field of study explanation
    user_query_lower = user_query.lower()
    fields = str(scholarship_row.get('Fields_of_Study', '')).split(',')
    for field in fields:
        if field.strip().lower() in user_query_lower:
            explanations.append(f"because you're interested in {field.strip()}")
            break

    # Country explanation
    if user_country and user_country.lower() in str(scholarship_row.get('Country', '')).lower():
        explanations.append(f"available in your preferred country: {scholarship_row['Country']}")

    # Fee preference explanation
    if fee_pref and isinstance(fee_pref, bool) and fee_pref and str(scholarship_row.get('Application_Fee', '')).strip().lower() == 'no':
        explanations.append("no application fee, as you requested")

    # Deadline explanation
    if user_deadline:
        try:
            deadline = pd.to_datetime(scholarship_row.get('Deadline'), errors='coerce')
            if pd.notnull(deadline) and deadline >= user_deadline:
                explanations.append("meets your deadline preference")
        except Exception:
            pass

    # If no specific explanations, fallback
    if not explanations:
        return "This scholarship may match your interests based on general similarity."

    return "This scholarship is recommended " + " and ".join(explanations) + "."


def recommend_scholarships(df, field_of_study=None, country=None, application_fee=None):
    filtered = df.copy()
    if field_of_study:
        filtered = filtered[filtered['Fields_of_Study'].str.contains(field_of_study, case=False, na=False)]
    if country:
        filtered = filtered[filtered['Country'].str.contains(country, case=False, na=False)]
    if application_fee is not None:
        expected_fee_value = "no" if application_fee else "yes"
        filtered = filtered[filtered['Application_Fee'].str.lower().str.strip() == expected_fee_value]
    return filtered


def rerank_scholarships(df, user_country=None, user_deadline=None):
    df = df.copy()
    df['Rerank_Score'] = df['Similarity']

    if user_country:
        df['Country_Match'] = df['Country'].str.contains(user_country, case=False, na=False)
        df.loc[df['Country_Match'], 'Rerank_Score'] += 0.1

    if user_deadline:
        df['Deadline'] = pd.to_datetime(df['Deadline'], errors='coerce')
        df.loc[df['Deadline'] >= user_deadline, 'Rerank_Score'] += 0.1

    df = df.sort_values(by='Rerank_Score', ascending=False)
    return df.drop(columns=['Country_Match'], errors='ignore')


def semantic_recommend(df, user_query, top_n=5, user_country=None, user_deadline=None, fee_pref=None, user_degree=None):
    # Strict country filter
    if user_country:
        pattern = fr"\b{user_country}\b"
        df = df[df['Country'].str.contains(pattern, case=False, na=False, regex=True)]
        if df.empty:
            return df

    text_corpus = df['Description'].fillna('') + " " + df['Eligibility_Criteria'].fillna('')
    embeddings = model.encode(text_corpus.tolist(), convert_to_tensor=True)
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    scores = cosine_similarity(
        query_embedding.cpu().numpy().reshape(1, -1),
        embeddings.cpu().numpy()
    )[0]

    df = df.copy()
    df['Similarity'] = scores

    # Degree bias
    if user_degree:
        degree_keywords = {
            'undergraduate': ['undergraduate', 'bachelor'],
            'graduate': ['graduate', 'master', 'phd', 'doctorate']
        }
        keywords = degree_keywords.get(user_degree.lower(), [])
        df['Degree_Match'] = df['Eligibility_Criteria'].fillna('').str.lower().apply(
            lambda text: any(kw in text for kw in keywords)
        )
        df.loc[df['Degree_Match'], 'Similarity'] += 0.1

    df = rerank_scholarships(df, user_country=user_country, user_deadline=user_deadline)

    top_matches = df.head(top_n).copy()
    top_matches['Explanation'] = top_matches.apply(
        lambda row: generate_explanation(
            user_query=user_query,
            scholarship_row=row,
            user_country=user_country,
            user_deadline=user_deadline,
            fee_pref=fee_pref,
            user_degree=user_degree
        ), axis=1
    )

    return top_matches.reset_index(drop=True)


def load_data(path='data/raw/scholarships_mock.csv'):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path, encoding='latin1')
    expected_columns = {
        'Title', 'Provider', 'Description', 'Eligibility_Criteria',
        'Fields_of_Study', 'Amount', 'Deadline', 'Country', 'Application_Fee'
    }
    if not expected_columns.issubset(df.columns):
        raise ValueError(f"CSV is missing expected columns. Found: {df.columns.tolist()}")
    print(f"✅ Loaded data with shape: {df.shape}")
    return df


def get_user_profile():
    print("\n🤖 Please answer a few questions:")
    degree = input("Level of study? (Bachelor's/Master's/PhD or skip): ").strip().lower() or None
    field = input("Field of study? (e.g., Engineering) or skip: ").strip().lower() or None
    fee_ans = input("No‑fee only? (Yes/No or skip): ").strip().lower()
    fee_pref = True if fee_ans == "yes" else False if fee_ans == "no" else None
    country_input = input("Preferred country? or skip: ").strip() or None
    deadline_input = input("Deadline cutoff (YYYY-MM-DD) or skip: ").strip()
    deadline = None
    if deadline_input:
        try:
            deadline = pd.to_datetime(deadline_input)
        except:
            print("⚠️ Invalid date format, ignoring deadline.")
    return {
        "degree": degree,
        "field": field,
        "no_fee": fee_pref,
        "country": country_input,
        "deadline": deadline
    }


def main():
    print("📦 Loading scholarship data...")
    df = load_data()
    if df.empty:
        print("❌ No data loaded.")
        return

    print("✅ Data loaded successfully.")
    print(df.head())

    print("\n🤖 Select Recommendation Mode:")
    print("1 - Rule-based filtering")
    print("2 - Semantic (AI) matching")
    mode = input("Enter 1 or 2: ").strip()

    if mode == "1":
        # (unchanged rule-based mode)
        profile = get_user_profile()
        matches = recommend_scholarships(
            df,
            field_of_study=profile['field'],
            country=profile['country'],
            application_fee=profile['no_fee']
        )
        if profile['degree']:
            matches = matches[
                matches['Eligibility_Criteria'].str.contains(profile['degree'], case=False, na=False)
            ]
        if matches.empty:
            print("❌ No scholarships found.")
        else:
            print(matches[['Title','Provider','Fields_of_Study','Country','Application_Fee']])

    elif mode == "2":
        # ── Modified Semantic (AI) mode with chatbot-like flow ──
        original_query = input(
            "Enter your request (e.g., 'I want scholarships in the USA'): "
        ).strip()

        # Infer country from original query
        inferred_country = None
        for country in pycountry.countries:
            if country.name.lower() in original_query.lower():
                inferred_country = country.name
                break

        # Paraphrase via Ollama
        print("\n🤖 Paraphrasing your query…")
        paraphrase_prompt = (
            "Paraphrase this query exactly—do not add anything:\n\n" + original_query
        )
        refined_query = query_ollama_cli(paraphrase_prompt)
        print("Paraphrased query:", refined_query)

        # PASS 1: Quick results
        print("\n🔎 Quick results based on country…\n")
        initial = semantic_recommend(
            df,
            user_query=refined_query,
            user_country=inferred_country,
            top_n=5
        )
        if initial.empty:
            print("❌ No quick matches. Try a broader request.")
        else:
            for idx, row in initial.iterrows():
                print(f"{idx+1}. {row['Title']} — {row['Provider']} ({row['Country']})")
        print("\nLet's refine further…\n")

        # PASS 2: Follow-ups
        field = input("Field of study? (or skip): ").strip().lower() or None
        fee_ans = input("No‑fee only? (Yes/No or skip): ").strip().lower()
        fee_pref = True if fee_ans == "yes" else False if fee_ans == "no" else None
        deadline_input = input("Deadline (YYYY-MM-DD or skip): ").strip()
        deadline = None
        if deadline_input:
            try:
                deadline = pd.to_datetime(deadline_input)
            except:
                print("⚠️ Invalid date, skipping.")
        degree = input("Degree level? undergraduate/graduate or skip: ").strip().lower() or None

        # PASS 3: Final refined results
        print("\n🔎 Final refined recommendations:\n")
        final = semantic_recommend(
            df,
            user_query=refined_query,
            user_country=inferred_country,
            user_deadline=deadline,
            fee_pref=fee_pref,
            user_degree=degree,
            top_n=5
        )
        if final.empty:
            print("❌ No scholarships found after refinement.")
        else:
            for idx, row in final.iterrows():
                print(f"{idx+1}. {row['Title']} — {row['Provider']}")
                print(f"   {row['Explanation']}\n")

    else:
        print("⚠️ Invalid mode. Please choose 1 or 2.")


if __name__ == "__main__":
    main()
