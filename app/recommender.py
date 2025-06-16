import nltk
import pandas as pd
import subprocess
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

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
    if fee_pref and fee_pref.lower() == 'yes' and str(scholarship_row.get('Application_Fee', '')).strip().lower() == 'no':
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
    # Strict country filter using word-boundary regex
    if user_country:
        pattern = fr"\b{user_country}\b"
        df = df[df['Country'].str.contains(pattern, case=False, na=False, regex=True)]
        # If no scholarships after filtering, return empty DataFrame early
        if df.empty:
            return df

    # Build text corpus and embeddings
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

    # Rerank for country and deadlines
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
        ),
        axis=1
    )

    return top_matches.reset_index(drop=True)
