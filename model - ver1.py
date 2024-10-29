# Import necessary libraries
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import AutoTokenizer, AutoModel
from umap import UMAP
import hdbscan
import numpy as np
import torch
import nltk
from gensim.models import HdpModel
from gensim.corpora import Dictionary
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load stop words
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

# 1. Load and Preprocess the 20 Newsgroups dataset
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    return ' '.join([word for word in tokens if word.isalpha() and word not in stop_words])

data = fetch_20newsgroups(subset='all', categories=['sci.space', 'comp.graphics'], remove=('headers', 'footers', 'quotes'))
texts = [preprocess_text(doc) for doc in data.data]

# 2. Embed Text using a Pre-trained Language Model (e.g., BERT)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

def get_embeddings(texts):
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).detach().numpy())
    return np.vstack(embeddings)

embeddings = get_embeddings(texts)

# 3. Apply UMAP for dimensionality reduction
umap = UMAP(n_components=10, random_state=42)
umap_embeddings = umap.fit_transform(embeddings)

# 4. Apply HDBSCAN for clustering
clusterer = hdbscan.HDBSCAN(min_cluster_size=15, metric='euclidean')
labels = clusterer.fit_predict(umap_embeddings)

# 5. TF-IDF for LDA and HDP Topic Modeling
vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(texts)

# LDA Model
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(tfidf_matrix)

# HDP Model
dictionary = Dictionary([text.split() for text in texts])
corpus = [dictionary.doc2bow(text.split()) for text in texts]
hdp = HdpModel(corpus, id2word=dictionary)

# Output results
print("HDBSCAN Labels:", labels)
print("LDA Topics:")
for idx, topic in enumerate(lda.components_):
    top_terms = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]]
    print(f"Topic {idx+1}: {', '.join(top_terms)}")

print("HDP Topics:")
for i, topic in enumerate(hdp.show_topics(num_topics=5, formatted=False)):
    print(f"Topic {i+1}: {', '.join([word for word, prob in topic[1]])}")
