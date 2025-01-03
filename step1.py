from query_processing import QueryProcessor
import requests
import nltk
from bs4 import BeautifulSoup as bs
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
from collections import defaultdict
import math
from sklearn.metrics import precision_score, recall_score, f1_score
# Create a TF-IDF model
class TFIDF:
    def __init__(self, documents):
        self.documents = documents
        self.idf = self.compute_idf()

    def compute_tf(self, document):
        tf = defaultdict(float)
        for term in document:
            tf[term] += 1
        for term in tf:
            tf[term] /= len(document)
        return tf

    def compute_idf(self):
        idf = defaultdict(float)
        total_documents = len(self.documents)
        for document in self.documents.values():
            for term in set(document):
                idf[term] += 1
        for term in idf:
            idf[term] = math.log(total_documents / (1 + idf[term]))
        return idf

    def compute_tfidf(self, document):
        tf = self.compute_tf(document)
        tfidf = defaultdict(float)
        for term in tf:
            tfidf[term] = tf[term] * self.idf[term]
        return tfidf

    def rank(self, query):
        query_tokens = word_tokenize(query.lower())
        query_tokens = [re.sub(r'\W', '', token) for token in query_tokens if re.sub(r'\W', '', token)]
        query_tokens = [token for token in query_tokens if token not in stop_words]
        query_tokens = [lemmatizer.lemmatize(token) for token in query_tokens]

        query_tfidf = self.compute_tfidf(query_tokens)
        scores = defaultdict(float)

        for doc_id, document in self.documents.items():
            doc_tfidf = self.compute_tfidf(document)
            for term in query_tfidf:
                if term in doc_tfidf:
                    scores[doc_id] += query_tfidf[term] * doc_tfidf[term]

        ranked_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return ranked_docs
class VectorSpaceModel:
    def __init__(self, documents):
        self.documents = documents
        self.tfidf = TFIDF(documents)

    def compute_cosine_similarity(self, vec1, vec2):
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])

        sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
        sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)

        if not denominator:
            return 0.0
        else:
            return float(numerator) / denominator

    def rank(self, query):
        query_tokens = word_tokenize(query.lower())
        query_tokens = [re.sub(r'\W', '', token) for token in query_tokens if re.sub(r'\W', '', token)]
        query_tokens = [token for token in query_tokens if token not in stop_words]
        query_tokens = [lemmatizer.lemmatize(token) for token in query_tokens]

        query_tfidf = self.tfidf.compute_tfidf(query_tokens)
        scores = defaultdict(float)

        for doc_id, document in self.documents.items():
            doc_tfidf = self.tfidf.compute_tfidf(document)
            scores[doc_id] = self.compute_cosine_similarity(query_tfidf, doc_tfidf)

        ranked_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return ranked_docs
class OkapiBM25:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.avg_doc_len = self.compute_avg_doc_len()
        self.idf = self.compute_idf()

    def compute_avg_doc_len(self):
        total_length = sum(len(doc) for doc in self.documents.values())
        return total_length / len(self.documents)

    def compute_idf(self):
        idf = defaultdict(float)
        total_documents = len(self.documents)
        for document in self.documents.values():
            for term in set(document):
                idf[term] += 1
        for term in idf:
            idf[term] = math.log((total_documents - idf[term] + 0.5) / (idf[term] + 0.5) + 1)
        return idf

    def compute_bm25(self, document, query_tokens):
        doc_len = len(document)
        score = 0.0
        for term in query_tokens:
            if term in document:
                term_freq = document.count(term)
                numerator = self.idf[term] * term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                score += numerator / denominator
        return score

    def rank(self, query):
        query_tokens = word_tokenize(query.lower())
        query_tokens = [re.sub(r'\W', '', token) for token in query_tokens if re.sub(r'\W', '', token)]
        query_tokens = [token for token in query_tokens if token not in stop_words]
        query_tokens = [lemmatizer.lemmatize(token) for token in query_tokens]


        scores = defaultdict(float)
        for doc_id, document in self.documents.items():
            scores[doc_id] = self.compute_bm25(document, query_tokens)
            ranked_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return ranked_docs
# Function to evaluate the performance of the search engine
def evaluate_search_engine(model, test_queries):
    y_true = []
    y_pred = []

    for query, relevant_docs in test_queries.items():
        ranked_docs = model.rank(query)
        retrieved_docs = [doc_id for doc_id, _ in ranked_docs]

        for doc_id in relevant_docs:
            y_true.append(1 if doc_id in relevant_docs else 0)
            y_pred.append(1 if doc_id in retrieved_docs else 0)

    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")
#Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

url = "https://en.wikipedia.org/wiki/Enron"
page = requests.get(url)
soup = bs(page.text, 'html.parser')

#Extract text from the main content of the webpage
content = soup.find(id="bodyContent")
text = content.get_text()

#Tokenization
tokens = word_tokenize(text)

#Remove special characters and convert to lowercase
tokens = [re.sub(r'\W', '', token).lower() for token in tokens if re.sub(r'\W', '', token)]

#Remove stop words
stop_words = set(stopwords.words('english'))
tokens = [token for token in tokens if token not in stop_words]

#Stemming
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(token) for token in tokens]

#Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

print("Lemmatized Tokens:", lemmatized_tokens[:100])
#OR
#print("Stemmed Tokens:", stemmed_tokens[:100])

# Create inverted index
inverted_index = {}
for index, token in enumerate(lemmatized_tokens):
    if token not in inverted_index:
        inverted_index[token] = []
    inverted_index[token].append(index)
print("Inverted Index:", {k: v[:5] for k, v in inverted_index.items()})  # Print first 5 entries for brevity
# Save the inverted index to a file
qp =QueryProcessor(inverted_index)
# Example of a search 
print(qp.search("Enron or scandal"))
documents = {
    0: inverted_index,
}
tfidf = TFIDF(documents)
vsm = VectorSpaceModel(documents)

bm25 = OkapiBM25(inverted_index)
# Define a set of test queries and their expected relevant documents
test_queries = {
    "energy market": [0],
    "financial scandal": [0],
    "email communication": [0],
}


# Evaluate the performance of each model
print("Evaluating TF-IDF Model")
evaluate_search_engine(tfidf, test_queries)

print("Evaluating Vector Space Model")
evaluate_search_engine(vsm, test_queries)

print("Evaluating Okapi BM25 Model")
evaluate_search_engine(bm25, test_queries)




while True:
    print("\nChoose an option:")
    print("1. TF-IDF Ranking")
    print("2. Boolean Retrieval")
    print("3. Vector Space Model")
    print("4. Okapi BM25 Ranking")
    print("5. Exit")
    choice = input("Enter your choice (1-5): ")

    if choice == '1':
        query = input("Enter your query for TF-IDF Ranking: ")
        ranked_docs = tfidf.rank(query)
        print(f"Ranked documents for '{query}':", ranked_docs)
    elif choice == '2':
        query = input("Enter your query for Boolean Retrieval: ")
        qp =QueryProcessor(inverted_index)
        results = qp.search(query)
        print(f"Search results for '{query}':", results)
    elif choice == '3':
        query = input("Enter your query for Vector Space Model: ")
        ranked_docs = vsm.rank(query)
        print(f"Ranked documents for '{query}':", ranked_docs)
    elif choice == '4':
        query = input("Enter your query for Okapi BM25 Ranking: ")
        ranked_docs = bm25.rank(query)
        print(f"Ranked documents for '{query}':", ranked_docs)
    elif choice == '5':
        print("Exiting...")
        break
    else:
        print("Invalid choice. Please try again.")



