import math
import string
import os
import pickle
from .search_utils import BM25_B, BM25_K1, DEFAULT_SEARCH_LIMIT, CACHE_DIR, format_search_result, load_movies, load_stopwords
from nltk.stem import PorterStemmer
from collections import defaultdict, Counter

stemmer = PorterStemmer()

class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.term_frequencies = {}
        self.docmap = {}
        self.doc_lengths = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")       
        self.term_frequencies_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")       
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    def __get_avg_doc_length(self) -> float:
        no_docs = len(self.doc_lengths)
        total_length = 0
        for k, v in self.doc_lengths.items():
            total_length += v
        if no_docs == 0:
            return 0.0
        return total_length / no_docs

    def __add_document(self, doc_id: int, text: str):
        tokens = tokenize_text(text)
        self.term_frequencies[doc_id] = Counter()
        for token in tokens:
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1
        self.doc_lengths[doc_id] = len(tokens)

    def bm25(self, doc_id: int, term: str) -> float:
        tf_component = self.get_bm25_tf(doc_id, term)
        idf_component = self.get_bm25_idf(term)
        return tf_component * idf_component

    def bm25_search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
        query_tokens = tokenize_text(query)

        scores = {}
        for doc_id in self.docmap:
            score = 0.0
            for token in query_tokens:
                score += self.bm25(doc_id, token)
            scores[doc_id] = score

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in sorted_docs[:limit]:
            doc = self.docmap[doc_id]
            formatted_result = format_search_result(
                doc_id=doc["id"],
                title=doc["title"],
                document=doc["description"],
                score=score,
            )
            results.append(formatted_result)

        return results

    def get_documents(self, term) -> list[int]:
        lc_term = term.lower()
        doc_ids = self.index.get(lc_term, set())
        return sorted(list(doc_ids))

    def get_tf(self, doc_id: int, term: str) -> int:
        token = tokenize_text(term)
        if len(token) > 1:
            raise Exception("found more than one token")
        token_str = token[0]
        return self.term_frequencies[doc_id][token_str]

    def get_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log((doc_count + 1) / (term_doc_count + 1))

    def get_bm25_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log((doc_count - term_doc_count + 0.5) / (term_doc_count + 0.5) + 1)

    def get_bm25_tf(self, doc_id, term, k1 = BM25_K1, b = BM25_B) -> float:
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()
        len_norm = 1 - b+b * (doc_length / avg_doc_length)
        tf = self.get_tf(doc_id, term)
        sat_tf = (tf * (k1 + 1)) / (tf + k1 * len_norm)
        return sat_tf

    def build(self):
        movies = load_movies()
        for m in movies:
            self.docmap[m["id"]] = m
            title_and_descr_str = f"{m['title']} {m['description']}" 

            self.__add_document(doc_id=m['id'], text=title_and_descr_str)

    def save(self):
        os.makedirs(CACHE_DIR, exist_ok=True)

        with open(self.index_path, 'wb') as i_f:
            pickle.dump(self.index, i_f, pickle.HIGHEST_PROTOCOL)

        with open(self.docmap_path, 'wb') as dm_f:
            pickle.dump(self.docmap, dm_f, pickle.HIGHEST_PROTOCOL)

        with open(self.term_frequencies_path, 'wb') as tf_f:
            pickle.dump(self.term_frequencies, tf_f, pickle.HIGHEST_PROTOCOL)

        with open(self.doc_lengths_path, 'wb') as dl_f:
            pickle.dump(self.doc_lengths, dl_f, pickle.HIGHEST_PROTOCOL)

    def load(self):
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.term_frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)

def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()

def bm25_tf_command(doc_id, term, k1 = BM25_K1) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_tf(doc_id, term, k1)

def bm25_idf_command(term:str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_idf(term)

def bm25search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    idx.load()
    return idx.bm25_search(query, limit)

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    idx.load()
    query_tokens = tokenize_text(query)
    seen, results = set(), []
    for query_token in query_tokens:
        matching_doc_ids = idx.get_documents(query_token)
        for doc_id in matching_doc_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            doc = idx.docmap[doc_id]
            results.append(doc)
            if len(results) >= limit:
                return results

    return results

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("","",string.punctuation))
    return text

def has_matching_token(query_tokens:list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False

def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    cleaned_tokens = remove_stopwords(tokens)
    valid_tokens = []
    for token in cleaned_tokens:
        if token:
            stem_token = stemmer.stem(token)
            valid_tokens.append(stem_token)
    return valid_tokens


def remove_stopwords(tokens: list[str]) -> list[str]:
    stop_words = load_stopwords()
    cleaned_tokens = [t for t in tokens if t not in stop_words]
    return cleaned_tokens


