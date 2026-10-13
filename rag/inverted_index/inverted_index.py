import math
from collections import Counter, defaultdict
from typing import Any

# NOTE: if this file lives in the matching package, switch these to
# matching.constants..., matching.custom_types..., matching.helpers...
from constants.constants import BM25_B, BM25_K1, SEARCH_LIMIT
from custom_types.custom_types import Document
from helpers.helpers import tokenize


class InvertedIndex:
    # the parts this index contributes to a saved snapshot (see HybridSearch)
    PARTS = ("inverted_index", "term_frequencies", "doc_lengths")

    def __init__(self, docmap: dict[str, Document] | None = None) -> None:
        # a dictionary mapping tokens to set of document ids
        self.index: dict[str, set[str]] = {}
        # a dictionary mapping document ids to their full document objects;
        # HybridSearch passes the docmap it shares with the semantic index
        self.docmap: dict[str, Document] = {} if docmap is None else docmap
        # a dictionary mapping document ids to term frequencies
        self.term_frequencies: defaultdict[str, Counter] = defaultdict(Counter)
        # a dictionary mapping document ids to their lengths
        self.doc_lengths: dict[str, int] = {}
        # running sum of all document lengths. the average is then
        # total / count, which is O(1) and can never go stale, because every
        # method that changes doc_lengths also updates this total
        self.total_doc_length = 0

    # tokenize document content (text), add each token to the index with the document id;
    # if the document is already indexed, its old entry is replaced, not stacked on
    def add_document(self, text: str, doc_id: str) -> None:
        # clear any previous version first; otherwise Counter.update would
        # add the new counts on top of the old ones and double every tf
        self._unindex(doc_id)

        # tokenize the document content
        tokens = tokenize(text)
        # update the term frequencies of the document
        self.term_frequencies[doc_id].update(tokens)
        # save document length and keep the running total in sync
        self.doc_lengths[doc_id] = len(tokens)
        self.total_doc_length += len(tokens)

        # fill the index with the tokens
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

    # remove a document's index data (tf, postings, length) but leave the docmap alone;
    # shared by add_document (replace) and remove_document (delete)
    def _unindex(self, doc_id: str) -> None:
        # pop, not [], so the defaultdict does not insert an empty entry
        for token in self.term_frequencies.pop(doc_id, {}):
            postings = self.index.get(token)
            if postings is not None:
                postings.discard(doc_id)
                # drop empty postings so the token stops matching in bm25_search
                if not postings:
                    del self.index[token]
        # subtract the old length from the running total (0 if never indexed)
        self.total_doc_length -= self.doc_lengths.pop(doc_id, 0)

    # remove every trace of a document from the index; also the delete path.
    # the docmap is shared with the semantic index, so this removes it there too
    def remove_document(self, doc_id: str) -> None:
        self._unindex(doc_id)
        self.docmap.pop(doc_id, None)

    # calculate the average doc length across all documents
    def get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return self.total_doc_length / len(self.doc_lengths)

    # get the set of document ids of a token
    def get_documents(self, token: str) -> set[str]:
        return self.index.get(token) or set()

    # iterate over all the documents and add them to the docmap and the index;
    # calling it again with the same docs is safe (add_document replaces)
    def build(self, documents: list[Document]) -> None:
        for doc in documents:
            self.docmap[doc.id] = doc
            self.add_document(doc.content, doc.id)

    # the persisted state; the docmap is saved once, by HybridSearch.
    # total_doc_length is not saved: it is derived from doc_lengths on restore
    def export_parts(self) -> dict[str, Any]:
        return {
            "inverted_index": self.index,
            "term_frequencies": self.term_frequencies,
            "doc_lengths": self.doc_lengths,
        }

    # restore state saved by export_parts; never builds anything
    def restore_parts(self, parts: dict[str, Any]) -> None:
        # rebuild the exact types this class relies on, in case the
        # serializer handed back plain dicts/lists (JSON does; pickle doesn't):
        # postings must be sets, term_frequencies must be defaultdict(Counter)
        self.index = {token: set(ids) for token, ids in parts["inverted_index"].items()}
        self.term_frequencies = defaultdict(
            Counter,
            {doc_id: Counter(tf) for doc_id, tf in parts["term_frequencies"].items()},
        )
        self.doc_lengths = dict(parts["doc_lengths"])
        # recompute the derived running total
        self.total_doc_length = sum(self.doc_lengths.values())

    # get the frequency of a single token
    def get_tf(self, doc_id: str, token: str) -> int:
        # check if the document exists in the term frequencies
        if doc_id not in self.term_frequencies:
            return 0
        return self.term_frequencies[doc_id][token]

    # calculate the idf score of a single term
    def get_idf(self, term: str) -> float:
        # number of documents
        n = len(self.docmap)
        # get document frequency for the given term
        df = len(self.get_documents(term))
        return math.log((n - df + 0.5) / (df + 0.5) + 1)

    # calculate the saturated tf score
    def get_bm25_tf(self, doc_id: str, token: str) -> float:
        # O(1) now, thanks to the running total
        avg_len = self.get_avg_doc_length()
        length_norm = 1
        # calc length norm of the document
        if avg_len != 0:
            doc_len = self.doc_lengths[doc_id]
            length_norm = 1 - BM25_B + BM25_B * (doc_len / avg_len)
        # get the term frequency of the term
        tf = self.get_tf(doc_id, token)
        return (tf * (BM25_K1 + 1)) / (tf + BM25_K1 * length_norm)

    # calculate bm25 score of a token
    def bm25(self, doc_id: str, token: str) -> float:
        idf = self.get_idf(token)
        tf = self.get_bm25_tf(doc_id, token)
        return idf * tf

    # implement the bm25 search algorithm. results are returned sorted by
    # score, descending, so callers never need to re-sort them.
    # limit=None returns every matching document
    def bm25_search(self, query: str, limit: int | None = SEARCH_LIMIT) -> dict[str, float]:
        # tokenize the query
        tokens = tokenize(query)
        scores = defaultdict(float)

        for token in tokens:
            if token in self.index:
                for doc_id in self.index[token]:
                    scores[doc_id] += self.bm25(doc_id, token)

        # ties broken by doc id: self.index holds sets, whose iteration order
        # for strings changes between processes (hash randomization), so
        # without a tiebreaker equal-score documents swapped ranks run to run
        top = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
        # plain dicts keep insertion order, so the ranking is preserved
        return dict(top)
