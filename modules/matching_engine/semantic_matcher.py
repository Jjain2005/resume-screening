import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from sentence_transformers import SentenceTransformer
from ..resume_processing.models import ProcessedResume, EvidenceSentence

DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"

class SemanticMatcher:
    """
    Computes semantic similarity between JD requirements and resume sentences
    using local SentenceTransformers models, with embedding caching.
    """
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        similarity_threshold: float = 0.76,
        partial_threshold: float = 0.65
    ):
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.partial_threshold = partial_threshold
        self._model: Optional[SentenceTransformer] = None
        self._embedding_cache: Dict[str, np.ndarray] = {}

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def change_model(self, new_model_name: str):
        if new_model_name != self.model_name:
            self.model_name = new_model_name
            self._model = None
            self._embedding_cache.clear()

    def get_embedding(self, text: str) -> np.ndarray:
        cleaned = text.strip()
        if cleaned in self._embedding_cache:
            return self._embedding_cache[cleaned]

        emb = self.model.encode(cleaned, convert_to_numpy=True, normalize_embeddings=True)
        self._embedding_cache[cleaned] = emb
        return emb

    def get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        missing_indices = []
        missing_texts = []
        results = [None] * len(texts)

        for i, t in enumerate(texts):
            cleaned = t.strip()
            if cleaned in self._embedding_cache:
                results[i] = self._embedding_cache[cleaned]
            else:
                missing_indices.append(i)
                missing_texts.append(cleaned)

        if missing_texts:
            new_embs = self.model.encode(missing_texts, convert_to_numpy=True, normalize_embeddings=True)
            for idx, text, emb in zip(missing_indices, missing_texts, new_embs):
                self._embedding_cache[text] = emb
                results[idx] = emb

        return results

    def cache_jd_embeddings(self, requirements: List[Any]) -> Dict[str, np.ndarray]:
        """Precomputes and caches normalized embeddings for all requirements in a JD once."""
        req_texts = [r.text.strip() for r in requirements]
        embs = self.get_embeddings_batch(req_texts)
        return {r.id: emb for r, emb in zip(requirements, embs)}

    def batch_match_resume_against_jd(
        self,
        resume: ProcessedResume,
        jd_embeddings: Dict[str, np.ndarray]
    ) -> Dict[str, Tuple[EvidenceSentence, float]]:
        """
        Vectorized batch match: calculates dot product of all resume sentences
        against all precomputed JD embeddings in a single NumPy matrix multiplication.
        """
        if not resume.all_sentences or not jd_embeddings:
            return {}

        sentences = resume.all_sentences
        sentence_texts = [s.text for s in sentences]
        sentence_embs = self.get_embeddings_batch(sentence_texts)
        sentence_matrix = np.array(sentence_embs)  # Shape: (N, D)

        req_ids = list(jd_embeddings.keys())
        req_matrix = np.array([jd_embeddings[rid] for rid in req_ids])  # Shape: (K, D)

        # Dot product (N, D) @ (D, K) -> (N, K)
        sim_matrix = np.dot(sentence_matrix, req_matrix.T)

        results = {}
        for col_idx, rid in enumerate(req_ids):
            col_scores = sim_matrix[:, col_idx]
            best_idx = int(np.argmax(col_scores))
            results[rid] = (sentences[best_idx], float(col_scores[best_idx]))

        return results

    def find_best_semantic_match(
        self,
        query_text: str,
        resume: ProcessedResume
    ) -> Optional[Tuple[EvidenceSentence, float]]:
        """
        Finds the single most semantically similar sentence in the resume.
        Returns (best_sentence, cosine_similarity) or None if no sentences exist.
        """
        if not resume.all_sentences:
            return None

        query_emb = self.get_embedding(query_text)
        sentences = resume.all_sentences
        sentence_texts = [s.text for s in sentences]
        sentence_embs = self.get_embeddings_batch(sentence_texts)

        # Since embeddings are normalized, dot product equals cosine similarity
        matrix = np.array(sentence_embs)
        similarities = np.dot(matrix, query_emb)

        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        return sentences[best_idx], best_score

    def find_all_matches_above_threshold(
        self,
        query_text: str,
        resume: ProcessedResume,
        threshold: Optional[float] = None
    ) -> List[Tuple[EvidenceSentence, float]]:
        """Returns all sentences in the resume scoring above the given threshold."""
        if not resume.all_sentences:
            return []

        th = threshold if threshold is not None else self.partial_threshold
        query_emb = self.get_embedding(query_text)
        sentences = resume.all_sentences
        sentence_texts = [s.text for s in sentences]
        sentence_embs = self.get_embeddings_batch(sentence_texts)

        matrix = np.array(sentence_embs)
        similarities = np.dot(matrix, query_emb)

        matches = []
        for idx, score in enumerate(similarities):
            if score >= th:
                matches.append((sentences[idx], float(score)))

        # Sort descending by score
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def find_top_k_semantic_matches(
        self,
        query_text: str,
        resume: ProcessedResume,
        k: int = 3,
        threshold: Optional[float] = None
    ) -> List[Tuple[EvidenceSentence, float]]:
        """Returns top-k sentences in the resume scoring above the given threshold."""
        matches = self.find_all_matches_above_threshold(query_text, resume, threshold=threshold)
        return matches[:k]

