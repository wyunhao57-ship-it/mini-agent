import math
import json
from typing import List, Tuple, Dict


class VectorStore:
    """
    手写向量存储与检索
    纯Python实现余弦相似度搜索，无FAISS
    """

    def __init__(self):
        self.vectors: List[List[float]] = []
        self.texts: List[str] = []
        self.metadata: List[Dict] = []

    def add(self, texts: List[str], vectors: List[List[float]],
            metadata: List[Dict] = None):
        """批量添加文档"""
        if len(texts) != len(vectors):
            raise ValueError("texts和vectors长度不一致")

        self.texts.extend(texts)
        self.vectors.extend(vectors)

        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{} for _ in texts])

    def search(self, query_vector: List[float], top_k: int = 3) -> List[Tuple[str, float]]:
        """
        余弦相似度检索
        余弦相似度 = (A·B) / (|A| * |B|)
        """
        if not self.vectors:
            return []

        scores = []
        query_norm = self._l2_norm(query_vector)

        for i, vec in enumerate(self.vectors):
            dot_product = sum(a * b for a, b in zip(query_vector, vec))
            vec_norm = self._l2_norm(vec)

            if vec_norm == 0:
                similarity = 0
            else:
                similarity = dot_product / (query_norm * vec_norm)

            scores.append((i, similarity))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_results = scores[:top_k]

        return [(self.texts[i], score) for i, score in top_results]

    def _l2_norm(self, vector: List[float]) -> float:
        """L2范数"""
        return math.sqrt(sum(x * x for x in vector))

    def save(self, filepath: str):
        """保存到JSON"""
        data = {
            'vectors': self.vectors,
            'texts': self.texts,
            'metadata': self.metadata
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self, filepath: str):
        """从JSON加载"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.vectors = data['vectors']
            self.texts = data['texts']
            self.metadata = data['metadata']