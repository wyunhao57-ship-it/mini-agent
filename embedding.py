import os
import requests
import time
from typing import List


class EmbeddingModel:
    """智谱Embedding API封装"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('ZHIPU_API_KEY')
        if not self.api_key:
            raise ValueError("请设置ZHIPU_API_KEY环境变量")

        self.api_url = "https://open.bigmodel.cn/api/paas/v4/embeddings"
        self.model = "embedding-3"

    def embed(self, texts: List[str]) -> List[List[float]]:
        """批量获取embedding，自动分批（智谱限64条/次）"""
        if not texts:
            return []

        batch_size = 64
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self._call_api(batch)
            all_embeddings.extend(batch_embeddings)
            time.sleep(0.1)

        return all_embeddings

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "input": texts
        }

        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()

        data = response.json()
        return [item['embedding'] for item in data['data']]

    def embed_query(self, text: str) -> List[float]:
        """单条查询"""
        return self.embed([text])[0]