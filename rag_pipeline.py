import os
import requests
from document_loader import DocumentLoader
from embedding import EmbeddingModel
from vector_store import VectorStore


class RAGPipeline:
    """Agent文档问答：RAG检索 + LLM推理生成"""

    def __init__(self, api_key: str = None):
        self.loader = DocumentLoader(chunk_size=500, chunk_overlap=50)
        self.embedder = EmbeddingModel(api_key=api_key)
        self.store = VectorStore()
        self.api_key = api_key  # 保存API Key用于生成

    def index_document(self, file_path: str):
        """索引文档：加载→切分→向量化→存储"""
        print(f"正在加载文档: {file_path}")
        chunks = self.loader.load(file_path)
        print(f"切分成 {len(chunks)} 个chunk")

        print("正在生成embedding...")
        vectors = self.embedder.embed(chunks)

        print("正在存入向量库...")
        self.store.add(chunks, vectors)
        print("索引完成！")

    def query(self, question: str, top_k: int = 3) -> str:
        """RAG检索：返回检索到的上下文（原始片段）"""
        print(f"\n查询: {question}")

        # 1. 问题向量化
        query_vector = self.embedder.embed_query(question)

        # 2. 检索相似chunk
        results = self.store.search(query_vector, top_k=top_k)

        # 3. 组装上下文
        context = "\n\n".join([
            f"[相似度: {score:.4f}]\n{text[:300]}..."
            for text, score in results
        ])

        return context

    def generate_answer(self, question: str, top_k: int = 3) -> str:
        """
        Agent模式：检索 + LLM推理生成完整回答
        这才是智能体的核心能力
        """
        print(f"\n🤔 用户问题: {question}")

        # Step 1: 检索相关知识
        print("🔍 正在检索相关知识...")
        context = self.query(question, top_k=top_k)

        # Step 2: 构造Prompt让模型推理
        prompt = f"""你是一个智能文档问答助手。请基于以下参考资料回答用户问题。

参考资料：
{context}

用户问题：{question}

要求：
1. 仔细阅读参考资料
2. 给出准确、简洁的回答
3. 如果资料中没有相关信息，请说明"根据文档内容无法确定"

请用中文回答："""

        # Step 3: 调用智谱Chat API生成回答
        print("🧠 正在生成回答...")
        try:
            response = requests.post(
                "https://open.bigmodel.cn/api/paas/v4/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "glm-4-flash",  # 免费模型
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7
                },
                timeout=60
            )
            response.raise_for_status()
            answer = response.json()['choices'][0]['message']['content']
            return f"\n💡 回答：\n{answer}\n\n📚 参考来源：\n{context}"
        except Exception as e:
            return f"生成回答时出错: {e}\n\n检索到的上下文：\n{context}"

    def save_index(self, filepath: str = "index.json"):
        """保存索引"""
        self.store.save(filepath)
        print(f"索引已保存到: {filepath}")

    def load_index(self, filepath: str = "index.json"):
        """加载索引"""
        self.store.load(filepath)
        print(f"索引已加载: {filepath}")