# Mini Agent - 智能文档问答助手

手写轻量化智能体（Agent），支持PDF/TXT/MD文档的智能问答。从零实现RAG检索+LLM推理生成完整闭环。

## 核心能力

🤖 **Agent模式**：检索 + LLM推理生成完整回答（不只是片段）
🔍 **手写RAG**：纯Python实现向量检索（余弦相似度，无FAISS/LangChain）
📄 **多格式支持**：PDF、TXT、Markdown
🧩 **模块化架构**：DocumentLoader → TextSplitter → Embedding → VectorStore → Generator

## 技术架构

```
用户提问
    ↓
[DocumentLoader] 加载PDF/TXT/MD
    ↓
[TextSplitter] 滑动窗口切分(500字/块, 重叠50字)
    ↓
[EmbeddingModel] 智谱API向量化
    ↓
[VectorStore] 手写余弦相似度检索
    ↓
[Generator] 智谱Chat API推理生成回答
    ↓
完整答案 + 参考来源
```

**为什么手写？** 理解RAG每个环节的原理，面试能讲清楚"余弦相似度怎么算"、"为什么分块"、"检索后怎么生成"。

## 快速开始

### 1. 安装依赖
```bash
pip install requests PyPDF2 python-dotenv
```

### 2. 配置API Key
复制 `.env.example` 为 `.env`，填入智谱API Key：
```env
ZHIPU_API_KEY=your_key_here
```

### 3. 运行Agent
```bash
python main.py
```

## 项目结构

```
mini-agent/
├── main.py              # 入口：Agent问答交互
├── rag_pipeline.py      # 核心：RAG流程 + LLM生成
├── document_loader.py   # 文档加载与智能切分
├── embedding.py         # 智谱Embedding API封装
├── vector_store.py      # 手写向量存储与余弦相似度检索
├── requirements.txt
└── .env.example
```

## 核心代码解析

### 1. 手写向量检索（VectorStore）
```python
def search(self, query_vector, top_k=3):
    # 余弦相似度 = 点积 / (|A| * |B|)
    for vec in self.vectors:
        dot = sum(a*b for a,b in zip(query_vector, vec))
        norm = l2_norm(query_vector) * l2_norm(vec)
        similarity = dot / norm
    # 排序返回Top-K
```
**不依赖FAISS**，纯Python实现，面试能讲清楚原理。

### 2. Agent生成（RAGPipeline）
```python
def generate_answer(self, question):
    # Step1: 检索相关知识
    context = self.query(question)
    
    # Step2: 构造Prompt让模型推理
    prompt = f"基于以下资料回答：{context}\n问题：{question}"
    
    # Step3: 调用智谱Chat API生成
    answer = call_llm(prompt)
    return answer
```
**完整Agent闭环**：检索(Recall) → 推理(Reasoning) → 生成(Generation)

### 3. 双API调用
- **Embedding API**：`/embeddings` 文档向量化
- **Chat API**：`/chat/completions` 生成回答

## 示例输出

```
🤔 用户问题: 文档里讲了什么内容？
🔍 正在检索相关知识...
🧠 正在生成回答...

💡 回答：
这份文档介绍了xxx的基本信息、教育经历、技术技能和项目经验。
他是一名人工智能专业本科生，熟悉Python、PyTorch、BERT等技术，
有车牌识别和智能问答系统两个项目经验。

📚 参考来源：
[相似度: 0.2761] 项目经验：基于CNN的车牌识别系统...
[相似度: 0.2370] 教育经历：xxx大学xxx专业...
```


## 依赖

- Python 3.8+
- 智谱API Key（免费额度够用）

## 许可证

MIT
