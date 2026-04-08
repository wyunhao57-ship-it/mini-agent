import os
from dotenv import load_dotenv
from rag_pipeline import RAGPipeline

# 加载 .env 文件
load_dotenv()

# 从环境变量读取API Key
api_key = os.getenv("ZHIPU_API_KEY")
if not api_key:
    raise ValueError("请先设置 ZHIPU_API_KEY 环境变量，或在 .env 文件中配置")

rag = RAGPipeline(api_key=api_key)

# 1. 索引文档（支持txt/md/pdf）
print("=" * 50)
print("📄 步骤1：索引文档")
print("=" * 50)
rag.index_document("C:/Users/hp/Desktop/1.pdf")

# 2. 保存索引（下次直接加载，不用重新解析）
rag.save_index("my_index.json")

# 3. Agent模式：提问并获得完整回答（检索 + 生成）
print("\n" + "=" * 50)
print("🤖 步骤2：Agent问答模式")
print("=" * 50)

questions = [
    "文档里讲了什么内容？",
    "这份简历的求职意向是什么？",
    "他有什么项目经验？"
]

for q in questions:
    result = rag.generate_answer(q)  # Agent模式：检索+LLM生成
    print(result)
    print("-" * 50)