import sys

dependencies = [
    "fastapi",
    "uvicorn",
    "sse_starlette",
    "langchain",
    "langchain_community",
    "langchain_core",
    "langchain_openai",
    "langgraph",
    "dashscope",
    "openai",
    "pymilvus",
    "pydantic",
    "pydantic_settings",
    "httpx",
    "aiohttp",
    "aiofiles",
    "python_multipart",
    "loguru",
    "dotenv",
    "langchain_milvus",
]

print(f"Python version: {sys.version}")
print("=" * 60)

missing = []
for dep in dependencies:
    try:
        __import__(dep)
        print(f"✅ {dep}")
    except ImportError as e:
        print(f"❌ {dep} - 缺少")
        missing.append(dep)

print("=" * 60)
if missing:
    print(f"缺少 {len(missing)} 个依赖包")
else:
    print("所有依赖包都已安装!")
