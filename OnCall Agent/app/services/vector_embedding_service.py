"""向量嵌入服务模块 - 基于 LangChain Embeddings 标准接口"""

from typing import List, Optional

from langchain_core.embeddings import Embeddings
from openai import OpenAI
from loguru import logger

from app.config import config


class DashScopeEmbeddings(Embeddings):
    """阿里云 DashScope Text Embedding (OpenAI 兼容模式)
    
    实现 LangChain 标准 Embeddings 接口:
    - embed_documents(texts: List[str]) → List[List[float]]: 批量嵌入文档
    - embed_query(text: str) → List[float]: 嵌入单个查询
    
    支持懒加载：API Key 无效时不立即抛出异常，延迟到实际调用时处理。
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-v4",
        dimensions: int = 1024,
    ):
        """
        初始化 DashScope Embeddings（懒加载模式）
        
        Args:
            api_key: DashScope API Key
            model: 嵌入模型名称
            dimensions: 向量维度
        """
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self._client: Optional[OpenAI] = None
        
        if api_key and api_key != "your-api-key-here":
            self._lazy_init_client()
            masked_key = self._mask_api_key(api_key)
            logger.info(
                f"DashScope Embeddings 初始化完成 - "
                f"模型: {model}, 维度: {dimensions}, API Key: {masked_key}"
            )
        else:
            logger.warning(
                "DashScope Embeddings 延迟初始化: DASHSCOPE_API_KEY 未设置或无效，"
                "请在 .env 文件中配置正确的 API Key 以启用向量嵌入功能。"
            )

    def _lazy_init_client(self):
        """延迟初始化 OpenAI 客户端"""
        if self._client is None:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )

    @staticmethod
    def _mask_api_key(api_key: str) -> str:
        """掩码 API Key 用于日志"""
        if len(api_key) > 8:
            return f"{api_key[:8]}...{api_key[-4:]}"
        return "***"

    def _ensure_client(self):
        """确保客户端已初始化，未初始化则抛友好提示"""
        if not self.api_key or self.api_key == "your-api-key-here":
            raise RuntimeError(
                "向量嵌入功能未启用: 请先在 .env 文件中配置有效的 DASHSCOPE_API_KEY"
            )
        self._lazy_init_client()
        return self._client

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量嵌入文档列表 (LangChain 标准接口)
        
        Args:
            texts: 文本列表
            
        Returns:
            List[List[float]]: 嵌入向量列表
        """
        if not texts:
            return []
        
        client = self._ensure_client()
        
        try:
            logger.info(f"批量嵌入 {len(texts)} 个文档")
            
            # 批量调用 API
            response = client.embeddings.create(
                model=self.model,
                input=texts,
                dimensions=self.dimensions,
                encoding_format="float"
            )
            
            embeddings = [item.embedding for item in response.data]
            logger.debug(f"批量嵌入完成, 维度: {len(embeddings[0])}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"批量嵌入失败: {e}")
            raise RuntimeError(f"批量嵌入失败: {e}") from e

    def embed_query(self, text: str) -> List[float]:
        """
        嵌入单个查询文本 (LangChain 标准接口)
        
        Args:
            text: 查询文本
            
        Returns:
            List[float]: 嵌入向量
        """
        if not text or not text.strip():
            raise ValueError("查询文本不能为空")
        
        client = self._ensure_client()
        
        try:
            logger.debug(f"嵌入查询, 长度: {len(text)} 字符")
            
            response = client.embeddings.create(
                model=self.model,
                input=text,
                dimensions=self.dimensions,
                encoding_format="float"
            )
            
            embedding = response.data[0].embedding
            logger.debug(f"查询嵌入完成, 维度: {len(embedding)}")
            
            return embedding
            
        except Exception as e:
            logger.error(f"查询嵌入失败: {e}")
            raise RuntimeError(f"查询嵌入失败: {e}") from e


# 全局单例 - 懒加载模式
vector_embedding_service = DashScopeEmbeddings(
    api_key=config.dashscope_api_key,
    model=config.dashscope_embedding_model,
    dimensions=1024
)
