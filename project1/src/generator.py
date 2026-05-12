"""
Enterprise-RAG: Answer generator with CoT prompting, hallucination prevention,
and conversation history support.
"""
import json
from typing import Any

from loguru import logger
from openai import OpenAI

from src.config import config


class Generator:
    """LLM-based answer generator with advanced prompt engineering."""

    def __init__(self):
        cfg = config.get("llm", {})
        self.provider = cfg.get("provider", "local")
        self.model_name = cfg.get("model_name", "Qwen/Qwen2.5-7B-Instruct")
        self.max_tokens = cfg.get("max_tokens", 2048)
        self.temperature = cfg.get("temperature", 0.1)
        self.history_turns = config.get("conversation", {}).get("history_turns", 3)

        if self.provider in ("local", "deepseek"):
            self.api_base = cfg.get("api_base", "http://localhost:8000/v1")
            self.api_key = cfg.get("api_key", "not-needed")
        elif self.provider == "openai":
            openai_cfg = cfg.get("openai", {})
            self.api_base = openai_cfg.get("api_base", "https://api.openai.com/v1")
            self.api_key = openai_cfg.get("api_key", "")
            self.model_name = openai_cfg.get("model_name", "gpt-4o-mini")

        self.client = OpenAI(base_url=self.api_base, api_key=self.api_key)

    def generate(
        self,
        query: str,
        contexts: list[dict[str, Any]],
        history: list[dict[str, str]] | None = None,
        stream: bool = False,
    ) -> str | Any:
        """
        Generate an answer based on retrieved contexts and optional history.

        Args:
            query: User question
            contexts: Retrieved context chunks with metadata
            history: Recent conversation history
            stream: If True, return streaming response object

        Returns:
            Generated answer string or streaming iterator
        """
        prompt = self._build_prompt(query, contexts, history)
        system_prompt = self._build_system_prompt()

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=stream,
            )

            if stream:
                return response
            else:
                return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"抱歉，答案生成失败：{str(e)}"

    def generate_with_cot(
        self,
        query: str,
        contexts: list[dict[str, Any]],
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, str]:
        """Generate answer with Chain-of-Thought reasoning."""
        # Step 1: Analyze and reason
        reasoning_prompt = self._build_reasoning_prompt(query, contexts, history)
        system_prompt = self._build_system_prompt()

        try:
            reasoning_resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": reasoning_prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            reasoning = reasoning_resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"CoT reasoning failed: {e}")
            reasoning = "无法完成推理过程"

        # Step 2: Generate final answer
        answer_prompt = self._build_answer_prompt(query, contexts, reasoning, history)

        try:
            answer_resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": answer_prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            answer = answer_resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"CoT answer generation failed: {e}")
            answer = f"抱歉，答案生成失败：{str(e)}"

        return {
            "answer": answer,
            "reasoning": reasoning,
        }

    def _build_system_prompt(self) -> str:
        """Build the system prompt with anti-hallucination guardrails."""
        return """你是一个企业知识库智能助手。请严格遵守以下规则：

1. **基于上下文回答**：只能使用提供的参考文档内容回答问题。
2. **不知道就说不知道**：如果参考文档中没有相关信息，请明确回答"根据现有资料，无法回答该问题"，不要编造或猜测。
3. **引用来源**：回答时必须引用具体的文档来源（文档名、页码）。
4. **结构化输出**：使用清晰的分点、表格等格式组织答案。
5. **保持客观**：不添加个人观点或外部知识。"""

    def _build_prompt(
        self,
        query: str,
        contexts: list[dict[str, Any]],
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """Build the main generation prompt with contexts and history."""
        # Format contexts
        context_parts = []
        for i, ctx in enumerate(contexts, start=1):
            meta = ctx.get("metadata", {})
            source = meta.get("source", "未知")
            page = meta.get("page", "")
            page_info = f" (第{page}页)" if page else ""
            context_parts.append(
                f"【参考片段 {i}】来源: {source}{page_info}\n{ctx['text']}"
            )

        contexts_str = "\n\n".join(context_parts)

        # Format history
        history_str = ""
        if history:
            recent = history[-(self.history_turns * 2):]
            history_parts = []
            for msg in recent:
                role = "用户" if msg["role"] == "user" else "助手"
                history_parts.append(f"{role}: {msg['content']}")
            history_str = "\n".join(history_parts)

        prompt = f"""请根据以下参考文档内容回答用户问题。

## 参考文档内容
{contexts_str}

"""
        if history_str:
            prompt += f"""## 历史对话
{history_str}

"""

        prompt += f"""## 用户问题
{query}

## 回答要求
请基于上述参考文档内容回答问题，并在答案末尾列出所引用的文档来源（文档名、页码）。如果参考文档中没有相关信息，请如实说明。

答案："""

        return prompt

    def _build_reasoning_prompt(
        self,
        query: str,
        contexts: list[dict[str, Any]],
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """Build CoT reasoning prompt."""
        context_summaries = []
        for i, ctx in enumerate(contexts, start=1):
            meta = ctx.get("metadata", {})
            context_summaries.append(
                f"[片段{i}] 来源={meta.get('source', '未知')}, "
                f"页码={meta.get('page', 'N/A')}\n内容: {ctx['text'][:300]}..."
            )

        return f"""请对以下问题进行分步推理分析，不要直接给出最终答案。

问题: {query}

相关参考片段:
{chr(10).join(context_summaries)}

请按以下步骤分析：
1. 理解用户问题的核心意图
2. 逐一检查每个参考片段是否与问题相关
3. 从相关片段中提取关键信息
4. 判断信息是否充分回答用户问题

请输出你的分析过程："""

    def _build_answer_prompt(
        self,
        query: str,
        contexts: list[dict[str, Any]],
        reasoning: str,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """Build final answer prompt based on reasoning."""
        context_parts = []
        for i, ctx in enumerate(contexts, start=1):
            meta = ctx.get("metadata", {})
            source = meta.get("source", "未知")
            page = meta.get("page", "")
            page_info = f" (第{page}页)" if page else ""
            context_parts.append(
                f"【参考片段 {i}】来源: {source}{page_info}\n{ctx['text']}"
            )

        return f"""基于以下分析过程和参考文档内容，生成最终答案。

## 分析过程
{reasoning}

## 参考文档内容
{chr(10).join(context_parts)}

## 用户问题
{query}

请生成清晰、有引用的最终答案。如果参考文档信息不足，请明确说明。
答案："""

    def expand_query(self, query: str) -> list[str]:
        """Generate query variations for improved recall."""
        prompt = f"""请为以下问题生成 2 个语义相同但表达方式不同的变体。只输出变体问题，每行一个，不要编号。

原始问题: {query}

变体:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.7,
            )
            text = response.choices[0].message.content.strip()
            variations = [line.strip("- 1234567890. ") for line in text.split("\n") if line.strip()]
            return variations[:2]
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return []
