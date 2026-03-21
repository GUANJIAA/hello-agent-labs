import os
from typing import List, Dict, Any, Optional
from openai import OpenAI

INITIAL_PROMPT_TEMPLATE = """
你是一位资深的Python程序源。请根据以下要求，编写一个Python函数。
你的代码必须要包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。

要求：{task}

请直接输出代码，不要包含任何额外的解释。
"""

REFLECT_PROMPT_TEMPLATE = """
你是一位极其严格的代码评审专家和资深算法工程师，对代码的性能有极致的要求。
你的任务是审查以下Python代码，并专注于找出其在<strong>算法效率</strong>上的主要瓶颈。
# 原始任务:
{task}
# 待审查的代码:
```python
{code}
```
请分析该代码的时间复杂度，并思考是否存在一种<strong>算法上更优</strong>的解决方案来显著提升性能。
如果存在，请清晰地指出当前算法的不足，并提出具体的、可行的改进算法建议（例如，使用筛法替代试除法）。
如果代码在算法层面已经达到最优，才能回答“无需改进”。
请直接输出你的反馈，不要包含任何额外的解释。
"""

REFINE_PROMPT_TEMPLATE = """
你是一位资深的Python程序员。你正在根据一位代码评审专家的反馈来优化你的代码。
# 原始任务:
{task}
# 你上一轮尝试的代码:
{last_code_attempt}
评审员的反馈：
{feedback}
请根据评审员的反馈，生成一个优化后的新版本代码。
你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。
请直接输出优化后的代码，不要包含任何额外的解释。
"""

class HelloAgentsLLM:
    """
    大语言模型客户端类，用于与OpenAI API进行交互
    """
    def __init__(self, model: str = None, apiKey: str = None, baseUrl: str = None, timeout: int = None) -> None:
        """
        初始化大语言模型客户端
        
        Args:
            model: 模型ID
            apiKey: API密钥
            baseUrl: API基础URL
            timeout: 请求超时时间（秒）
        """
        self.model = model or os.getenv("LLM_MODEL_ID")
        self.apiKey = apiKey or os.getenv("LLM_API_KEY")
        self.baseUrl = baseUrl or os.getenv("LLM_BASE_URL")
        self.timeout = timeout or int(os.getenv("LLM_API_TIMEOUT", 60))

        if not all([self.model, self.apiKey, self.baseUrl]):
            raise ValueError("模型ID、API密钥和API基础URL不能为空")

        self.client = OpenAI(
            api_key=self.apiKey,
            base_url=self.baseUrl,
            timeout=self.timeout
        )

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        """
        调用大语言模型生成响应
        
        Args:
            messages: 消息列表
            temperature: 生成温度
            
        Returns:
            模型生成的文本
        """
        print(f"调用模型 {self.model}，温度 {temperature}，消息 {messages}")
        try:
            response = self.client.chat.completions.create(
                model = self.model,
                messages = messages,
                temperature = temperature,
                stream = True
            )

            # 处理流式响应
            print("大语言模型响应成功:")
            collected_content = []
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected_content.append(content)
            print() # 在流式输出结束后换行
            return "".join(collected_content)

        except Exception as e:
            print(f"调用大语言模型时出错: {e}")
            return ""

class Memory:
    """
    一个简单的短期记忆模块，用于存储智能体的行动与反思轨迹.
    """

    def __init__(self):
        """
        初始化一个空列表来存储所有记录。
        """
        self.records = []

    def add_record(self, record_type: str, content: str):
        """
        向记忆中添加一条新纪录。

        参数:
        - record_type: 记录类型，例如 "action" 或 "reflection"。
        - content: 记录的具体内容。
        """
        record = {"type": record_type, "content": content}
        self.records.append(record)
        print(f"记忆更新，添加记录: {record}")

    def get_trajectory(self) -> str:
        """
        将所有记忆记录格式化为一个连贯的字符串文本，用于构建提示词.
        """
        trajectory_parts = []
        for record in self.records:
            if record['type'] == 'action':
                trajectory_parts.append(f"智能体执行: {record['content']}")
            elif record['type'] == 'reflection':
                trajectory_parts.append(f"智能体反思: {record['content']}")
        return "\n".join(trajectory_parts)

    def get_last_execution(self) -> Optional[str]:
        """
        获取最近一次智能体执行的记录内容。
        如果不存在，则返回 None.
        """
        for record in self.records:
            if record['type'] == 'action':
                return record['content']
        return None

    
class ReflectionAgent:
    def __init__(self, llm_client, max_iterations=3):
        self.llm_client = llm_client
        self.memory = Memory()
        self.max_iterations = max_iterations

    def run(self, task: str):
        print(f"\n--- 开始处理任务 ---\n任务: {task}")
        # --- 1. 初始执行 ---
        print("\n--- 正在进行初始尝试 ---")
        initial_prompt = INITIAL_PROMPT_TEMPLATE.format(task=task)
        initial_code = self._get_llm_response(initial_prompt)
        self.memory.add_record("action", initial_code)
        # --- 2. 迭代循环:反思与优化 ---
        for i in range(self.max_iterations):
            print(f"\n--- 第 {i+1}/{self.max_iterations} 轮迭代 ---")
            # a. 反思
            print("\n-> 正在进行反思...")
            last_code = self.memory.get_last_execution()
            reflect_prompt = REFLECT_PROMPT_TEMPLATE.format(task=task,
            code=last_code)
            feedback = self._get_llm_response(reflect_prompt)
            self.memory.add_record("reflection", feedback)
            # b. 检查是否需要停止
            if "无需改进" in feedback:
                print("\n✅ 反思认为代码已无需改进，任务完成。")
                break
            # c. 优化
            print("\n-> 正在进行优化...")
            refine_prompt = REFINE_PROMPT_TEMPLATE.format(
                task=task,
                last_code_attempt=last_code,
                feedback=feedback
            )
            refined_code = self._get_llm_response(refine_prompt)
            self.memory.add_record("action", refined_code)
        final_code = self.memory.get_last_execution()
        print(f"\n--- 任务完成 ---\n最终生成的代码:\n```python\n{final_code}\n```")
        return final_code

    def _get_llm_response(self, prompt: str) -> str:
        """一个辅助方法，用于调用LLM并获取完整的流式响应。"""
        messages = [{"role": "user", "content": prompt}]
        response_text = self.llm_client.think(messages=messages) or ""
        return response_text


def main():
    """
    主函数，演示如何使用ReflectionAgent
    """
    try:
        llm_client = HelloAgentsLLM()
        agent = ReflectionAgent(llm_client, max_iterations=3)
        
        task = "编写一个函数，计算给定列表中所有偶数的和。"
        final_code = agent.run(task)
        
        print(f"\n最终生成的代码:\n{final_code}")
        
    except ValueError as e:
        print(f"配置错误: {e}")
        print("请确保已设置以下环境变量:")
        print("  - LLM_MODEL_ID")
        print("  - LLM_API_KEY")
        print("  - LLM_BASE_URL")
        print("  - LLM_API_TIMEOUT (可选)")
    except Exception as e:
        print(f"运行时错误: {e}")


if __name__ == "__main__":
    main()