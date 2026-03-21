

import os
from typing import List, Dict
from openai import OpenAI

PLANNER_PROMPT_TEMPLATE = """
你是一个顶级AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的计划。
请确保计划中的每个步骤都是一个独立的，可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表， 其中每个元素都是一个字符串， 表示一个子任务。
问题: {question}

请严格按照以下格式输出计划，```python与```作为前后缀是必要的:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""

EXECUTOR_PROMPT_TEMPLATE = """
你是一个顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题，完整的计划，以及到目前为止已经完成的步骤和结果。
请你专注于解决当前步骤，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题: {question}
# 完成计划: {plan}
# 历史步骤与结果: {history}
# 当前步骤: {current_step}

请仅输出针对"当前步骤"的回答:
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

class Planner:
    """
    计划生成器类，用于将复杂问题分解为可执行的步骤
    """
    def __init__(self, llm_client):
        """
        初始化计划生成器
        
        Args:
            llm_client: 大语言模型客户端
        """
        self.llm_client = llm_client

    def plan(self, question: str) -> List[str]:
        """
        根据用户问题生成一个行动计划
        
        Args:
            question: 用户问题
            
        Returns:
            行动计划列表
        """
        prompt = PLANNER_PROMPT_TEMPLATE.format(question=question)

        # 构建消息列表
        messages = [{"role": "user", "content": prompt}]

        print("--- 调用LLM 生成计划 ---")

        # 使用流式输出获取计划
        responseText = self.llm_client.think(messages)
        if responseText:
            print("\n\n---完整模型响应---")
            print(responseText)

        # 解析LLM输出的列表字符串
        try:
            if not responseText:
                print("模型未返回有效响应")
                return []
            
            # 找到```python与```之间的内容
            start_index = responseText.index("```python")
            end_index = responseText.index("```", start_index + len("```python"))
            plan_str = responseText[start_index + len("```python"):end_index]
            # 从字符串中提取列表
            plan = eval(plan_str)
            if isinstance(plan, list):
                return plan
            else:
                print("解析结果不是有效的列表")
                return []
        except ValueError as e:
            print(f"解析计划时出错: {e}")
            return []
        except Exception as e:
            print(f"处理计划时发生未知错误: {e}")
            return []

class Executor:
    """
    执行器类，用于按照计划逐步执行任务
    """
    def __init__(self, llm_client):
        """
        初始化执行器
        
        Args:
            llm_client: 大语言模型客户端
        """
        self.llm_client = llm_client

    def execute(self, question: str, plan: list[str]) -> str:
        """
        根据计划，逐步执行并解决问题。
        
        Args:
            question: 用户问题
            plan: 行动计划列表
            
        Returns:
            最终执行结果
        """
        history = "" # 用于存储历史步骤和结果的字符串
        final_result = ""

        print("--- 调用LLM 执行计划 ---")

        if not plan:
            print("计划为空，无法执行")
            return ""

        for i, step in enumerate(plan):
            print(f"当前步骤: {step}")

            try:
                prompt = EXECUTOR_PROMPT_TEMPLATE.format(
                    question=question,
                    plan=plan,
                    history=history if history else "无",
                    current_step=step
                )

                messages = [{"role": "user", "content": prompt}]

                responseText = self.llm_client.think(messages) or ""

                # 更新历史记录, 为下一步做准备
                history += f"步骤 {i+1}: {step} -> {responseText}\n"
                
                print(f"步骤 {i+1} 执行结果: {responseText}")
                final_result = responseText
            except Exception as e:
                print(f"执行步骤 {i+1} 时出错: {e}")
                final_result = f"执行步骤 {i+1} 时出错: {e}"
                break

        return final_result

class PlanAndSolveAgent:
    """
    计划与解决代理类，整合计划生成和执行功能
    """
    def __init__(self, llm_client):
        """
        初始化计划和执行器
        
        Args:
            llm_client: 大语言模型客户端
        """
        self.llm_client = llm_client
        self.planner = Planner(llm_client)
        self.executor = Executor(llm_client)

    def run(self, question: str) -> str:
        """
        运行计划和执行器，解决用户问题。
        
        Args:
            question: 用户问题
            
        Returns:
            最终解决结果
        """
        plan = self.planner.plan(question)
        if not plan:
            return "无法生成计划"
        result = self.executor.execute(question, plan)
        print(f"最终结果: {result}")
        return result

def main():
    """
    主函数，演示如何使用 PlanAndSolveAgent
    """
    try:
        # 初始化大语言模型客户端
        llm_client = HelloAgentsLLM()
        
        # 创建计划与解决代理
        agent = PlanAndSolveAgent(llm_client)
        
        # 示例问题
        question = "如何制作一个简单的番茄炒蛋？"
        
        print(f"用户问题: {question}\n")
        
        # 运行代理解决问题
        result = agent.run(question)
        
        print("\n" + "="*50)
        print("问题解决完成！")
        print("="*50)
        
    except ValueError as e:
        print(f"配置错误: {e}")
        print("请确保已设置以下环境变量:")
        print("  - LLM_MODEL_ID")
        print("  - LLM_API_KEY")
        print("  - LLM_BASE_URL")
    except Exception as e:
        print(f"程序运行出错: {e}")

if __name__ == "__main__":
    main()
