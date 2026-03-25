"""Plan and Solve Agent实现 - 分解规划与逐步执行的智能体"""

import ast
from typing import Optional, List, Dict

from click import prompt
from core.agent import Agent
from core.llm import HelloAgentsLLM
from core.config import Config
from core.message import Message

# 默认规划器提示词模板
DEFAULT_PLANNER_PROMPT = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""

# 默认执行器提示词模板
DEFAULT_EXECUTOR_PROMPT = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决"当前步骤"，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对"当前步骤"的回答:
"""

class Planner:
    """规划器 - 负责将复杂问题分解为简单步骤"""

    def __init__(self, llm_client: HelloAgentsLLM, prompt_template: Optional[str] = None):
        self.llm_client = llm_client
        self.prompt_template = prompt_template if prompt_template else DEFAULT_PLANNER_PROMPT

    def plan(self, question: str, **kwargs) -> List[str]:
        """
        生成执行计划

        Args:
            question: 要解决的问题
            **kwargs: LLM调用参数

        Returns:
            List[str]: 执行计划
        """
        prompt = self.prompt_template.format(question=question, **kwargs)
        message = [Message(role="user", content=prompt)]

        print("正在生成计划...")
        response_text = self.llm_client.generate(message)
        print(f"计划已经生成: {response_text}")

        try:
            # 提取Python代码块中的列表
            plan_str = response_text.split("```python")[1].split("```")[0]
            plan = ast.literal_eval(plan_str)
            return plan if isinstance(plan, list) else []
        except ValueError:
            raise ValueError("计划格式错误")
       
class Exectuor:
    """执行器 - 负责按照计划执行任务"""
    def __init__(self, llm_client: HelloAgentsLLM, prompt_template: Optional[str] = None):
        self.llm_client = llm_client
        self.prompt_template = prompt_template if prompt_template else DEFAULT_EXECUTOR_PROMPT

    def execute(self, question: str, plan: List[str], **kwargs) -> str:
        """
        按计划执行任务

        Args:
            question: 原始问题
            plan: 执行计划
            **kwargs: LLM调用参数

        Returns:
            str: 执行结果
        """
        history = ""
        final_answer = ""

        print("\n开始执行任务...")
        for i, step in enumerate(plan, 1):
            print(f"正在执行步骤 {i}/{len(plan)}:{step}")
            prompt = self.prompt_template.format(
                question=question,
                plan=plan,
                history=history if history else "无",
                current_step=step,
            )
            messages = [{"role":"user", "content":prompt}]

            response_text = self.llm_client.invoke(message, **kwargs) or ""

            history += f"步骤 {i}: {response_text}\n"
            final_answer = response_text
            print(f"步骤 {i} 执行结果: {response_text}")
            
        return final_answer

class PlanAndSolveAgent(Agent):
    """Plan and Solve Agent - 分解规划与逐步执行的智能体
    
    这个Agent能够：
    1. 将复杂问题分解为简单步骤
    2. 按照计划逐步执行
    3. 维护执行历史和上下文
    4. 得出最终答案
    
    特别适合多步骤推理、数学问题、复杂分析等任务
    """
    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        custom_prompts: Optional[Dict[str, str]] = None,
    ):
        """
        初始化Plan and Solve Agent

        Args:
            name: Agent的名称
            llm: 用于生成计划和执行任务的LLM模型
            system_prompt: 系统提示词，用于引导Agent的行为
            config: 配置参数，用于调整Agent的行为
            custom_prompts: 自定义提示词模板，用于覆盖默认模板
        """
        super().__init__(name, llm, system_prompt, config)

        # 设置提示词模板：用户自定义优先，否则使用默认模板
        if custom_prompts:
            planner_prompt = custom_prompts.get("planner")
            exectuor_prompt = custom_prompts.get("exectuor")
        else:
            planner_prompt = None
            exectuor_prompt = None


        self.planner = Planner(llm, planner_prompt)
        self.exectuor = Exectuor(llm, exectuor_prompt)

    def run(self, input_text: str, **kwargs) -> str:
        """
        运行Plan and Solve Agent

        Args:
            input_text: 要解决问题
            **kwargs: 其他参数

        Returns:
            最终答案
        """
        print(f"{self.name} 开始处理问题:{input_text}")

        # 1. 生成计划
        plan = self.planner.plan(input_text, **kwargs)
        if not plan:
            final_answer = "无法生成有效的行动计划，任务终止。"
            print(f"\n--- 任务终止 ---\n{final_answer}")
            
            # 保存到历史记录
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(final_answer, "assistant"))
            
            return final_answer
        
        # 2. 执行计划
        final_answer = self.executor.execute(input_text, plan, **kwargs)
        print(f"\n--- 任务完成 ---\n最终答案: {final_answer}")
        
        # 保存到历史记录
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))
        
        return final_answer