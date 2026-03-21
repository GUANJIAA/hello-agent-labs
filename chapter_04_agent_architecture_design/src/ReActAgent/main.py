import os
import re
from openai import OpenAI
from serpapi import GoogleSearch
from typing import Dict, Any, List

# ReAct 提示词模板
REACT_PROMPT_TEMPLATE = """
请注意，你是一个有能力调用外部工具的智能助手。
可用工具如下:
{tools}
请严格按照以下格式进行回应:
Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一:
- `{{tool_name}}[{{tool_input}}]`:调用一个可用工具。
- `Finish[最终答案]`:当你认为已经获得最终答案时。
- 当你收集到足够的信息，能够回答用户的最终问题时，你必须在Action:字段后使用 Finish[最终答案] 来输
出最终答案。
现在，请开始解决以下问题:
Question: {question}
History: {history}
"""

def search(query: str) -> str:
    """
    一个基于SerpAPI的实战网页搜索引擎工具
    它会智能地解析搜索结果，提取出相关的信息
    """
    print(f"搜索: {query}")
    try:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return "SERPAPI_API_KEY 环境变量未设置"
        params = {
            "engine": "google",
            "api_key": api_key,
            "q": query,
            "gl": "cn",
            "hl": "zh-cn",
        }

        client = GoogleSearch(params)
        results = client.get_dict()

        # 智能解析:优先寻找最直接的答案
        if "answer_box_list" in results:
            return "\n".join(results["answer_box_list"])
        if "answer_box" in results and "answer" in results["answer_box"]:
            return results["answer_box"]["answer"]
        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            return results["knowledge_graph"]["description"]
        if "organic_results" in results and results["organic_results"]:
            # 如果没有直接答案，则返回前三个有机结果的摘要
            snippets = [
                f"[{i+1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                for i, res in enumerate(results["organic_results"][:3])
            ]
            return "\n\n".join(snippets)
        return f"对不起，没有找到关于 '{query}' 的信息。"
    except Exception as e:
        return f"搜索时发生错误: {e}"

class ToolExecutor:
    """
    一个工具执行器，负责管理和执行工具。
    """
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

    def registerTool(self, name: str, description: str, func: callable):
        """
        注册一个工具。
        :param name: 工具的名称
        :param description: 工具的描述
        :param func: 工具的执行函数
        """
        if name in self.tools:
            print(f"工具 {name} 已存在，无法重复注册")
            return
        self.tools[name] = {
            "description": description,
            "function": func
        }
        print(f"工具 {name} 已注册")

    def getTool(self, name: str) -> callable:
        """
        获取一个工具的执行函数。
        :param name: 工具的名称
        :return: 工具的执行函数
        """
        if name not in self.tools:
            print(f"工具 {name} 不存在")
            return None
        return self.tools[name]["function"]

    def getAvailableTools(self) -> str:
        """
        获取所有已注册工具的描述。
        :return: 所有已注册工具的描述
        """
        return "\n".join([f"{name}: {desc}" for name, desc in self.tools.items()])

class HelloAgentsLLM:
    def __init__(self, model: str = None, apiKey: str = None, baseUrl: str = None, timeout: int = None) -> None:
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

class ReActAgent:
    """
    一个ReAct智能体，能够调用外部工具来解决用户的问题。
    """
    def __init__(self, llm_client: HelloAgentsLLM, toolExecutor: ToolExecutor, max_steps: int = 5):
        self.llm_client = llm_client
        self.tools = toolExecutor
        self.max_steps = max_steps
        self.history = []

    def run(self, question: str) -> str:
        """
        运行ReAct智能体，解决用户的问题。
        :param question: 用户的问题
        :return: 智能体的回复
        """
        self.history = []
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1
            print(f"第 {current_step} 步")
            
            # 1.格式化提示词
            tools_description = self.tools.getAvailableTools()
            history_str = "\n".join(self.history)
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=tools_description,
                question=question,
                history=history_str
            )

            # 2.调用LLM进行思考
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.think(messages)

            if not response_text:
                print("大语言模型返回空响应")
                continue

            # 3.解析LLM的输出
            thought, action = self._parse_output(response_text)
            if thought:
                print(f"思考:{thought}")
            if not action:
                print("警告：未能解析出有效的Action, 流程终止")
                break

            # 4.执行Action
            if action.startswith("Finish"):
                # 如果是Finish指令,提取最终答案并结束
                final_answer = re.match(r"Finish\[(.*)\]", action).group(1)
                print(f"最终答案：{final_answer}")
                return final_answer

            tool_name, tool_input = self._parse_action(action)
            if not tool_name or not tool_input:
                print("警告：未能解析出有效的工具调用, 流程终止")
                continue

            print(f"调用工具 {tool_name}，输入 {tool_input}")

            tool_function = self.tools.getTool(tool_name)
            if not tool_function:
                observation = f"错误：工具 {tool_name} 未注册"
            else:
                observation = tool_function(tool_input)
            print(f"工具 {tool_name} 输出：{observation}")

            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")
        
        print("流程结束")
        print(f"最终历史：{self.history}")
        return ""

    def _parse_output(self, text: str):
        """解析LLM的输出，提取Thought和Action。
        """
        # Thought: 匹配到 Action: 或文本末尾
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text,
        re.DOTALL)
        # Action: 匹配到文本末尾
        action_match = re.search(r"Action:\s*(.*?)$", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_action(self, action_text: str):
        """解析Action字符串，提取工具名称和输入。
        """
        match = re.match(r"(\w+)\[(.*)\]", action_text, re.DOTALL)
        if match:
            return match.group(1), match.group(2)
        return None, None

if __name__ == "__main__":
    # 初始化组件
    executor = ToolExecutor()
    executor.registerTool("search", "一个网页搜索工具，用于获取最新信息。", search)
    
    llm = HelloAgentsLLM()
    agent = ReActAgent(llm, executor)
    
    # 测试
    question = "2024年巴黎奥运会金牌榜前三名是谁？"
    print(f"开始解决问题: {question}")
    agent.run(question)
       