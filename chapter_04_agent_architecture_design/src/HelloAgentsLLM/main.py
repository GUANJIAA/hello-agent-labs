import os
from openai import OpenAI
from typing import List, Dict

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

if __name__ == '__main__':
    try:
        llmClient = HelloAgentsLLM()

        exampleMessage = [
            {"role": "system", "content": "You are a helpful assistant that writes Python code."},
            {"role": "user", "content": "写一个快速排序算法"}
        ]

        print("---调用LLM---")
        responseText = llmClient.think(exampleMessage)
        if responseText:
            print("\n\n---完整模型响应---")
            print(responseText)
    except ValueError as e:
        print(e)
