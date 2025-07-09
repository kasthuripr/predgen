import requests
def call_llm_with_tools(messages, tools=None, model="phi3-mini"):
    payload = {
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",  # Let the model decide whether to call a tool
        "temperature": 0.3
    }

    response = requests.post("http://localhost:1234/v1/chat/completions", json=payload)
    return response.json()


