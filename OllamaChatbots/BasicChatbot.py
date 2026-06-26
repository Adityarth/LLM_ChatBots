from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0.7
)

print("qwen2.5 Chatbot")
print("Type 'exit' to quit\n")

while True:
    prompt = input("You: ")

    if prompt.lower() == "exit":
        break

    response = llm.invoke(prompt)
    print("\nGemma:", response.content)
    print()
