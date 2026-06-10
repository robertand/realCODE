class QuestionTool:
    name = "question"
    description = "Ask the user a question and get their response. Use this when you need clarification or a decision."

    schema = {
        "description": "Ask the user a question",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask the user"
                }
            },
            "required": ["question"]
        }
    }

    def execute(self, question: str) -> str:
        print(f"\n\033[93m[QUESTION] {question}\033[0m")
        answer = input("Your answer: ")
        return f"[user answered: {answer}]"