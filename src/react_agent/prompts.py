"""Default prompts used by the agent."""

SYSTEM_PROMPT = """You are a helpful AI assistant.

System time: {system_time}"""

SUMMARIZATION_PROMPT = """Summarize the following conversation concisely, \
preserving key facts, decisions, and context that would be needed to \
continue the conversation. Focus on what the user asked, what was \
found/decided, and any ongoing tasks.

Conversation:
{conversation}

Summary:"""

TITLE_PROMPT = """다음 사용자 메시지를 한줄 제목(20자 이내)으로 요약해줘. 제목만 출력해.

메시지: {message}

제목:"""
