import json

from langchain.schema import AIMessage, HumanMessage
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate


Llama2ChatPrompt = """<s>[INST] <<SYS>>
Você é um assistente prestativo que fornece respostas concisas às perguntas dos usuários com o mínimo de frases possível, no máximo 3. Você não se repete. Evita listas com marcadores ou emojis.
<</SYS>>

{chat_history}<s>[INST] Contexto: {input} [/INST]"""

Llama2ChatQAPrompt = """<s>[INST] <<SYS>>
Use o histórico da conversa a seguir e trechos de contexto para responder à pergunta no final. Se você não souber a resposta, apenas diga que não sabe, não tente inventar uma resposta. Não se repita. Evite listas com marcadores ou emojis.
<</SYS>>

{chat_history}<s>[INST] Contexto: {context}

{question} [/INST]"""

Llama2ChatCondensedQAPrompt = """<s>[INST] <<SYS>>
Dada a seguinte conversa e a pergunta no final, reformule a entrada de acompanhamento para ser uma pergunta independente, na mesma língua que a entrada de acompanhamento. Você não se repete. Você evita listas com marcadores ou emojis.
<</SYS>>

{chat_history}<s>[INST] {question} [/INST]"""


Llama2ChatPromptTemplate = PromptTemplate.from_template(Llama2ChatPrompt)
Llama2ChatQAPromptTemplate = PromptTemplate.from_template(Llama2ChatQAPrompt)
Llama2ChatCondensedQAPromptTemplate = PromptTemplate.from_template(
    Llama2ChatCondensedQAPrompt
)


class Llama2ConversationBufferMemory(ConversationBufferMemory):
    @property
    def buffer_as_str(self) -> str:
        return self.get_buffer_string()

    def get_buffer_string(self) -> str:
        """modified version of https://github.com/langchain-ai/langchain/blob/bed06a4f4ab802bedb3533021da920c05a736810/libs/langchain/langchain/schema/messages.py#L14"""
        human_message_cnt = 0
        string_messages = []
        for m in self.chat_memory.messages:
            if isinstance(m, HumanMessage):
                if human_message_cnt == 0:
                    message = f"{m.content} [/INST]"
                else:
                    message = f"<s>[INST] {m.content} [/INST]"
                human_message_cnt += 1
            elif isinstance(m, AIMessage):
                message = f"{m.content} </s>"
            else:
                raise ValueError(f"Got unsupported message type: {m}")
            string_messages.append(message)

        return "".join(string_messages)
