import genai_core.clients

# from langchain.llms import Bedrock
from langchain.prompts.prompt import PromptTemplate

from .base import Bedrock
from ..base import ModelAdapter
from genai_core.registry import registry


class BedrockClaudeAdapter(ModelAdapter):
    def __init__(self, model_id, *args, **kwargs):
        self.model_id = model_id

        super().__init__(*args, **kwargs)

    def get_llm(self, model_kwargs={}):
        bedrock = genai_core.clients.get_bedrock_client()
        params = {}
        if "temperature" in model_kwargs:
            params["temperature"] = model_kwargs["temperature"]
        if "topP" in model_kwargs:
            params["top_p"] = model_kwargs["topP"]
        if "maxTokens" in model_kwargs:
            params["max_tokens"] = model_kwargs["maxTokens"]

        params["anthropic_version"] = "bedrock-2023-05-31"
        return Bedrock(
            client=bedrock,
            model_id=self.model_id,
            model_kwargs=params,
            streaming=model_kwargs.get("streaming", False),
            callbacks=[self.callback_handler],
        )

    def get_qa_prompt(self,variables=None):
        template = """
            Sistema: Desempenhe o papel de bot de FAQ, que está analisando os documentos da RFP para resumir e responder as perguntas sobre o texto em <context>.
            <context>
            {context}
            </context>
            Ao responder à pergunta, use a voz ativa, responda em português do Brasil. Em vez do bot de FAQ, use "nós" da {CompanyName}.
            Na resposta, não use frases como "com base no contexto fornecido". A documentação indica que, em vez disso, use "nós".
            Não gere ou proponha novas ações, novas informações, novas estatísticas ou novas métricas que não estejam no contexto. Não use números da pergunta.
            Em vez de "eu", use "nós" da {CompanyName}.
            Se a pergunta não puder ser respondida a partir do contexto, sempre responda como "Não foi possível responder: Nenhuma informação disponível" para responder com as informações atuais disponíveis para a pergunta {question}.
            Usuário: {question}
            """

        return PromptTemplate(
                template=template, input_variables=["context", "question"], partial_variables=variables
            )
                

    def get_prompt(self):
        template = """A seguir, uma conversa amigável entre um humano e uma IA. Se a IA não sabe a resposta para uma pergunta, ela diz sinceramente que não sabe.
Conversa corrente:
{chat_history}

Pergunta: {input}"""

        input_variables = ["input", "chat_history"]
        prompt_template_args = {
            "chat_history": "{chat_history}",
            "input_variables": input_variables,
            "template": template,
        }
        prompt_template = PromptTemplate(**prompt_template_args)

        return prompt_template

    def get_condense_question_prompt(self):
        template = """<conv>
{chat_history}
</conv>

<followup>
{question}
</followup>

Dada a conversa dentro das tags <conv></conv>, reformule a pergunta de acompanhamento que você encontra dentro de <followup></followup> para ser uma pergunta independente, na mesma língua que a pergunta de acompanhamento.
"""

        return PromptTemplate(
            input_variables=["chat_history", "question"],
            chat_history="{chat_history}",
            template=template,
        )


# Register the adapter
registry.register(r"^bedrock.anthropic.claude*", BedrockClaudeAdapter)
