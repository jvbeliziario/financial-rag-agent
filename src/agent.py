"""
agent.py
Agente conversacional de Q&A financeiro com interface pelo Gradio.

Memoria:
- Curto prazo:  ultimas SHORT_TERM_K trocas mantidas em memoria (InMemoryChatMessageHistory)
- Medio prazo:  resumo da sessao atual gerado pelo LLM a cada SHORT_TERM_K turnos
- Longo prazo:  resumos de sessoes anteriores persistidos em JSON
- rewrite: caso a pergunta tenha referencia anaforica(this, isso, that), LLm reescreve a pergunta de forma autocontida
"""

import os
import re
import json
import sys
from datetime import datetime
from pathlib import Path
from langchain_community.llms import Ollama
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

# Chamada ao retriever ***********************************************************
try:
    from src.retriever import build_retriever
except Exception:
    from retriever import build_retriever

# Configuracoes ****************************************************************

LLM_MODEL          = "qwen2.5:7b"
MEMORY_FILE        = Path(__file__).resolve().parent.parent / "memory" / "long_term.json"
SHORT_TERM_K       = 6
MAX_PAST_SUMMARIES = 3
LLM_NUM_PREDICT    = 384
LLM_NUM_CTX        = 4096
LLM_NUM_THREAD     = os.cpu_count() or 4


# Prompts ****************************************************************

SYSTEM_PROMPT = """\
You are a precise financial analyst assistant.
Answer questions based on real financial documents (10-K, 10-Q, earnings reports).

Guidelines:
- Base your answers strictly on the provided context chunks.
- If the answer is not in the context, say so — do not hallucinate figures.
- When citing numbers, mention the document and period they come from.
- Keep answers concise. Use bullet points for multi-part answers.
- If the user asks a follow-up without specifying a company, use only the
  provided context; if ambiguity remains, ask which company.

Session summary: {session_summary}

Relevant chunks:
{context}

Conversation history:
{history}
"""

SUMMARIZATION_PROMPT = """\
Summarize the following financial Q&A conversation in 3-5 sentences.
Focus on: companies discussed, key figures mentioned, and main conclusions.

Conversation:
{conversation}

Summary:
"""

REWRITE_PROMPT = """\
Given the conversation history and a follow-up question, rewrite the question
as a single standalone search query. Keep it short and factual.
If the question already makes sense on its own, return it unchanged.

History:
{history}

Follow-up question: {question}

Standalone query:"""

# Query rewriting ****************************************************************

# Termos que indicam que a pergunta depende do contexto anterior para fazer sentido.
# Quando detectados, o rewrite e disparado antes de chamar o retriever.
_REFERENTIAL = re.compile(
    r"\b(this|it|that|its|these|those|isso|ele|ela|disso|the same|such)\b",
    re.IGNORECASE,
)

def needs_rewrite(question: str) -> bool:
    return bool(_REFERENTIAL.search(question))

def rewrite_query(llm: Ollama, question: str, history_text: str) -> str:
    """
    Reformula a pergunta atual em uma query autocontida usando o historico.
    Caso a query contenha referencias anaforicas, o LLM e chamado para reescrever a pergunta de forma autocontida.
    Se o LLM falha, a pergunta original retorna como fallback
    """
    if not history_text or history_text == "No previous messages.":
        return question
    try:
        prompt = REWRITE_PROMPT.format(history=history_text, question=question)
        result = llm.invoke(prompt).strip()
        return result.splitlines()[0] if result else question
    except Exception:
        return question


# Memoria de longo prazo ****************************************************************

def load_long_term_memory() -> dict:
    if MEMORY_FILE.exists():
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "past_summaries": [],
        "last_session":   None,
    }

def save_long_term_memory(memory: dict) -> None:
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)

def update_long_term_memory(memory: dict, session_summary: str) -> None:
    if session_summary:
        entry = {
            "date":datetime.now().strftime("%Y-%m-%d %H:%M"),
            "summary": session_summary,
        }
        memory["past_summaries"].append(entry)
        memory["past_summaries"] = memory["past_summaries"][-MAX_PAST_SUMMARIES:]

    memory["last_session"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    save_long_term_memory(memory)

# Memoria de medio prazo ****************************************************************

def format_history_as_text(messages: list) -> str:
    lines = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            lines.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            lines.append(f"Assistant: {msg.content}")
    return "\n".join(lines)

def summarize_conversation(llm: Ollama, conversation_text: str) -> str:
    if not conversation_text.strip():
        return ""
    try:
        prompt = SUMMARIZATION_PROMPT.format(conversation=conversation_text)
        return llm.invoke(prompt).strip()
    except Exception as e:
        print(f"Summarization failed: {e}")
        return ""


# Estado global da sessao ****************************************************************

_retriever = None
_llm = None
_short_term_mem  = None
_long_term_mem = None
_session_summary = ""
_turn_count = 0

def initialize_agent():
    global _retriever, _llm, _short_term_mem, _long_term_mem, _session_summary

    if _llm is not None:
        return

    _llm = Ollama(
        model=LLM_MODEL,
        temperature=0,
        num_predict=LLM_NUM_PREDICT,
        num_ctx=LLM_NUM_CTX,
        num_thread=LLM_NUM_THREAD,
    )

    _retriever = build_retriever()
    _short_term_mem = InMemoryChatMessageHistory()
    _long_term_mem = load_long_term_memory()

    if _long_term_mem.get("past_summaries"):
        last = _long_term_mem["past_summaries"][-1]
        _session_summary = f"[Previous session on {last['date']}]: {last['summary']}"

# Funcao principal de resposta ****************************************************************

def answer_question(message: str, history: list) -> str:
    global _session_summary, _turn_count

    initialize_agent()

    message = message.strip()
    if not message:
        return "Please type a question."

    if any(p in message.lower() for p in ["show memory", "what do you remember"]):
        return _format_memory_report()

    # Monta o historico antes do rewrite 
    messages = list(_short_term_mem.messages) if _short_term_mem else []
    history_text = format_history_as_text(messages[-SHORT_TERM_K * 2:]) or "No previous messages."

    # Se a pergunta contem referencia anaforica, reescreve antes de buscar
    search_query = rewrite_query(_llm, message, history_text) if needs_rewrite(message) else message

    # Recupera chunks com a query 
    chunks = _retriever.invoke(search_query)
    context = "\n\n---\n\n".join(
        f"[{doc.metadata.get('company', '?')} | {doc.metadata.get('year', '?')} | "
        f"p.{doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in chunks
    )

    # Contexto de longo prazo
    past_context = ""
    if _long_term_mem.get("past_summaries"):
        entries = _long_term_mem["past_summaries"][-2:]
        past_context = "\n".join(f"- [{e['date']}]: {e['summary']}" for e in entries)

    full_summary = _session_summary
    if past_context:
        full_summary += f"\n\nPast sessions:\n{past_context}"

    # Chamada ao LLM
    prompt = SYSTEM_PROMPT.format(
        session_summary=full_summary or "No summary yet.",
        context=context or "No relevant chunks found.",
        history=history_text,
    ) + f"\nUser: {message}\nAssistant:"

    try:
        response = _llm.invoke(prompt)
    except Exception as e:
        return (
            f"LLM error: {e}\n\n"
            f"Make sure Ollama is running (ollama serve) and the model is available "
            f"(ollama pull {LLM_MODEL})."
        )

    # Atualiza curto prazo
    _short_term_mem.add_user_message(message)
    _short_term_mem.add_ai_message(response)
    if len(_short_term_mem.messages) > SHORT_TERM_K * 2:
        del _short_term_mem.messages[:-SHORT_TERM_K * 2]

    _turn_count += 1

    # Sumarizacao de medio prazo
    if _turn_count % SHORT_TERM_K == 0:
        conversation_text = format_history_as_text(list(_short_term_mem.messages))
        new_summary = summarize_conversation(_llm, conversation_text)
        if new_summary:
            _session_summary = (
                (_session_summary + "\n\n" + new_summary).strip()
                if _session_summary else new_summary
            )

    return response

# Relatorio de memoria ****************************************************************

def _format_memory_report() -> str:
    lt    = _long_term_mem or {}
    lines = [
        "Memory Report",
        f"Last session: {lt.get('last_session', 'N/A')}",
        f"Session summary: {_session_summary or 'No summary yet.'}",
    ]

    past = lt.get("past_summaries", [])
    if past:
        lines.append("\nPast sessions:")
        for entry in past[-3:]:
            lines.append(f"  [{entry['date']}]: {entry['summary']}")
    else:
        lines.append("\nNo past sessions stored yet.")

    return "\n".join(lines)

# Callbacks para o Gradio ****************************************************************

def clear_session():
    global _session_summary, _turn_count

    if _short_term_mem:
        _short_term_mem.clear()

    _turn_count  = 0
    _session_summary = ""

    if _long_term_mem:
        update_long_term_memory(_long_term_mem, "")

    return [], _get_status_text()


def _get_status_text() -> str:
    if _retriever is None:
        return "Agent not initialized yet."

    lt = _long_term_mem or {}
    past_count = len(lt.get("past_summaries", []))
    last = lt.get("last_session", "never")

    return (
        f"Past sessions stored: {past_count}\n"
        f"Last session: {last}\n"
        f"Current turn: {_turn_count}"
    )

def on_session_end():
    if _long_term_mem:
        update_long_term_memory(_long_term_mem, _session_summary)

if __name__ == "__main__":
    try:
        from . import interface as ui
    except Exception:
        import interface as ui

    app = ui.build_gradio_app(sys.modules[__name__])
    app.launch(
        share=False,
        show_error=True,
        inbrowser=True,
    )