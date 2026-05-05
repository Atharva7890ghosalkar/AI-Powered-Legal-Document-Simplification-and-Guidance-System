import json
import re
import sys
from collections.abc import Iterator
from pathlib import Path

import requests
import streamlit as st

# Ensure project root is importable when Streamlit runs from frontend/ context.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import get_settings
from src.services.local_chat_db import LocalChatDB

settings = get_settings()
API_BASE = settings.backend_base_url.rstrip("/")
BACKEND_PUBLIC_URL = settings.backend_public_url.rstrip("/")
CHAT_DB = LocalChatDB()

st.set_page_config(page_title="Legal Guidance Assistant", page_icon="⚖️", layout="wide")


def _get_query_param(name: str) -> str | None:
    value = st.query_params.get(name)
    if value is None:
        return None
    if isinstance(value, list):
        return value[0] if value else None
    return str(value)


def _clear_auth_query_params() -> None:
    for key in ["auth_token", "auth_error"]:
        if key in st.query_params:
            del st.query_params[key]


def _build_google_auth_url() -> str:
    return f"{BACKEND_PUBLIC_URL}/auth/google/login?next={settings.frontend_base_url}"


def _fetch_current_user(auth_token: str) -> dict | None:
    try:
        response = requests.get(
            f"{API_BASE}/auth/me",
            headers={"Authorization": f"Bearer {auth_token}"},
            timeout=15,
        )
        if not response.ok:
            return None
        return response.json()
    except Exception:
        return None


def _is_google_auth_configured() -> bool:
    return bool(settings.google_client_id and settings.google_client_secret and settings.google_redirect_uri)


def _get_guest_user() -> dict:
    return {
        "id": "local-guest-user",
        "email": "guest@local.dev",
        "name": "Local Guest",
        "picture": "",
    }


def _logout() -> None:
    for key in [
        "user",
        "auth_token",
        "chat_owner_id",
        "chats",
        "active_chat_id",
        "chat_selector_id",
        "upload_response",
    ]:
        if key in st.session_state:
            del st.session_state[key]
    _clear_auth_query_params()


if "user" not in st.session_state:
    st.session_state.user = None
if "auth_token" not in st.session_state:
    st.session_state.auth_token = None

incoming_token = _get_query_param("auth_token")
incoming_error = _get_query_param("auth_error")
if incoming_token:
    st.session_state.auth_token = incoming_token
    _clear_auth_query_params()
    st.rerun()
if incoming_error:
    st.error(f"Google sign-in failed: {incoming_error}")
    _clear_auth_query_params()

if not _is_google_auth_configured():
    if not st.session_state.user:
        st.session_state.user = _get_guest_user()
    st.info("Google sign-in is not configured. Running in local guest mode.")

if st.session_state.auth_token and not st.session_state.user:
    current_user = _fetch_current_user(st.session_state.auth_token)
    if current_user:
        st.session_state.user = current_user
        CHAT_DB.upsert_user(
            user_id=current_user["id"],
            email=current_user.get("email"),
            name=current_user.get("name"),
            picture=current_user.get("picture"),
        )
    else:
        st.session_state.auth_token = None

if not st.session_state.user:
    st.title("Legal Guidance Assistant")
    st.caption("Sign in with Google to access your personal multi-chat history.")
    st.link_button("Sign in with Google", _build_google_auth_url(), use_container_width=True)
    st.stop()

user = st.session_state.user
user_id = user["id"]

if st.session_state.get("chat_owner_id") != user_id:
    stored_chats = CHAT_DB.list_chats(user_id)
    if not stored_chats:
        stored_chats = [CHAT_DB.create_chat("Chat 1", user_id=user_id)]
    st.session_state.chats = stored_chats
    st.session_state.active_chat_id = st.session_state.chats[0]["id"]
    st.session_state.chat_selector_id = st.session_state.active_chat_id
    st.session_state.chat_owner_id = user_id

if "upload_response" not in st.session_state:
    st.session_state.upload_response = None

st.title("AI Legal Document Simplification and Guidance")
st.caption("Indian legal clauses, risks, and guidance in plain English.")


def get_active_chat() -> dict:
    for chat in st.session_state.chats:
        if chat["id"] == st.session_state.active_chat_id:
            return chat
    st.session_state.active_chat_id = st.session_state.chats[0]["id"]
    return st.session_state.chats[0]


def create_new_chat() -> None:
    next_idx = get_next_chat_number()
    new_chat = CHAT_DB.create_chat(f"Chat {next_idx}", user_id=user_id)
    st.session_state.chats.append(new_chat)
    st.session_state.active_chat_id = new_chat["id"]
    st.session_state.chat_selector_id = new_chat["id"]


def get_next_chat_number() -> int:
    max_num = 0
    for chat in st.session_state.chats:
        title = str(chat.get("title", "")).strip().lower()
        if not title.startswith("chat "):
            continue
        suffix = title.replace("chat ", "", 1).strip()
        if suffix.isdigit():
            max_num = max(max_num, int(suffix))
    return max_num + 1 if max_num > 0 else 1


def delete_active_chat() -> None:
    active_id = st.session_state.active_chat_id
    CHAT_DB.delete_chat(active_id, user_id=user_id)
    st.session_state.chats = [chat for chat in st.session_state.chats if chat["id"] != active_id]

    if not st.session_state.chats:
        fallback = CHAT_DB.create_chat("Chat 1", user_id=user_id)
        st.session_state.chats = [fallback]
        st.session_state.active_chat_id = fallback["id"]
        st.session_state.chat_selector_id = fallback["id"]
        return

    next_chat = st.session_state.chats[0]
    st.session_state.active_chat_id = next_chat["id"]
    st.session_state.chat_selector_id = next_chat["id"]


def ensure_chat_messages_loaded(chat: dict) -> None:
    if chat.get("messages"):
        return
    chat["messages"] = CHAT_DB.load_messages(chat["id"])


def analyze_pdf(uploaded_file) -> None:
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
    with st.spinner("Analyzing document..."):
        response = requests.post(f"{API_BASE}/upload", files=files, timeout=180)
    if response.ok:
        active_chat = get_active_chat()
        upload_data = response.json()
        active_chat["upload_response"] = upload_data
        st.session_state.upload_response = upload_data
        CHAT_DB.save_upload_response(chat_id=active_chat["id"], upload_response=upload_data)
        with st.spinner("Preparing instant explanation..."):
            add_auto_document_overview(active_chat)
        st.success("Document analyzed.")
    else:
        st.error(response.text)


def stream_answer(payload: dict) -> Iterator[tuple[str, list[dict], list[dict], str | None]]:
    answer_parts: list[str] = []
    retrieved_context: list[dict] = []
    citations: list[dict] = []
    error_message: str | None = None

    response = requests.post(
        f"{API_BASE}/ask/stream",
        json=payload,
        timeout=300,
        stream=True,
        headers={"Accept": "text/event-stream"},
    )
    if not response.ok:
        yield "", [], [], response.text
        return

    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        if not raw_line.startswith("data: "):
            continue
        data = raw_line[6:]
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue

        event_type = event.get("type")
        if event_type == "context":
            retrieved_context = event.get("retrieved_context", [])
            citations = event.get("citations", [])
        elif event_type == "token":
            delta = event.get("delta", "")
            answer_parts.append(delta)
            yield "".join(answer_parts), retrieved_context, citations, None
        elif event_type == "error":
            error_message = event.get("message", "Streaming failed")
            break
        elif event_type == "done":
            if not answer_parts and event.get("answer"):
                answer_parts = [event.get("answer", "")]
            retrieved_context = event.get("retrieved_context", retrieved_context)
            citations = event.get("citations", citations)

    yield "".join(answer_parts), retrieved_context, citations, error_message


def build_document_context(upload_data: dict | None) -> str | None:
    if not upload_data:
        return None

    summary = upload_data.get("summary", "")
    clauses = upload_data.get("clauses", []) or []
    if not clauses and not summary:
        return None

    date_pattern = re.compile(
        r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b",
        flags=re.IGNORECASE,
    )
    date_clauses = [c for c in clauses if date_pattern.search((c or {}).get("clause_text", ""))]
    top_clauses = clauses[:6]
    important = date_clauses[:6]

    lines = [f"Document Summary: {summary}"] if summary else []
    if top_clauses:
        lines.append("Key Clauses:")
        for item in top_clauses:
            text = (item.get("clause_text", "") or "").strip()
            if text:
                lines.append(f"- {text[:600]}")
    if important:
        lines.append("Date-Relevant Clauses:")
        for item in important:
            text = (item.get("clause_text", "") or "").strip()
            if text:
                lines.append(f"- {text[:600]}")

    context = "\n".join(lines).strip()
    return context[:6000] if context else None


def run_stream_request(payload: dict) -> tuple[str, list[dict], list[dict], str | None]:
    latest_text = ""
    latest_context: list[dict] = []
    latest_citations: list[dict] = []
    stream_error: str | None = None
    for text, context, citations, err in stream_answer(payload):
        latest_text = text
        latest_context = context
        latest_citations = citations
        stream_error = err
    return latest_text, latest_context, latest_citations, stream_error


def add_auto_document_overview(active_chat: dict) -> None:
    upload_data = active_chat.get("upload_response")
    if not upload_data:
        return

    payload = {
        "question": (
            "Explain this uploaded legal document in a smooth and practical way. "
            "Cover document purpose, key obligations, important dates, major risks, and next steps."
        ),
        "clause_text": build_document_context(upload_data),
    }
    latest_text, latest_context, latest_citations, stream_error = run_stream_request(payload)
    if stream_error:
        active_chat["messages"].append({"role": "assistant", "text": f"Error: {stream_error}"})
        return
    if latest_text.strip():
        CHAT_DB.save_message(
            chat_id=active_chat["id"],
            role="assistant",
            text=latest_text,
            retrieved_context=latest_context,
            citations=latest_citations,
        )
        active_chat["messages"].append(
            {
                "role": "assistant",
                "text": latest_text,
                "retrieved_context": latest_context,
                "citations": latest_citations,
            }
        )


with st.sidebar:
    st.subheader("Account")
    st.write(user.get("name", "User"))
    st.caption(user.get("email", ""))
    if _is_google_auth_configured():
        if st.button("Sign out", use_container_width=True):
            _logout()
            st.rerun()
    else:
        st.caption("Guest mode is active for local development.")

    st.subheader("Chat Sessions")
    if st.button("New Chat", use_container_width=True, key="new_chat_btn"):
        create_new_chat()
        st.rerun()
    if st.button("Delete Current Chat", use_container_width=True, key="delete_chat_btn"):
        delete_active_chat()
        st.rerun()

    chat_ids = [chat["id"] for chat in st.session_state.chats]
    chat_title_map = {chat["id"]: chat["title"] for chat in st.session_state.chats}
    selected_chat_id = st.selectbox(
        "Select chat",
        options=chat_ids,
        format_func=lambda cid: chat_title_map.get(cid, "Chat"),
        key="chat_selector_id",
    )
    st.session_state.active_chat_id = selected_chat_id

    st.subheader("Document Upload")
    sidebar_uploaded = st.file_uploader("Upload legal PDF", type=["pdf"], key="sidebar_pdf_upload")
    if sidebar_uploaded and st.button("Analyze Document", use_container_width=True, key="sidebar_analyze_btn"):
        analyze_pdf(sidebar_uploaded)

active_chat = get_active_chat()
ensure_chat_messages_loaded(active_chat)
upload_data = active_chat.get("upload_response")
if upload_data:
    st.subheader("Document Summary")
    st.write(upload_data.get("summary", ""))

    risks = upload_data.get("risks", [])
    risky = [r for r in risks if r.get("risk_level") in {"high", "medium"}]
    st.subheader("Detected Risks")
    if risky:
        for risk in risky[:12]:
            st.markdown(
                f"- **Clause {risk['clause_id']}** `{risk['clause_type']}`: `{risk['risk_level']}` | Triggers: {', '.join(risk['triggers']) or 'none'}"
            )
    else:
        st.write("No medium/high risks detected.")

st.subheader("Legal Assistant Chat")
for item in active_chat["messages"]:
    with st.chat_message(item["role"]):
        st.markdown(item["text"])
        if item["role"] == "assistant" and item.get("retrieved_context"):
            with st.expander("Retrieved Legal Context"):
                for chunk in item["retrieved_context"]:
                    st.markdown(f"- {chunk.get('text', '')}")
        if item["role"] == "assistant" and item.get("citations"):
            with st.expander("Citations"):
                for citation in item["citations"]:
                    st.markdown(
                        f"- **{citation.get('citation', 'Reference')}** | Section: `{citation.get('section', 'Not found')}` | "
                        f"Type: `{citation.get('reference_type', 'legal_reference')}`"
                    )

user_input = st.chat_input("Ask about a clause, risk, or legal obligation...")
if user_input:
    CHAT_DB.save_message(chat_id=active_chat["id"], role="user", text=user_input)
    active_chat["messages"].append({"role": "user", "text": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    payload = {
        "question": user_input,
        "clause_text": build_document_context(active_chat.get("upload_response")),
    }
    with st.chat_message("assistant"):
        placeholder = st.empty()
        latest_text = ""
        latest_context: list[dict] = []
        latest_citations: list[dict] = []
        stream_error: str | None = None

        with st.spinner("Generating guidance..."):
            for text, context, citations, err in stream_answer(payload):
                latest_text = text
                latest_context = context
                latest_citations = citations
                stream_error = err
                if latest_text:
                    placeholder.markdown(latest_text)

        if stream_error:
            error_text = f"Error: {stream_error}"
            placeholder.markdown(error_text)
            active_chat["messages"].append({"role": "assistant", "text": error_text})
            CHAT_DB.save_message(chat_id=active_chat["id"], role="assistant", text=error_text)
        else:
            if latest_context:
                with st.expander("Retrieved Legal Context"):
                    for chunk in latest_context:
                        st.markdown(f"- {chunk.get('text', '')}")
            if latest_citations:
                with st.expander("Citations"):
                    for citation in latest_citations:
                        st.markdown(
                            f"- **{citation.get('citation', 'Reference')}** | Section: `{citation.get('section', 'Not found')}` | "
                            f"Type: `{citation.get('reference_type', 'legal_reference')}`"
                        )
            active_chat["messages"].append(
                {
                    "role": "assistant",
                    "text": latest_text,
                    "retrieved_context": latest_context,
                    "citations": latest_citations,
                }
            )
            CHAT_DB.save_message(
                chat_id=active_chat["id"],
                role="assistant",
                text=latest_text,
                retrieved_context=latest_context,
                citations=latest_citations,
            )
