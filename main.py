import os
import json
import uuid
from typing import TypedDict, List
import re

from flask import Flask, render_template, request, redirect, url_for, jsonify, session, flash
from flask_cors import CORS
from dotenv import load_dotenv
import bcrypt

from azure.data.tables import TableServiceClient
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import (
    generate_blob_sas,
    BlobSasPermissions,
    BlobServiceClient,
    ContentSettings,
)
from datetime import datetime, timedelta, timezone

# LangChain/LangGraph
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langchain_openai import AzureChatOpenAI

# -------------------------
#  Load environment variables
# -------------------------
load_dotenv()

# -------------------------
#  Azure OpenAI & Search Setup (BIM)
# -------------------------
openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment_name = os.getenv("AZURE_OPENAI_CHATGPT_DEPLOYMENT")
embedding_deployment_name = os.getenv("AZURE_OPENAI_EMB_DEPLOYMENT_NAME")

openai_client = AzureOpenAI(
    api_key=openai_api_key,
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
    azure_endpoint=openai_endpoint
)

search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
search_index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
search_api_key = os.getenv("AZURE_SEARCH_API_KEY")

search_client = SearchClient(
    endpoint=search_endpoint,
    index_name=search_index_name,
    credential=AzureKeyCredential(search_api_key)
)

# -------------------------
#  CEE Azure OpenAI & Search Setup
# -------------------------
cee_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
cee_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
cee_deployment_name = os.getenv("AZURE_OPENAI_CHATGPT_DEPLOYMENT")
cee_embedding_deployment_name = os.getenv("AZURE_OPENAI_EMB_DEPLOYMENT_NAME")

cee_openai_client = AzureOpenAI(
    api_key=cee_openai_api_key,
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
    azure_endpoint=cee_openai_endpoint
)

cee_search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
cee_search_index_name = os.getenv("CCE_4360_AZURE_SEARCH_INDEX_NAME")
cee_search_api_key = os.getenv("AZURE_SEARCH_API_KEY")

cee_search_client = SearchClient(
    endpoint=cee_search_endpoint,
    index_name=cee_search_index_name,
    credential=AzureKeyCredential(cee_search_api_key)
)

# -------------------------
#  Flask App Setup
# -------------------------
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "some_dev_secret_key")  # session secret
CORS(app)

# -------------------------
#  Azure Table Storage Setup
# -------------------------
connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
table_name = os.getenv("AZURE_TABLE_NAME", "Users")

table_service = TableServiceClient.from_connection_string(conn_str=connection_string)
table_client = table_service.get_table_client(table_name=table_name)

# -------------------------
#  Azure Blob Storage (User Analytics)
# -------------------------
BLOB_CONN_STR = os.getenv("AZURE_CONNECTION_STRING") or os.getenv("AZURE_STORAGE_CONNECTION_STRING")
USER_LOGS_CONTAINER = os.getenv("USER_LOGS_CONTAINER", "user-logs")  # <- container name

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _blob_service() -> BlobServiceClient:
    if not BLOB_CONN_STR:
        raise RuntimeError("Missing AZURE_CONNECTION_STRING or AZURE_STORAGE_CONNECTION_STRING")
    return BlobServiceClient.from_connection_string(BLOB_CONN_STR)

def _ensure_container(svc: BlobServiceClient, name: str):
    c = svc.get_container_client(name)
    try:
        c.create_container()
    except Exception:
        pass
    return c

def _ensure_user_prefix(container, username: str):
    """Azure Blob is flat; simulate folders via '<username>/' prefix. Create a tiny placeholder if empty."""
    prefix = f"{username}/"
    existing = list(container.list_blobs(name_starts_with=prefix))
    if not existing:
        container.get_blob_client(f"{prefix}.init").upload_blob(
            b"",
            overwrite=True,
            content_settings=ContentSettings(content_type="application/octet-stream"),
        )

def _upload_json(container, blob_path: str, data: dict):
    payload = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    container.get_blob_client(blob_path).upload_blob(
        payload,
        overwrite=True,
        content_settings=ContentSettings(content_type="application/json"),
    )

# -------------------------
#  Conversation State Schema (adds suggestions)
# -------------------------
class ChatState(TypedDict):
    messages: list     # list of LangChain message objects
    suggestions: List[str]

# -------------------------
#  Conversation Memory Setup (BIM)
# -------------------------
chat_memory = MemorySaver()
chat_workflow = StateGraph(state_schema=ChatState)
chat_model = AzureChatOpenAI(
    azure_endpoint=openai_endpoint,
    api_key=openai_api_key,
    deployment_name=deployment_name,
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview")
)

# -------------------------
#  Conversation Memory Setup (CEE)
# -------------------------
cee_memory = MemorySaver()
cee_workflow = StateGraph(state_schema=ChatState)
cee_model = AzureChatOpenAI(
    azure_endpoint=openai_endpoint,
    api_key=openai_api_key,
    deployment_name=deployment_name,
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview")
)

# -------------------------
#  Helper Functions
# -------------------------
def get_user_by_email(email):
    """Retrieves a user from Azure Table Storage by email (RowKey)."""
    try:
        entity = table_client.get_entity(partition_key="User", row_key=email)
        return entity
    except ResourceNotFoundError:
        return None
    except Exception:
        return None

def create_user(email, password):
    """Inserts new user record with bcrypt-hashed password."""
    hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    entity = {"PartitionKey": "User", "RowKey": email.lower(), "PasswordHash": hashed_pw}
    table_client.create_entity(entity=entity)

def create_user_with_referral(username, referral_code):
    """Creates user with 'ReferralCode' if not exists."""
    try:
        _ = table_client.get_entity(partition_key="User", row_key=username.lower())
        print(f"User '{username}' already exists.")
    except ResourceNotFoundError:
        entity = {"PartitionKey": "User", "RowKey": username.lower(), "ReferralCode": referral_code}
        table_client.create_entity(entity=entity)
        print(f"User '{username}' created with referral '{referral_code}'.")

def verify_password(password, stored_hash):
    """Validate plaintext vs bcrypt hash."""
    return bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8'))

def generate_embedding(text):
    """Returns embedding vector from Azure OpenAI."""
    try:
        response = openai_client.embeddings.create(
            input=text,
            model=embedding_deployment_name,
            dimensions=3072
        )
        return response.data[0].embedding
    except Exception as e:
        return f"Error generating embedding: {e}"

def cee_generate_embedding(text):
    """Returns embedding from CEE client."""
    try:
        response = cee_openai_client.embeddings.create(
            input=text,
            model=cee_embedding_deployment_name,
            dimensions=3072
        )
        return response.data[0].embedding
    except Exception as e:
        return f"Error generating embedding: {e}"

def search_matching_documents(embedding, threshold=0.5):
    """Queries Azure Cognitive Search vector index."""
    try:
        vector_query = VectorizedQuery(
            vector=embedding,
            k_nearest_neighbors=50,
            fields="embedding"
        )
        results = search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            select=["document_name", "content", "page_number", "sas_url"],
            top=5
        )
        matches = []
        for r in results:
            if r["@search.score"] >= threshold:
                matches.append({
                    "document_name": r["document_name"],
                    "page_number": r["page_number"],
                    "sas_url": r.get("sas_url")
                })
        return matches
    except Exception as e:
        return f"Error searching documents: {e}"

def cee_search_matching_documents(embedding, threshold=0.5):
    """CEE vector search."""
    try:
        vector_query = VectorizedQuery(vector=embedding, k_nearest_neighbors=50, fields="embedding")
        results = cee_search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            select=["document_name", "content", "page_number", "sas_url"],
            top=5
        )
        matches = []
        for r in results:
            if r["@search.score"] >= threshold:
                matches.append({
                    "document_name": r["document_name"],
                    "page_number": r["page_number"],
                    "sas_url": r.get("sas_url")
                })
        return matches
    except Exception as e:
        return f"Error searching documents: {e}"

def generate_refined_response(user_prompt, matching_documents):
    """Builds prompt with doc refs then calls OpenAI chat."""
    refined = f"User Query: {user_prompt}\n\nReferences:\n"
    for doc in matching_documents:
        link = f"[{doc['document_name']}, Page {doc['page_number']}]({doc['sas_url']})" if doc.get('sas_url') else f"{doc['document_name']}, Page {doc['page_number']}"
        refined += f"- {link}\n"
    response = openai_client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role":"system","content":"You are a helpful assistant."},
            {"role":"user","content":"If there is no reference then just answer precisely."},
            {"role":"user","content":refined},
            {"role":"user","content":f"Reply to '{user_prompt}' based on references."}
        ],
        temperature=0.1
    )
    return response.choices[0].message.content

def cee_generate_refined_response(user_prompt, matching_documents):
    """CEE refined response."""
    refined = f"User Query: {user_prompt}\n\nReferences:\n"
    for doc in matching_documents:
        link = f"[{doc['document_name']}, Page {doc['page_number']}]({doc['sas_url']})" if doc.get('sas_url') else f"{doc['document_name']}, Page {doc['page_number']}"
        refined += f"- {link}\n"
    response = cee_openai_client.chat.completions.create(
        model=cee_deployment_name,
        messages=[
            {"role":"system","content":"You are a helpful assistant."},
            {"role":"user","content":"If there is no reference then just answer precisely."},
            {"role":"user","content":refined},
            {"role":"user","content":f"Reply to '{user_prompt}' based on references."}
        ],
        temperature=0.1
    )
    return response.choices[0].message.content

# ---------------------------
#  Follow-up suggestion helpers (use LangGraph history)
# ---------------------------
def _serialize_history_for_prompt(history):
    """Turn LangChain messages into compact role:text lines for prompting."""
    lines = []
    for m in history[-10:]:  # keep prompt lean
        if isinstance(m, HumanMessage):
            role = "USER"
        elif isinstance(m, AIMessage):
            role = "ASSISTANT"
        elif isinstance(m, SystemMessage):
            role = "SYSTEM"
        else:
            role = "OTHER"
        text = (getattr(m, "content", "") or "").strip()
        if len(text) > 1200:
            text = text[:1200] + " …"
        lines.append(f"{role}: {text}")
    return "\n".join(lines)

def suggest_next_questions_from_history(history, client, model_name, how_many=3):
    convo = _serialize_history_for_prompt(history)
    system_msg = {
        "role": "system",
        "content": "You are a helpful assistant. Propose concise, practical follow-up questions. No chit-chat."
    }
    user_msg = {
        "role": "user",
        "content": (
            f"Conversation so far:\n{convo}\n\n"
            "Return ONLY a JSON array of 2-3 strings. "
            "Do NOT add code fences or any prose."
        )
    }

    resp = client.chat.completions.create(
        model=model_name,
        messages=[system_msg, user_msg],
        temperature=0.5,
        max_tokens=200
    )
    raw = (resp.choices[0].message.content or "").strip()

    # Strip code fences if present
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
        raw = re.sub(r"```$", "", raw).strip()

    # Try direct JSON parse
    try:
        arr = json.loads(raw)
    except Exception:
        # Extract first [...] block
        m = re.search(r"$begin:math:display$.*$end:math:display$", raw, flags=re.DOTALL)
        arr = []
        if m:
            try:
                arr = json.loads(m.group(0))
            except Exception:
                arr = []

    # Fallback line split
    if not arr or not isinstance(arr, list):
        arr = [l.strip(" ,\"'") for l in raw.splitlines() if l.strip()]

    # Final cleanup
    clean = []
    for s in arr:
        if isinstance(s, str):
            t = s.strip().strip(",").strip('"').strip("'")
            if t:
                clean.append(t)
    return clean[:how_many]

# -------------------------------------------------------------------
# BIM memory node: embed → search → inject ref → include full history → invoke → append → suggest
# -------------------------------------------------------------------
def chat_call_model(state: ChatState):
    history   = state["messages"]
    user_text = history[-1].content

    # Embedding + vector search
    emb = openai_client.embeddings.create(
        input=user_text,
        model=embedding_deployment_name,
        dimensions=3072
    ).data[0].embedding

    hits = search_client.search(
        search_text=None,
        vector_queries=[VectorizedQuery(vector=emb, k_nearest_neighbors=5, fields="embedding")],
        select=["document_name","page_number","sas_url"],
        top=5
    )

    refined = f"User Query: {user_text}\n\nReference:\n"
    for r in hits:
        if r["@search.score"] >= 0.5:
            url = r.get("sas_url","")
            refined += f"- [{r['document_name']}, Page {r['page_number']}]({url})\n"

    to_model = [
        {"role":"system","content":"You are a helpful assistant."},
        {"role":"user",  "content":"If there is no reference then just answer the user prompt precisely without extra information."},
        {"role":"user",  "content": refined},
        {"role":"user",  "content":"Please also check if any previous messages in the chat history can help answer the user prompt. if not then ignore previous messages."},
        {"role":"user",  "content":f"Reply to {user_text} based on the references provided. Mention document name and page number and sas url as hyperlink if available."},
        {"role":"user",  "content":"For an example if User user prompt is generic greeting, the response should be a generic greeting."},
        {"role":"user",  "content":"Do not say that no reference found in the document; smartly answer based on your knowledge."},
        {"role":"user",  "content":"Do not say that 'If you need further details, feel free to ask!'"}
    ]
    # Append actual conversation so far
    for msg in history:
        if isinstance(msg, HumanMessage):
            to_model.append({"role":"user",      "content":msg.content})
        elif isinstance(msg, AIMessage):
            to_model.append({"role":"assistant", "content":msg.content})
        elif isinstance(msg, SystemMessage):
            to_model.append({"role":"system",    "content":msg.content})

    resp = openai_client.chat.completions.create(
        model=deployment_name,
        messages=to_model,
        temperature=0.1,
        max_tokens=2000
    )

    ai_msg = AIMessage(content=resp.choices[0].message.content)
    next_history = history + [ai_msg]

    # NEW: generate suggestions from the same (updated) history
    suggestions = suggest_next_questions_from_history(
        history=next_history,
        client=openai_client,
        model_name=deployment_name,
        how_many=3
    )

    return {"messages": next_history, "suggestions": suggestions}

# Re-attach & compile for BIM
chat_workflow.add_edge(START, "model")
chat_workflow.add_node("model", chat_call_model)
chat_app = chat_workflow.compile(checkpointer=chat_memory)

# -------------------------------------------------------------------
# CEE memory node: same full‐history pattern + suggestions
# -------------------------------------------------------------------
def cee_call_model(state: ChatState):
    history   = state["messages"]
    user_text = history[-1].content

    emb = cee_openai_client.embeddings.create(
        input=user_text,
        model=cee_embedding_deployment_name,
        dimensions=3072
    ).data[0].embedding

    hits = cee_search_client.search(
        search_text=None,
        vector_queries=[VectorizedQuery(vector=emb, k_nearest_neighbors=5, fields="embedding")],
        select=["document_name","page_number","sas_url"],
        top=5
    )

    refined = f"User Query: {user_text}\n\nReference:\n"
    for r in hits:
        if r["@search.score"] >= 0.5:
            url = r.get("sas_url","")
            refined += f"- [{r['document_name']}, Page {r['page_number']}]({url})\n"

    to_model = [
        {"role":"system","content":"You are a helpful assistant."},
        {"role":"user",  "content":"If there is no reference then just answer the user prompt precisely without extra information."},
        {"role":"user",  "content": refined},
        {"role":"user",  "content":"Please also check if any previous messages in the chat history can help answer the user prompt. if not then ignore previous messages."},
        {"role":"user",  "content":f"Reply to {user_text} based on the references provided. Mention document name and page number and sas url as hyperlink if available."},
        {"role":"user",  "content":"For an example if User user prompt is generic greeting, the response should be a generic greeting."},
        {"role":"user",  "content":"Do not say that no reference found in the document; smartly answer based on your knowledge."},
        {"role":"user",  "content":"Do not say that 'If you need further details, feel free to ask!'"}
    ]
    for msg in history:
        if isinstance(msg, HumanMessage):
            to_model.append({"role":"user",      "content":msg.content})
        elif isinstance(msg, AIMessage):
            to_model.append({"role":"assistant", "content":msg.content})
        elif isinstance(msg, SystemMessage):
            to_model.append({"role":"system",    "content":msg.content})

    resp = cee_openai_client.chat.completions.create(
        model=cee_deployment_name,
        messages=to_model,
        temperature=0.1,
        max_tokens=2000
    )

    ai_msg = AIMessage(content=resp.choices[0].message.content)
    next_history = history + [ai_msg]

    suggestions = suggest_next_questions_from_history(
        history=next_history,
        client=cee_openai_client,
        model_name=cee_deployment_name,
        how_many=3
    )

    return {"messages": next_history, "suggestions": suggestions}

# Re-attach & compile for CEE
cee_workflow.add_edge(START, "model")
cee_workflow.add_node("model", cee_call_model)
cee_chat_app = cee_workflow.compile(checkpointer=cee_memory)

# ---------------------------
#   Auto-login helper (Qualtrics link)
# ---------------------------
def _try_auto_login_from_query() -> bool:
    """
    Auto-login when URL has ?userID=<name>&referal=MTU (or &referral=MTU).
    Creates the referral user if missing, initializes the session, and returns True on success.
    """
    user_id = (request.args.get('userID') or "").strip().lower()
    referral = (request.args.get('referal') or request.args.get('referral') or "").strip()
    valid_referral = os.getenv("REFERRAL_CODE", "MTU")

    if user_id and referral == valid_referral:
        existing_user = get_user_by_email(user_id)
        if not existing_user:
            create_user_with_referral(user_id, referral)
        _initialize_session_user(user_id)  # IMPORTANT: this should clear the session internally
        return True
    return False


# ---------------------------
#   Flask Routes (modified)
# ---------------------------
@app.route('/')
def home():
    # Auto-login from Qualtrics link if params are present and valid
    if _try_auto_login_from_query():
        return render_template('index.html')  # chat UI

    # Normal behavior
    if session.get("logged_in"):
        return render_template('index.html')  # chat UI
    else:
        return render_template('home.html')   # landing page


@app.route('/bim')
def bim_page():
    return render_template('index.html')


@app.route('/cee')
def cee_page():
    return render_template('cee.html')


@app.route('/ai_platform')
def ai_platform():
    return render_template('ai_platform.html')


@app.route('/download_materials')
def download_materials():
    account = os.getenv("AZURE_ACCOUNT_NAME")
    key     = os.getenv("AZURE_ACCOUNT_KEY")
    blob    = "curriculum_materials.zip"
    sas = generate_blob_sas(
        account_name=account,
        container_name="curriculum-material",
        blob_name=blob,
        account_key=key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.utcnow() + timedelta(hours=1)
    )
    url = f"https://{account}.blob.core.windows.net/curriculum-material/{blob}?{sas}"
    return redirect(url)


# ---------------------------
#  AUTH + ANALYTICS HOOKS
# ---------------------------
@app.route('/logout_and_home')
def logout_and_home():
    _finalize_and_persist_session()
    session.clear()
    return redirect(url_for('home'))


@app.route('/teams')
def teams():
    return render_template('teams.html')


@app.route('/curriculum_materials')
def curriculum_materials():
    return render_template('curriculum_materials.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    # Optional: also allow GET auto-login at /login if the link points here
    if request.method == 'GET' and _try_auto_login_from_query():
        return redirect(url_for('home'))

    if request.method == 'POST':
        email = (request.form.get('username') or "").strip().lower()
        password = request.form.get('password') or ""

        user_entity = get_user_by_email(email)
        if not user_entity:
            error = "Invalid email or password!"
            return render_template('login.html', error=error)

        stored_hash = user_entity.get("PasswordHash", "")
        if stored_hash and verify_password(password, stored_hash):
            _initialize_session_user(email)   # <<< initialize analytics + user folder
            return redirect(url_for('home'))
        else:
            error = "Invalid email or password!"
            return render_template('login.html', error=error)

    return render_template('login.html')


@app.route('/login_referral', methods=['GET', 'POST'])
def login_referral():
    # NEW: support GET links like /login_referral?userID=abc&referal=MTU
    if request.method == 'GET':
        if _try_auto_login_from_query():
            return redirect(url_for('home'))
        return render_template('login.html')

    # Existing POST flow unchanged
    ref_username = (request.form.get('ref_username', '')).strip().lower()
    ref_code = (request.form.get('ref_code', '')).strip()
    VALID_REFERRAL_CODE = os.getenv("REFERRAL_CODE", "MTU")

    if ref_code == VALID_REFERRAL_CODE:
        existing_user = get_user_by_email(ref_username)
        if not existing_user:
            create_user_with_referral(ref_username, ref_code)
        _initialize_session_user(ref_username)   # <<< initialize analytics + user folder
        return redirect(url_for('home'))
    else:
        ref_error = "Invalid referral code!"
        return render_template('login.html', ref_error=ref_error)


@app.route('/register_referral', methods=['GET', 'POST'])
def register_referral():
    if request.method == 'POST':
        ref_username = (request.form.get('ref_username', '')).strip().lower()
        ref_code = (request.form.get('ref_code', '')).strip()
        VALID_REFERRAL_CODE = os.getenv("REFERRAL_CODE", "MTU")

        if ref_code == VALID_REFERRAL_CODE:
            existing_user = get_user_by_email(ref_username)
            if existing_user:
                ref_error = "User with this username already exists!"
                return render_template('register.html', ref_error=ref_error)
            else:
                create_user_with_referral(ref_username, ref_code)
                _initialize_session_user(ref_username)  # <<< initialize analytics + user folder
                return redirect(url_for('home'))
        else:
            ref_error = "Invalid referral code!"
            return render_template('register.html', ref_error=ref_error)
    else:
        return render_template('register.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = (request.form.get('username') or "").strip().lower()
        password = request.form.get('password') or ""

        existing_user = get_user_by_email(email)
        if existing_user:
            error = "User with this email already exists!"
            return render_template('register.html', error=error)

        create_user(email, password)
        return redirect(url_for('login'))
    else:
        return render_template('register.html')


@app.route('/logout')
def logout():
    _finalize_and_persist_session()
    session.clear()
    return redirect(url_for('login'))


# ---------------------------
#   Chat-related endpoints (unchanged)
# ---------------------------
@app.route('/chat', methods=['POST'])
def chat():
    if not session.get('logged_in'):
        return jsonify({'bot_reply': "You must be logged in to chat."}), 401

    data = request.json or {}
    user_message = (data.get('user_message') or '').strip()

    # persist a per-thread counter for BIM replies
    thread_id = session.get('thread_id') or str(uuid.uuid4())
    session['thread_id'] = thread_id
    session.setdefault('bim_ai_count', 0)

    config = {"configurable": {"thread_id": thread_id}}
    human = HumanMessage(content=user_message)

    reply, suggestions = None, []
    for event in chat_app.stream({"messages": [human], "suggestions": []}, config, stream_mode="values"):
        reply = event.get("messages", [])[-1].content if event.get("messages") else ""
        suggestions = event.get("suggestions", suggestions)

    # increment counter after each bot reply
    session['bim_ai_count'] = int(session.get('bim_ai_count', 0)) + 1
    ai_count = session['bim_ai_count']

    # capture chat messages
    _append_chat_message(role="user", model="bim", text=user_message)
    _append_chat_message(role="assistant", model="bim", text=reply or "")

    # checkpoint (best effort)
    _checkpoint_chat_to_blob()

    # only surface suggestions on every 5th reply
    if ai_count % 5 != 0:
        suggestions = []

    payload = {'bot_reply': reply or ""}
    if suggestions:
        payload['suggested_questions'] = suggestions
    return jsonify(payload)


@app.route('/cee_chat', methods=['POST'])
def cee_chat():
    if not session.get('logged_in'):
        return jsonify({'bot_reply': "You must be logged in to chat."}), 401

    data = request.json or {}
    user_message = (data.get('user_message') or '').strip()

    # persist a separate counter for CEE replies
    cee_thread_id = session.get('cee_thread_id') or str(uuid.uuid4())
    session['cee_thread_id'] = cee_thread_id
    session.setdefault('cee_ai_count', 0)

    config = {"configurable": {"thread_id": cee_thread_id}}
    human = HumanMessage(content=user_message)

    reply, suggestions = None, []
    for event in cee_chat_app.stream({"messages": [human], "suggestions": []}, config, stream_mode="values"):
        reply = event.get("messages", [])[-1].content if event.get("messages") else ""
        suggestions = event.get("suggestions", suggestions)

    # increment counter
    session['cee_ai_count'] = int(session.get('cee_ai_count', 0)) + 1
    ai_count = session['cee_ai_count']

    # capture chat messages
    _append_chat_message(role="user", model="cee", text=user_message)
    _append_chat_message(role="assistant", model="cee", text=reply or "")

    # checkpoint (best effort)
    _checkpoint_chat_to_blob()

    if ai_count % 5 != 0:
        suggestions = []

    payload = {'bot_reply': reply or ""}
    if suggestions:
        payload['suggested_questions'] = suggestions
    return jsonify(payload)

# ---------------------------
#   Analytics helpers (session lifecycle)
# ---------------------------
def _initialize_session_user(username: str):
    """Initialize analytics session on successful login/referral."""
    session['logged_in'] = True
    session['user_email'] = username  # reuse key for referral login too
    login_time = datetime.now(timezone.utc)
    session['login_time_utc'] = login_time.isoformat().replace("+00:00", "Z")
    # Make folder name include timestamp
    timestamp_str = login_time.strftime("%Y-%m-%d_%H-%M-%S")
    session_uuid = str(uuid.uuid4())
    session['session_id'] = f"{timestamp_str}_{session_uuid}"
    session['chat_log'] = []  # list of {ts_utc, role, model, text}

    # Ensure user folder and write initial checkpoint
    try:
        svc = _blob_service()
        container = _ensure_container(svc, USER_LOGS_CONTAINER)
        _ensure_user_prefix(container, username)

        checkpoint = {
            "username": username,
            "session_id": session['session_id'],
            "login_time_utc": session['login_time_utc'],
            "messages": [],
        }
        _upload_json(container, f"{username}/{session['session_id']}/checkpoint.json", checkpoint)
        _upload_json(container, f"{username}/latest.json", checkpoint)
    except Exception as e:
        try:
            flash(f"Warning: could not initialize user folder in Blob Storage ({e}).")
        except Exception:
            pass

def _append_chat_message(role: str, model: str, text: str):
    if not text:
        return
    entry = {"ts_utc": _utc_now_iso(), "role": role, "model": model, "text": text}
    session.setdefault('chat_log', []).append(entry)

def _checkpoint_chat_to_blob():
    """Best-effort incremental upload of the chat log to Blob."""
    try:
        username = session.get('user_email') or session.get('username') or "anonymous"
        svc = _blob_service()
        container = _ensure_container(svc, USER_LOGS_CONTAINER)
        payload = {
            "username": username,
            "session_id": session.get('session_id'),
            "login_time_utc": session.get('login_time_utc'),
            "messages": session.get('chat_log', []),
        }
        _upload_json(container, f"{username}/{session.get('session_id')}/checkpoint.json", payload)
        _upload_json(container, f"{username}/latest.json", payload)
    except Exception as e:
        app.logger.warning(f"Checkpoint upload failed: {e}")

def _finalize_and_persist_session():
    """On logout, compute duration and persist final session.json to Blob."""
    try:
        username = session.get('user_email') or session.get('username')
        if not username:
            return
        session_id = session.get('session_id') or str(uuid.uuid4())
        login_time_iso = session.get('login_time_utc')
        logout_time_iso = _utc_now_iso()

        def _parse(ts: str):
            return datetime.fromisoformat(ts.replace('Z', '+00:00'))

        try:
            start_dt = _parse(login_time_iso) if login_time_iso else None
            end_dt = _parse(logout_time_iso)
            duration_seconds = int((end_dt - start_dt).total_seconds()) if start_dt else None
        except Exception:
            duration_seconds = None

        messages = session.get('chat_log', [])
        final = {
            "username": username,
            "session_id": session_id,
            "login_time_utc": login_time_iso,
            "logout_time_utc": logout_time_iso,
            "duration_seconds": duration_seconds,
            "message_count": len(messages),
            "messages": messages,
        }

        svc = _blob_service()
        container = _ensure_container(svc, USER_LOGS_CONTAINER)
        base = f"{username}/{session_id}/"
        _upload_json(container, base + "session.json", final)
        _upload_json(container, f"{username}/latest.json", final)
    except Exception as e:
        try:
            flash(f"Warning: could not persist session to Blob Storage ({e}).")
        except Exception:
            app.logger.warning(f"Persist session failed: {e}")

# ---------------------------
#   Entry
# ---------------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port, debug=True)