import json
import os
import secrets
import shutil
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from flask import (
    Flask,
    Response,
    has_request_context,
    jsonify,
    redirect,
    render_template_string,
    request,
    send_from_directory,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash
## (doing this to make a commit)
APP_TITLE = "OCreator"
DATA_DIR = Path(__file__).resolve().parent / "data"
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
DB_PATH = DATA_DIR / "ocs.json"
SETTINGS_PATH = DATA_DIR / "settings.json"
PROVIDERS_PATH = DATA_DIR / "providers.json"
SECURITY_DIR = DATA_DIR / "security"
SECURITY_CONFIG_PATH = SECURITY_DIR / "config.json"
USERS_PATH = SECURITY_DIR / "users.json"
USER_DATA_DIR = DATA_DIR / "users"

PROVIDERS = [
    {"id": "openai", "label": "OpenAI"},
    {"id": "gemini", "label": "Gemini"},
    {"id": "anthropic", "label": "Anthropic"},
    {"id": "grok", "label": "Grok"},
    {"id": "openrouter", "label": "OpenRouter"},
    {"id": "openai_compatible", "label": "OpenAI Compatible"},
]

PROVIDER_DEFAULTS = {
    "openai": {"model": "gpt-5-nano", "base_url": "https://api.openai.com/v1"},
    "gemini": {"model": "gemini-2.5-flash", "base_url": "https://generativelanguage.googleapis.com/v1beta"},
    "anthropic": {"model": "claude-3-5-haiku-latest", "base_url": "https://api.anthropic.com/v1"},
    "grok": {"model": "grok-3-mini", "base_url": "https://api.x.ai/v1"},
    "openrouter": {"model": "openai/gpt-5-nano", "base_url": "https://openrouter.ai/api/v1"},
    "openai_compatible": {"model": "your-model", "base_url": "http://localhost:1234/v1"},
}

DEFAULT_SETTINGS = {
    "provider": "openai",
    "api_key": "",
    "model": PROVIDER_DEFAULTS["openai"]["model"],
    "base_url": PROVIDER_DEFAULTS["openai"]["base_url"],
    "temperature": 0.7,
    "max_tokens": 1200,
    "pre_cut_enabled": False,
    "pre_cut_markers": "",
    "pre_cut_replace_enabled": False,
    "pre_cut_replace_rules": "",
}

MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0
MAX_PRE_CUT_TEXT_LEN = 12000
MAX_PRE_CUT_RULE_LINES = 200

DEFAULT_SECURITY_CONFIG = {"enabled": False}
DEFAULT_USERS_DB = {"users": []}

DEFAULT_TEMPLATE = "\n".join(
    [
        "Name:",
        "Age:",
        "Species:",
        "Appearance:",
        "Origin:",
        "Occupation:",
        "Skills:",
        "Abilities:",
        "Gear:",
        "Relationships:",
        "Goals:",
        "Limits/Boundaries:",
        "Notes:",
    ]
)

DEFAULT_OC = {
    "id": "",
    "name": "Untitled OC",
    "tab_name": "Untitled OC",
    "mode": "scratch",
    "prompt": "",
    "existing_text": "",
    "enhance_notes": "",
    "template_mode": "default",
    "custom_template": "",
    "result_text": "",
    "updated_at": "",
    "pinned": False,
}

DEFAULT_PROVIDER_PROFILE = {
    "id": "",
    "name": "",
    "provider": "openai",
    "model": "",
    "base_url": "",
    "api_key": "",
    "notes": "",
    "updated_at": "",
}

app = Flask(__name__)
app.secret_key = os.environ.get("OCREATOR_SECRET_KEY", "") or secrets.token_hex(32)
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    PERMANENT_SESSION_LIFETIME=timedelta(days=14),
)


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SECURITY_DIR.mkdir(parents=True, exist_ok=True)
    USER_DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}
    return False


def username_key(username: str) -> str:
    return str(username or "").strip().lower()


def new_user_id() -> str:
    return uuid.uuid4().hex[:12]


def user_folder_from_id(user_id: str) -> str:
    cleaned = "".join(ch for ch in str(user_id or "").strip().lower() if ch.isalnum())
    return cleaned or new_user_id()


def legacy_folder_from_username(username: str) -> str:
    base = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in username.strip().lower())
    return base.strip("_")


def ensure_unique_user_id(users: List[Dict[str, Any]], preferred: str = "") -> str:
    existing = {str(item.get("id", "") or "").strip().lower() for item in users if isinstance(item, dict)}
    if preferred:
        candidate = "".join(ch for ch in str(preferred).strip().lower() if ch.isalnum())
        if candidate and candidate not in existing:
            return candidate
    candidate = new_user_id()
    while candidate in existing:
        candidate = new_user_id()
    return candidate


def migrate_user_folder(old_folder: str, new_folder: str) -> None:
    old_name = str(old_folder or "").strip()
    new_name = str(new_folder or "").strip()
    if not old_name or not new_name or old_name == new_name:
        return
    old_path = USER_DATA_DIR / old_name
    new_path = USER_DATA_DIR / new_name
    if not old_path.exists() or not old_path.is_dir():
        return
    new_path.mkdir(parents=True, exist_ok=True)
    for filename in ("ocs.json", "settings.json", "providers.json"):
        src = old_path / filename
        dst = new_path / filename
        if src.exists() and not dst.exists():
            try:
                shutil.move(str(src), str(dst))
            except Exception:
                pass
    try:
        old_path.rmdir()
    except Exception:
        pass


def load_security_config() -> Dict[str, Any]:
    raw = load_json(SECURITY_CONFIG_PATH, DEFAULT_SECURITY_CONFIG.copy())
    config = DEFAULT_SECURITY_CONFIG.copy()
    if isinstance(raw, dict):
        config.update(raw)
    config["enabled"] = parse_bool(config.get("enabled"))
    return config


def save_security_config(config: Dict[str, Any]) -> None:
    payload = {"enabled": parse_bool(config.get("enabled", False)), "updated_at": now_iso()}
    save_json(SECURITY_CONFIG_PATH, payload)


def load_users_db() -> Dict[str, Any]:
    raw = load_json(USERS_PATH, DEFAULT_USERS_DB.copy())
    users = raw.get("users", []) if isinstance(raw, dict) else []
    if not isinstance(users, list):
        users = []
    ensure_dirs()
    normalized: List[Dict[str, Any]] = []
    changed = False
    for item in users:
        if not isinstance(item, dict):
            changed = True
            continue
        username = str(item.get("username", "") or "").strip()
        password_hash = str(item.get("password_hash", "") or "").strip()
        if not username or not password_hash:
            changed = True
            continue
        user_id = ensure_unique_user_id(normalized, str(item.get("id", "") or ""))
        folder = user_folder_from_id(user_id)
        old_folder = str(item.get("folder", "") or "").strip()
        if not old_folder:
            old_folder = legacy_folder_from_username(username)
        if old_folder and old_folder != folder:
            migrate_user_folder(old_folder, folder)
            changed = True
        if str(item.get("id", "") or "").strip().lower() != user_id:
            changed = True
        if old_folder != folder:
            changed = True
        normalized.append(
            {
                "id": user_id,
                "username": username,
                "password_hash": password_hash,
                "folder": folder,
                "created_at": str(item.get("created_at", "") or ""),
                "updated_at": str(item.get("updated_at", "") or ""),
            }
        )
    if changed:
        save_users_db({"users": normalized})
    return {"users": normalized}


def save_users_db(users_db: Dict[str, Any]) -> None:
    users = users_db.get("users", []) if isinstance(users_db, dict) else []
    if not isinstance(users, list):
        users = []
    save_json(USERS_PATH, {"users": users})


def find_user(username: str) -> Optional[Dict[str, Any]]:
    key = username_key(username)
    if not key:
        return None
    users = load_users_db().get("users", [])
    for user in users:
        if username_key(str(user.get("username", ""))) == key:
            return user
    return None


def get_session_user() -> Optional[Dict[str, Any]]:
    if not has_request_context():
        return None
    username = str(session.get("username", "") or "").strip()
    if not username:
        return None
    return find_user(username)


def security_enabled() -> bool:
    return bool(load_security_config().get("enabled", False))


def user_data_dir(user: Dict[str, Any]) -> Path:
    user_id = str(user.get("id", "") or "").strip()
    folder = str(user.get("folder", "") or "").strip() or user_folder_from_id(user_id or new_user_id())
    path = USER_DATA_DIR / folder
    path.mkdir(parents=True, exist_ok=True)
    return path


def current_data_paths() -> Dict[str, Path]:
    if security_enabled() and has_request_context():
        user = get_session_user()
        if user:
            root = user_data_dir(user)
            return {
                "db": root / "ocs.json",
                "settings": root / "settings.json",
                "providers": root / "providers.json",
            }
    return {"db": DB_PATH, "settings": SETTINGS_PATH, "providers": PROVIDERS_PATH}


def copy_legacy_data_into_user(user: Dict[str, Any]) -> None:
    target_root = user_data_dir(user)
    for legacy_path, target_name in ((DB_PATH, "ocs.json"), (SETTINGS_PATH, "settings.json"), (PROVIDERS_PATH, "providers.json")):
        target = target_root / target_name
        if legacy_path.exists() and not target.exists():
            shutil.copy2(legacy_path, target)


def validate_username(username: str) -> Optional[str]:
    value = str(username or "").strip()
    if len(value) < 3:
        return "Username must be at least 3 characters."
    if len(value) > 64:
        return "Username is too long."
    if any(ch.isspace() for ch in value):
        return "Username cannot include spaces."
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    if any(ch not in allowed for ch in value):
        return "Username can only use letters, numbers, - and _."
    return None


def validate_password(password: str) -> Optional[str]:
    value = str(password or "")
    if len(value) < 8:
        return "Password must be at least 8 characters."
    return None


def save_user_credentials(username: str, password: str, copy_legacy_if_first: bool = True) -> Dict[str, Any]:
    users_db = load_users_db()
    users = users_db.get("users", [])
    key = username_key(username)
    now = now_iso()
    for user in users:
        if username_key(str(user.get("username", ""))) == key:
            user["username"] = username.strip()
            user["password_hash"] = generate_password_hash(password)
            user["updated_at"] = now
            if not str(user.get("id", "") or "").strip():
                user["id"] = ensure_unique_user_id(users)
            user["folder"] = user_folder_from_id(str(user.get("id", "") or ""))
            save_users_db(users_db)
            user_data_dir(user)
            return user

    is_first_user = len(users) == 0
    user_id = ensure_unique_user_id(users)
    folder = user_folder_from_id(user_id)
    user = {
        "id": user_id,
        "username": username.strip(),
        "password_hash": generate_password_hash(password),
        "folder": folder,
        "created_at": now,
        "updated_at": now,
    }
    users.append(user)
    save_users_db(users_db)
    user_data_dir(user)
    if copy_legacy_if_first and is_first_user:
        copy_legacy_data_into_user(user)
    return user


def verify_user_credentials(username: str, password: str) -> bool:
    user = find_user(username)
    if not user:
        return False
    password_hash = str(user.get("password_hash", "") or "")
    if not password_hash:
        return False
    return check_password_hash(password_hash, password)


def parse_pre_cut_replace_rules(raw: str) -> List[tuple[str, str]]:
    rules: List[tuple[str, str]] = []
    if not raw:
        return rules
    lines = str(raw).replace("\r\n", "\n").split("\n")
    for line in lines[:MAX_PRE_CUT_RULE_LINES]:
        text = line.strip()
        if not text:
            continue
        if "=>" in text:
            source, target = text.split("=>", 1)
        elif "->" in text:
            source, target = text.split("->", 1)
        else:
            continue
        source = source.strip()
        if not source:
            continue
        rules.append((source, target.strip()))
    return rules


def apply_pre_cut(text: str, settings: Dict[str, Any]) -> tuple[str, bool]:
    content = sanitize_text(text)
    changed = False

    if parse_bool(settings.get("pre_cut_enabled", False)):
        markers_raw = str(settings.get("pre_cut_markers", "") or "")
        markers = [line.strip() for line in markers_raw.replace("\r\n", "\n").split("\n") if line.strip()]
        if markers:
            lower_content = content.lower()
            first_index: Optional[int] = None
            for marker in markers:
                idx = lower_content.find(marker.lower())
                if idx >= 0 and (first_index is None or idx < first_index):
                    first_index = idx
            if first_index is not None:
                content = content[:first_index].rstrip()
                changed = True

    if parse_bool(settings.get("pre_cut_replace_enabled", False)):
        rules = parse_pre_cut_replace_rules(str(settings.get("pre_cut_replace_rules", "") or ""))
        if rules:
            for source, target in rules:
                if source in content:
                    content = content.replace(source, target)
                    changed = True

    return content, changed


def is_api_request() -> bool:
    path = request.path
    if (
        path.startswith("/oc/")
        or path.startswith("/settings")
        or path.startswith("/providers")
        or path.startswith("/security/")
    ):
        return True
    accept = (request.headers.get("Accept") or "").lower()
    return "application/json" in accept


@app.before_request
def require_auth_if_enabled() -> Optional[Response]:
    ensure_dirs()
    if request.path.startswith("/assets/"):
        return None
    if request.endpoint in {"assets_route", "secure_login_route", "login_route"}:
        return None
    if not security_enabled():
        return None
    user = get_session_user()
    if user:
        session["username"] = user["username"]
        session.permanent = True
        return None
    session.pop("username", None)
    if is_api_request() or request.method == "POST":
        return jsonify({"error": "Authentication required"}), 401
    next_path = request.path
    if request.query_string:
        next_path = f"{request.path}?{request.query_string.decode('utf-8', errors='ignore')}"
    return redirect(url_for("secure_login_route", next=next_path))


def clamp_settings_values(settings: Dict[str, Any]) -> Dict[str, Any]:
    provider = str(settings.get("provider", "openai") or "openai").strip()
    if provider not in {p["id"] for p in PROVIDERS}:
        provider = "openai"
    defaults = PROVIDER_DEFAULTS.get(provider, PROVIDER_DEFAULTS["openai"])

    try:
        temperature = float(settings.get("temperature", DEFAULT_SETTINGS["temperature"]))
    except Exception:
        temperature = DEFAULT_SETTINGS["temperature"]
    temperature = max(MIN_TEMPERATURE, min(MAX_TEMPERATURE, temperature))

    try:
        max_tokens = int(settings.get("max_tokens", DEFAULT_SETTINGS["max_tokens"]))
    except Exception:
        max_tokens = DEFAULT_SETTINGS["max_tokens"]
    pre_cut_enabled = parse_bool(settings.get("pre_cut_enabled", DEFAULT_SETTINGS["pre_cut_enabled"]))
    pre_cut_markers = str(settings.get("pre_cut_markers", DEFAULT_SETTINGS["pre_cut_markers"]) or "")
    pre_cut_replace_enabled = parse_bool(
        settings.get("pre_cut_replace_enabled", DEFAULT_SETTINGS["pre_cut_replace_enabled"])
    )
    pre_cut_replace_rules = str(settings.get("pre_cut_replace_rules", DEFAULT_SETTINGS["pre_cut_replace_rules"]) or "")
    if len(pre_cut_markers) > MAX_PRE_CUT_TEXT_LEN:
        pre_cut_markers = pre_cut_markers[:MAX_PRE_CUT_TEXT_LEN]
    if len(pre_cut_replace_rules) > MAX_PRE_CUT_TEXT_LEN:
        pre_cut_replace_rules = pre_cut_replace_rules[:MAX_PRE_CUT_TEXT_LEN]

    return {
        "provider": provider,
        "api_key": str(settings.get("api_key", "") or "").strip(),
        "model": str(settings.get("model", "") or "").strip() or defaults["model"],
        "base_url": str(settings.get("base_url", "") or "").strip() or defaults["base_url"],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "pre_cut_enabled": pre_cut_enabled,
        "pre_cut_markers": pre_cut_markers,
        "pre_cut_replace_enabled": pre_cut_replace_enabled,
        "pre_cut_replace_rules": pre_cut_replace_rules,
    }


def load_settings() -> Dict[str, Any]:
    settings_path = current_data_paths()["settings"]
    settings = load_json(settings_path, DEFAULT_SETTINGS.copy())
    merged = DEFAULT_SETTINGS.copy()
    if isinstance(settings, dict):
        merged.update({k: v for k, v in settings.items() if v is not None})
    return clamp_settings_values(merged)


def save_settings(settings: Dict[str, Any]) -> None:
    settings_path = current_data_paths()["settings"]
    save_json(settings_path, settings)


def load_provider_profiles() -> List[Dict[str, Any]]:
    providers_path = current_data_paths()["providers"]
    raw = load_json(providers_path, {"providers": []})
    items = raw.get("providers", []) if isinstance(raw, dict) else []
    if not isinstance(items, list):
        return []
    profiles: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        profile = DEFAULT_PROVIDER_PROFILE.copy()
        profile.update(item)
        if not profile.get("id"):
            profile["id"] = uuid.uuid4().hex[:10]
        profiles.append(profile)
    return profiles


def save_provider_profiles(profiles: List[Dict[str, Any]]) -> None:
    providers_path = current_data_paths()["providers"]
    save_json(providers_path, {"providers": profiles})


def load_db() -> Dict[str, Any]:
    db_path = current_data_paths()["db"]
    db = load_json(db_path, {"ocs": []})
    if "ocs" not in db or not isinstance(db["ocs"], list):
        db["ocs"] = []
    return db


def save_db(db: Dict[str, Any]) -> None:
    db_path = current_data_paths()["db"]
    save_json(db_path, db)


def ensure_oc_defaults(oc: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in DEFAULT_OC.items():
        oc.setdefault(key, value)
    if not str(oc.get("tab_name", "")).strip():
        oc["tab_name"] = oc.get("name", "Untitled OC")
    oc["pinned"] = bool(oc.get("pinned", False))
    return oc


def sort_ocs(ocs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = [ensure_oc_defaults(oc) for oc in ocs]
    pinned = [oc for oc in normalized if oc.get("pinned")]
    unpinned = [oc for oc in normalized if not oc.get("pinned")]
    pinned.sort(key=lambda oc: (str(oc.get("name") or "Untitled OC").strip().lower(), oc.get("id", "")))
    return pinned + unpinned


def get_oc(db: Dict[str, Any], oc_id: str) -> Optional[Dict[str, Any]]:
    for oc in db.get("ocs", []):
        if oc.get("id") == oc_id:
            return ensure_oc_defaults(oc)
    return None


def create_oc(db: Dict[str, Any]) -> Dict[str, Any]:
    oc = DEFAULT_OC.copy()
    oc["id"] = uuid.uuid4().hex[:10]
    oc["updated_at"] = now_iso()
    db["ocs"].append(oc)
    return oc


def sanitize_text(text: str) -> str:
    cleaned = (text or "").strip()
    return cleaned.replace("\u2014", "-")


def build_prompt(oc: Dict[str, Any], notes: str) -> str:
    mode = oc.get("mode", "scratch")
    template_mode = oc.get("template_mode", "default")
    prompt = sanitize_text(oc.get("prompt", ""))
    existing = sanitize_text(oc.get("existing_text", ""))
    enhance = sanitize_text(oc.get("enhance_notes", ""))
    notes = sanitize_text(notes or "")

    template = ""
    if template_mode == "default":
        template = DEFAULT_TEMPLATE
    elif template_mode == "custom":
        template = sanitize_text(oc.get("custom_template", ""))
    else:
        template = ""

    template_block = ""
    if template:
        template_block = f"Use this template verbatim with the same labels and order:\n{template}\n\n"
    else:
        template_block = (
            "No template is required. Produce a single clean OC persona block with short paragraphs.\n\n"
        )

    base = (
        "You are creating a playable OC persona for a human to roleplay as.\n"
        "Do not define personality, temperament, or how the player should act.\n"
        "Focus on concrete identity details, background, skills, gear, and facts.\n"
        "Do not use em dashes.\n"
        "Output plain text only. No JSON or markdown.\n\n"
    )

    if mode == "existing":
        context = (
            "Rewrite or clean up the existing OC text while keeping the same identity.\n"
            "Improve clarity, structure, and completeness.\n\n"
            f"OC prompt:\n{prompt or 'None'}\n\n"
            f"Existing OC text:\n{existing}\n\n"
            f"Requested improvements:\n{enhance or 'None'}\n\n"
        )
    elif mode == "enhance":
        context = (
            "Enhance the existing OC text with the changes requested.\n"
            "Keep the identity consistent while improving gaps or adding detail.\n\n"
            f"OC prompt:\n{prompt or 'None'}\n\n"
            f"Existing OC text:\n{existing}\n\n"
            f"Requested improvements:\n{enhance or 'None'}\n\n"
        )
    else:
        context = f"OC prompt:\n{prompt or 'None'}\n\n"

    if notes:
        context += f"Additional notes:\n{notes}\n\n"

    return base + template_block + context + "Return the final OC text now."


def build_quick_fix_prompt(source_text: str) -> str:
    cleaned = sanitize_text(source_text)
    return (
        "You are fixing OC text for clarity and readability.\n"
        "Only fix grammar, spelling, punctuation, formatting, and structure.\n"
        "Do not add new lore, powers, backstory, or facts.\n"
        "Keep meaning and details the same.\n"
        "Output plain text only.\n\n"
        f"OC text:\n{cleaned}\n\n"
        "Return the corrected OC text now."
    )


def request_json(url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(url, headers=headers, json=payload, timeout=90)
    if resp.status_code >= 400:
        raise RuntimeError(f"{resp.status_code} {resp.text}")
    return resp.json()


def call_openai_like(settings: Dict[str, Any], prompt: str) -> str:
    base_url = settings.get("base_url", "").rstrip("/")
    model = settings.get("model", "")
    url = base_url + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.get('api_key', '')}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": settings.get("temperature", 0.7),
        "max_tokens": settings.get("max_tokens", 1200),
    }
    data = request_json(url, headers, payload)
    return (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""


def call_anthropic(settings: Dict[str, Any], prompt: str) -> str:
    base_url = settings.get("base_url", "").rstrip("/")
    url = base_url + "/messages"
    headers = {
        "x-api-key": settings.get("api_key", ""),
        # Keep current with Anthropic docs if this API version is deprecated.
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.get("model", ""),
        "max_tokens": settings.get("max_tokens", 1200),
        "temperature": settings.get("temperature", 0.7),
        "messages": [{"role": "user", "content": prompt}],
    }
    data = request_json(url, headers, payload)
    content = data.get("content") or []
    if content and isinstance(content, list):
        return content[0].get("text", "") or ""
    return ""


def call_gemini(settings: Dict[str, Any], prompt: str) -> str:
    base_url = settings.get("base_url", "").rstrip("/")
    model = settings.get("model", "")
    url = base_url + f"/models/{model}:generateContent"
    params = {"key": settings.get("api_key", "")}
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    resp = requests.post(url, headers=headers, params=params, json=payload, timeout=90)
    if resp.status_code >= 400:
        raise RuntimeError(f"{resp.status_code} {resp.text}")
    data = resp.json()
    candidates = data.get("candidates") or []
    if not candidates:
        block_reason = data.get("promptFeedback", {}).get("blockReason")
        if block_reason:
            raise RuntimeError(f"Gemini blocked the prompt: {block_reason}")
        return ""

    texts: List[str] = []
    finish_reasons: List[str] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        finish_reason = str(candidate.get("finishReason", "")).strip()
        if finish_reason:
            finish_reasons.append(finish_reason)
        parts = candidate.get("content", {}).get("parts", []) or []
        for part in parts:
            if isinstance(part, dict) and part.get("text"):
                texts.append(str(part.get("text")))

    combined = "".join(texts).strip()
    if combined:
        return combined

    if finish_reasons:
        raise RuntimeError(f"Gemini returned no text. finishReason={','.join(finish_reasons)}")
    raise RuntimeError("Gemini returned no text.")


def call_llm(settings: Dict[str, Any], prompt: str) -> str:
    provider = settings.get("provider", "openai")
    if provider in {"openai", "openrouter", "grok", "openai_compatible"}:
        return call_openai_like(settings, prompt)
    if provider == "anthropic":
        return call_anthropic(settings, prompt)
    if provider == "gemini":
        return call_gemini(settings, prompt)
    raise RuntimeError("Unknown provider.")


@app.route("/secure-login", methods=["GET", "POST"])
def secure_login_route():
    ensure_dirs()
    next_path = str(request.args.get("next", "/") or "/")
    if not next_path.startswith("/"):
        next_path = "/"
    users = load_users_db().get("users", [])
    has_users = bool(users)
    mode = "signup" if not has_users else "login"
    current = get_session_user()
    if current and security_enabled():
        return redirect(next_path)

    error = ""
    prefill_username = ""
    if request.method == "POST":
        action = str(request.form.get("action", "login") or "login").strip().lower()
        if action not in {"login", "signup"}:
            action = "login"
        username = str(request.form.get("username", "") or "").strip()
        password = str(request.form.get("password", "") or "")
        confirm = str(request.form.get("confirm", "") or "")
        posted_next = str(request.form.get("next", "/") or "/")
        if posted_next.startswith("/"):
            next_path = posted_next
        prefill_username = username
        users = load_users_db().get("users", [])
        has_users = bool(users)
        if not has_users:
            action = "signup"

        if action == "signup":
            username_error = validate_username(username)
            password_error = validate_password(password)
            if username_error:
                error = username_error
            elif password_error:
                error = password_error
            elif password != confirm:
                error = "Passwords do not match."
            elif find_user(username):
                error = "Could not create account with that username."
            else:
                user = save_user_credentials(username, password, copy_legacy_if_first=not has_users)
                session["username"] = user["username"]
                session.permanent = True
                if not security_enabled():
                    save_security_config({"enabled": True})
                return redirect(next_path)
        else:
            if not verify_user_credentials(username, password):
                error = "Invalid username or password."
            else:
                user = find_user(username)
                if not user:
                    error = "Account no longer exists."
                else:
                    session["username"] = user["username"]
                    session.permanent = True
                    return redirect(next_path)

    return render_template_string(
        SECURE_LOGIN_TEMPLATE,
        app_title=APP_TITLE,
        error=error,
        mode=mode,
        has_users=has_users,
        next_path=next_path,
        username=prefill_username,
    )


@app.route("/login", methods=["GET", "POST"])
def login_route():
    return redirect(url_for("secure_login_route", next=request.args.get("next", "/")))


@app.route("/logout", methods=["POST"])
def logout_route():
    session.pop("username", None)
    if security_enabled():
        return redirect(url_for("secure_login_route"))
    return redirect(url_for("index"))


@app.route("/security/status")
def security_status_route():
    ensure_dirs()
    current = get_session_user()
    return jsonify(
        {
            "ok": True,
            "enabled": security_enabled(),
            "current_user": str(current.get("username", "") if current else ""),
        }
    )


@app.route("/security/config", methods=["POST"])
def security_config_route():
    ensure_dirs()
    payload = request.get_json(silent=True) or {}
    enabled = parse_bool(payload.get("enabled", False))
    users = load_users_db().get("users", [])
    if enabled and not users:
        return jsonify({"error": "Create at least one account from Secure Login first."}), 400
    save_security_config({"enabled": enabled})
    if not enabled:
        session.pop("username", None)
    return jsonify({"ok": True, "enabled": enabled})


@app.route("/security/users/save", methods=["POST"])
def security_save_user_route():
    ensure_dirs()
    current = get_session_user()
    if not current:
        return jsonify({"error": "Sign in first."}), 401
    payload = request.get_json(silent=True) or {}
    password = str(payload.get("password", "") or "")
    password_error = validate_password(password)
    if password_error:
        return jsonify({"error": password_error}), 400
    user = save_user_credentials(str(current.get("username", "") or ""), password, copy_legacy_if_first=False)
    return jsonify({"ok": True, "current_user": str(user.get("username", "") or "")})


@app.route("/")
def index():
    ensure_dirs()
    db = load_db()
    if not db["ocs"]:
        oc = create_oc(db)
        save_db(db)
        return redirect(url_for("edit_oc", oc_id=oc["id"]))
    sorted_ocs = sort_ocs(db["ocs"])
    return redirect(url_for("edit_oc", oc_id=sorted_ocs[0]["id"]))


@app.route("/oc/<oc_id>")
def edit_oc(oc_id: str):
    ensure_dirs()
    db = load_db()
    oc = get_oc(db, oc_id)
    if not oc:
        oc = create_oc(db)
        save_db(db)
    settings = load_settings()
    provider_profiles = load_provider_profiles()
    sorted_ocs = sort_ocs(db["ocs"])
    current_user = get_session_user()
    return render_template_string(
        TEMPLATE,
        app_title=APP_TITLE,
        oc=oc,
        ocs=sorted_ocs,
        settings=settings,
        providers=PROVIDERS,
        provider_profiles=provider_profiles,
        provider_defaults=PROVIDER_DEFAULTS,
        default_template=DEFAULT_TEMPLATE,
        security_enabled=security_enabled(),
        current_username=str(current_user.get("username", "") if current_user else ""),
    )


@app.route("/bot/<oc_id>")
def legacy_bot_route(oc_id: str):
    return redirect(url_for("edit_oc", oc_id=oc_id))


@app.route("/assets/<path:filename>")
def assets_route(filename: str):
    return send_from_directory(ASSETS_DIR, filename)


@app.route("/oc/new", methods=["POST"])
def create_oc_route():
    ensure_dirs()
    db = load_db()
    oc = create_oc(db)
    save_db(db)
    return redirect(url_for("edit_oc", oc_id=oc["id"]))


@app.route("/oc/<oc_id>/duplicate", methods=["POST"])
def duplicate_oc_route(oc_id: str):
    ensure_dirs()
    db = load_db()
    source = get_oc(db, oc_id)
    if not source:
        return jsonify({"error": "OC not found"}), 404
    duplicate = DEFAULT_OC.copy()
    duplicate.update({k: v for k, v in source.items() if k in DEFAULT_OC})
    duplicate["id"] = uuid.uuid4().hex[:10]
    duplicate["name"] = f"{source.get('name') or 'Untitled OC'} (Copy)"
    duplicate["tab_name"] = f"{source.get('tab_name') or source.get('name') or 'Untitled OC'} (Copy)"
    duplicate["pinned"] = False
    duplicate["updated_at"] = now_iso()
    db["ocs"].append(duplicate)
    save_db(db)
    return jsonify({"ok": True, "oc_id": duplicate["id"]})


@app.route("/oc/<oc_id>/pin", methods=["POST"])
def pin_oc_route(oc_id: str):
    ensure_dirs()
    db = load_db()
    oc = get_oc(db, oc_id)
    if not oc:
        return jsonify({"error": "OC not found"}), 404
    payload = request.get_json(silent=True) or {}
    if "pinned" in payload:
        oc["pinned"] = bool(payload.get("pinned"))
    else:
        oc["pinned"] = not bool(oc.get("pinned", False))
    oc["updated_at"] = now_iso()
    save_db(db)
    return jsonify({"ok": True, "pinned": oc["pinned"]})


@app.route("/oc/<oc_id>/delete", methods=["POST"])
def delete_oc_route(oc_id: str):
    ensure_dirs()
    db = load_db()
    before = len(db.get("ocs", []))
    db["ocs"] = [oc for oc in db.get("ocs", []) if oc.get("id") != oc_id]
    if len(db["ocs"]) == before:
        return jsonify({"error": "OC not found"}), 404
    if not db["ocs"]:
        create_oc(db)
    sorted_ocs = sort_ocs(db["ocs"])
    next_id = sorted_ocs[0]["id"]
    save_db(db)
    return jsonify({"ok": True, "next_oc_id": next_id})


@app.route("/oc/<oc_id>/save", methods=["POST"])
def save_oc(oc_id: str):
    ensure_dirs()
    db = load_db()
    oc = get_oc(db, oc_id)
    if not oc:
        return jsonify({"error": "OC not found"}), 404
    payload = request.get_json(silent=True) or {}
    for key in (
        "name",
        "tab_name",
        "mode",
        "prompt",
        "existing_text",
        "enhance_notes",
        "template_mode",
        "custom_template",
        "result_text",
    ):
        if key in payload:
            oc[key] = payload[key]
    oc["updated_at"] = now_iso()
    save_db(db)
    return jsonify({"ok": True})


@app.route("/oc/<oc_id>/generate", methods=["POST"])
def generate_oc(oc_id: str):
    ensure_dirs()
    db = load_db()
    oc = get_oc(db, oc_id)
    if not oc:
        return jsonify({"error": "OC not found"}), 404
    payload = request.get_json(silent=True) or {}
    notes = payload.get("notes", "")
    settings = load_settings()
    prompt = build_prompt(oc, notes)
    try:
        response = call_llm(settings, prompt)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
    response = sanitize_text(response)
    response, filtered = apply_pre_cut(response, settings)
    oc["result_text"] = response
    if not oc.get("name") or oc["name"] == "Untitled OC":
        first_line = response.splitlines()[0].strip()
        if first_line.lower().startswith("name:"):
            candidate = first_line.split(":", 1)[-1].strip()
            if candidate:
                oc["name"] = candidate
    oc["updated_at"] = now_iso()
    save_db(db)
    return jsonify({"result": response, "name": oc.get("name", ""), "filtered": filtered})


@app.route("/oc/<oc_id>/quick-fix", methods=["POST"])
def quick_fix_oc_route(oc_id: str):
    ensure_dirs()
    db = load_db()
    oc = get_oc(db, oc_id)
    if not oc:
        return jsonify({"error": "OC not found"}), 404
    source_text = sanitize_text(oc.get("existing_text", "") or oc.get("result_text", "") or "")
    if not source_text:
        return jsonify({"error": "Add Existing OC Text or generate a Result first."}), 400
    settings = load_settings()
    quick_settings = settings.copy()
    quick_settings["temperature"] = max(MIN_TEMPERATURE, min(0.2, float(settings.get("temperature", 0.0))))
    quick_settings["max_tokens"] = max(200, min(2000, int(settings.get("max_tokens", 1200))))
    try:
        response = call_llm(quick_settings, build_quick_fix_prompt(source_text))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
    response = sanitize_text(response)
    response, filtered = apply_pre_cut(response, settings)
    oc["result_text"] = response
    oc["updated_at"] = now_iso()
    save_db(db)
    return jsonify({"ok": True, "result": response, "filtered": filtered})


@app.route("/oc/<oc_id>/export")
def export_oc_route(oc_id: str):
    ensure_dirs()
    db = load_db()
    oc = get_oc(db, oc_id)
    if not oc:
        return jsonify({"error": "OC not found"}), 404
    fmt = str(request.args.get("format", "txt")).strip().lower()
    safe_name = "".join(c for c in (oc.get("name") or "oc") if c.isalnum() or c in {"-", "_", " "}).strip()
    safe_name = (safe_name or "oc").replace(" ", "_")
    if fmt == "json":
        payload = json.dumps(oc, indent=2, ensure_ascii=False)
        return Response(
            payload,
            mimetype="application/json",
            headers={"Content-Disposition": f'attachment; filename="{safe_name}.json"'},
        )
    text = sanitize_text(oc.get("result_text", "") or oc.get("existing_text", "") or oc.get("prompt", ""))
    return Response(
        text,
        mimetype="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}.txt"'},
    )


@app.route("/providers/save", methods=["POST"])
def save_provider_profile_route():
    ensure_dirs()
    payload = request.get_json(silent=True) or {}
    profile_id = str(payload.get("id", "")).strip()
    profiles = load_provider_profiles()
    provider = str(payload.get("provider") or "openai").strip()
    defaults = PROVIDER_DEFAULTS.get(provider, PROVIDER_DEFAULTS["openai"])
    profile = {
        "id": profile_id or uuid.uuid4().hex[:10],
        "name": str(payload.get("name", "")).strip() or "Unnamed Provider",
        "provider": provider,
        "model": str(payload.get("model", "")).strip() or defaults["model"],
        "base_url": str(payload.get("base_url", "")).strip() or defaults["base_url"],
        "api_key": str(payload.get("api_key", "")).strip(),
        "notes": str(payload.get("notes", "")).strip(),
        "updated_at": now_iso(),
    }
    replaced = False
    for i, existing in enumerate(profiles):
        if existing.get("id") == profile["id"]:
            profiles[i] = profile
            replaced = True
            break
    if not replaced:
        profiles.append(profile)
    save_provider_profiles(profiles)
    return jsonify({"ok": True, "profile": profile})


@app.route("/providers/<profile_id>", methods=["DELETE"])
def delete_provider_profile_route(profile_id: str):
    ensure_dirs()
    profiles = load_provider_profiles()
    new_profiles = [item for item in profiles if item.get("id") != profile_id]
    if len(new_profiles) == len(profiles):
        return jsonify({"error": "Provider profile not found"}), 404
    save_provider_profiles(new_profiles)
    return jsonify({"ok": True})


@app.route("/settings", methods=["POST"])
def save_settings_route():
    ensure_dirs()
    payload = request.get_json(silent=True) or {}
    settings = load_settings()
    settings.update({k: v for k, v in payload.items() if k in DEFAULT_SETTINGS})
    settings = clamp_settings_values(settings)
    save_settings(settings)
    return jsonify({"ok": True, "settings": settings})


@app.route("/settings/test", methods=["POST"])
def test_settings_route():
    ensure_dirs()
    payload = request.get_json(silent=True) or {}
    settings = load_settings()
    settings.update({k: v for k, v in payload.items() if k in DEFAULT_SETTINGS})
    settings = clamp_settings_values(settings)
    test_prompt = "Reply with exactly: hi"
    test_settings = settings.copy()
    test_settings["max_tokens"] = 40
    test_settings["temperature"] = max(MIN_TEMPERATURE, min(0.3, float(settings.get("temperature", 0.0))))
    try:
        result = call_llm(test_settings, test_prompt)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    return jsonify(
        {
            "ok": True,
            "provider": settings.get("provider", ""),
            "model": settings.get("model", ""),
            "preview": sanitize_text(result)[:120],
        }
    )


SECURE_LOGIN_TEMPLATE = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{{ app_title }} - Secure Login</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700&display=swap" rel="stylesheet" />
  <style>
    :root {
      --bg: #081018;
      --bg2: #0e1724;
      --card: rgba(16, 24, 38, 0.94);
      --ink: #f5f7fb;
      --muted: #97a3b6;
      --accent: #37caa0;
      --accent2: #5ea7ff;
      --border: rgba(132, 152, 181, 0.22);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: "Manrope", "Segoe UI", system-ui, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(900px circle at 20% 0%, rgba(55, 202, 160, 0.14), transparent 40%),
        radial-gradient(900px circle at 85% 5%, rgba(94, 167, 255, 0.17), transparent 45%),
        linear-gradient(135deg, var(--bg), var(--bg2));
      display: grid;
      place-items: center;
      padding: 20px;
    }
    .card {
      width: min(420px, 100%);
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 18px;
      box-shadow: 0 20px 40px rgba(0, 0, 0, 0.45);
    }
    h1 { margin: 0 0 6px; font-size: 22px; }
    p { margin: 0 0 12px; color: var(--muted); font-size: 14px; }
    label { display: block; font-size: 12px; color: var(--muted); margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.06em; }
    input {
      width: 100%;
      border: 1px solid var(--border);
      background: #0c1323;
      color: var(--ink);
      border-radius: 10px;
      padding: 10px 12px;
      margin-bottom: 12px;
      font-size: 14px;
    }
    .btn {
      width: 100%;
      border: 0;
      border-radius: 999px;
      padding: 10px 14px;
      font-weight: 700;
      cursor: pointer;
      background: linear-gradient(120deg, var(--accent), var(--accent2));
      color: #041015;
    }
    .mode-toggle {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-bottom: 12px;
    }
    .mode-btn {
      border: 1px solid var(--border);
      background: #0c1323;
      color: var(--muted);
      border-radius: 10px;
      padding: 9px 10px;
      font-weight: 700;
      cursor: pointer;
    }
    .mode-btn.active {
      color: #041015;
      border-color: transparent;
      background: linear-gradient(120deg, var(--accent), var(--accent2));
    }
    .mode-btn[disabled] {
      opacity: 0.55;
      cursor: default;
    }
    .toggle {
      display: flex;
      gap: 8px;
      align-items: center;
      margin-bottom: 12px;
      color: var(--muted);
      font-size: 12px;
    }
    .toggle input {
      width: auto;
      margin: 0;
    }
    .error {
      margin: 0 0 10px;
      border: 1px solid rgba(248, 113, 113, 0.4);
      background: rgba(220, 38, 38, 0.14);
      color: #fecaca;
      border-radius: 10px;
      padding: 8px 10px;
      font-size: 13px;
    }
  </style>
</head>
<body>
  <form class="card" method="post" action="{{ url_for('secure_login_route', next=next_path) }}">
    <h1>{{ app_title }} Secure Login</h1>
    {% if has_users %}
    <p>Sign in or create your own private account for this device.</p>
    {% else %}
    <p>No account exists yet. Create the first private account.</p>
    {% endif %}
    {% if error %}
    <div class="error">{{ error }}</div>
    {% endif %}
    <input type="hidden" name="next" value="{{ next_path }}" />
    <input type="hidden" name="action" id="auth_action" value="{{ mode }}" />
    <div class="mode-toggle">
      <button id="mode_login" class="mode-btn" type="button" {% if not has_users %}disabled{% endif %}>Sign In</button>
      <button id="mode_signup" class="mode-btn" type="button">Sign Up</button>
    </div>
    <label>Username</label>
    <input name="username" type="text" autocomplete="username" value="{{ username }}" required />
    <label>Password</label>
    <input id="pw" name="password" type="password" autocomplete="current-password" required />
    <div id="confirm_wrap">
      <label>Confirm Password</label>
      <input id="pw2" name="confirm" type="password" autocomplete="new-password" />
    </div>
    <label class="toggle"><input id="show_pw" type="checkbox" /> Show password</label>
    <button class="btn" id="submit_btn" type="submit">Enter App</button>
  </form>
  <script>
    const show = document.getElementById('show_pw');
    const pw = document.getElementById('pw');
    const pw2 = document.getElementById('pw2');
    const modeLogin = document.getElementById('mode_login');
    const modeSignup = document.getElementById('mode_signup');
    const actionInput = document.getElementById('auth_action');
    const confirmWrap = document.getElementById('confirm_wrap');
    const submitBtn = document.getElementById('submit_btn');
    const hasUsers = {{ has_users|tojson }};
    const defaultMode = {{ mode|tojson }};
    const seenKey = 'ocreator_seen_secure_login';

    function setMode(next) {
      const mode = next === 'signup' ? 'signup' : 'login';
      if (actionInput) actionInput.value = mode;
      if (modeLogin) modeLogin.classList.toggle('active', mode === 'login');
      if (modeSignup) modeSignup.classList.toggle('active', mode === 'signup');
      if (confirmWrap) confirmWrap.style.display = mode === 'signup' ? 'block' : 'none';
      if (pw2) {
        pw2.required = mode === 'signup';
      }
      if (submitBtn) {
        submitBtn.textContent = mode === 'signup' ? 'Create Account' : 'Enter App';
      }
    }

    const seenBefore = localStorage.getItem(seenKey) === '1';
    let initialMode = defaultMode === 'signup' ? 'signup' : 'login';
    if (hasUsers && !seenBefore) {
      initialMode = 'signup';
    }
    setMode(initialMode);
    localStorage.setItem(seenKey, '1');

    if (modeLogin && hasUsers) {
      modeLogin.addEventListener('click', () => setMode('login'));
    }
    if (modeSignup) {
      modeSignup.addEventListener('click', () => setMode('signup'));
    }

    if (show) {
      show.addEventListener('change', () => {
        const next = show.checked ? 'text' : 'password';
        if (pw) pw.type = next;
        if (pw2) pw2.type = next;
      });
    }
  </script>
</body>
</html>
"""


TEMPLATE = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{{ app_title }}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet" />
  <style>
    :root {
      --bg: #0b0d12;
      --bg-2: #121622;
      --ink: #f8fafc;
      --muted: #9aa2b2;
      --accent: #1fbf93;
      --accent-2: #41b6ff;
      --card: rgba(15, 18, 28, 0.92);
      --border: rgba(140, 160, 185, 0.18);
      --shadow: rgba(0, 0, 0, 0.45);
      --surface: #101521;
      --input-bg: #0f1626;
      --soft: rgba(18, 25, 38, 0.8);
      --focus: rgba(31, 191, 147, 0.35);
      --glow: rgba(31, 191, 147, 0.3);
      --radius: 18px;
    }
    :root[data-theme="light"] {
      --bg: #f7f8fb;
      --bg-2: #fdfdfd;
      --ink: #101520;
      --muted: #5d6575;
      --accent: #1b9c72;
      --accent-2: #3179ff;
      --card: rgba(255, 255, 255, 0.96);
      --border: rgba(45, 57, 78, 0.12);
      --shadow: rgba(46, 56, 76, 0.12);
      --surface: #f2f4f8;
      --input-bg: #f7f9fc;
      --soft: rgba(236, 241, 248, 0.8);
      --focus: rgba(27, 156, 114, 0.2);
      --glow: rgba(27, 156, 114, 0.18);
    }
    :root[data-theme="dark"] {
      --bg: #0a0c12;
      --bg-2: #101522;
      --ink: #f4f6fb;
      --muted: #8e97aa;
      --accent: #19c08b;
      --accent-2: #3fb4ff;
      --card: rgba(12, 16, 25, 0.94);
      --border: rgba(130, 150, 180, 0.16);
      --shadow: rgba(0, 0, 0, 0.55);
      --surface: #0c111d;
      --input-bg: #0b1220;
      --soft: rgba(17, 24, 36, 0.9);
      --focus: rgba(25, 192, 139, 0.35);
      --glow: rgba(25, 192, 139, 0.35);
    }
    :root[data-theme="stellar"] { --accent: #6ee7ff; --accent-2: #e879f9; --bg: #080a12; --bg-2: #111528; }
    :root[data-theme="nebula"] { --accent: #fb7185; --accent-2: #38bdf8; --bg: #0a0a16; --bg-2: #14112b; }
    :root[data-theme="solar"] { --accent: #fbbf24; --accent-2: #fb7185; --bg: #14100a; --bg-2: #211309; }
    :root[data-theme="lunar"] { --accent: #a5b4fc; --accent-2: #22d3ee; --bg: #0b0f18; --bg-2: #121a29; }
    :root[data-theme="eclipse"] { --accent: #34d399; --accent-2: #6366f1; --bg: #06080f; --bg-2: #0f1422; }
    :root[data-theme="aurora"] { --accent: #34d399; --accent-2: #38bdf8; --bg: #0a1115; --bg-2: #0f1923; }
    :root[data-theme="cosmos"] { --accent: #f472b6; --accent-2: #60a5fa; --bg: #0b0b18; --bg-2: #131028; }
    :root[data-theme="supernova"] { --accent: #f97316; --accent-2: #f43f5e; --bg: #140b0a; --bg-2: #1d0f0b; }
    :root[data-theme="void"] { --accent: #22d3ee; --accent-2: #a78bfa; --bg: #020308; --bg-2: #0b0f1a; }
    :root[data-theme="orbit"] { --accent: #60a5fa; --accent-2: #34d399; --bg: #0b0f17; --bg-2: #121a2a; }
    :root[data-theme="plasma"] { --accent: #ec4899; --accent-2: #f59e0b; --bg: #110714; --bg-2: #1a0a1d; }
    :root[data-theme="meteor"] { --accent: #f97316; --accent-2: #22d3ee; --bg: #100c0c; --bg-2: #171010; }
    :root[data-theme="gdimp"] {
      --accent: #00ff66;
      --accent-2: #ff00aa;
      --bg: #ff0000;
      --bg-2: #00a2ff;
      --ink: #111111;
      --muted: #1b1b1b;
      --card: rgba(255, 255, 255, 0.9);
      --border: rgba(0, 0, 0, 0.35);
      --surface: #fff761;
      --input-bg: #ffffff;
      --soft: rgba(255, 255, 255, 0.8);
      --focus: rgba(255, 0, 170, 0.35);
      --glow: rgba(0, 255, 102, 0.35);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: "Manrope", "Segoe UI", system-ui, sans-serif;
      color: var(--ink);
      background: radial-gradient(1200px circle at 15% 0%, rgba(31, 191, 147, 0.15), transparent 40%),
                  radial-gradient(1000px circle at 85% 10%, rgba(65, 182, 255, 0.18), transparent 45%),
                  linear-gradient(135deg, var(--bg), var(--bg-2));
    }
    :root[data-theme="gdimp"] body {
      font-family: "Comic Sans MS", "Manrope", "Segoe UI", cursive, sans-serif;
      background: linear-gradient(
        90deg,
        #ff3b3b 0%,
        #ff8a00 16%,
        #ffe45c 32%,
        #44d66b 48%,
        #23b5ff 64%,
        #6a5cff 80%,
        #d34bff 100%
      );
      background-size: 100% 100%;
      animation: gdimp-rainbow-shift 70s ease-in-out infinite;
    }
    :root[data-theme="gdimp"] * {
      font-family: "Comic Sans MS", "Comic Sans", cursive, sans-serif !important;
    }
    :root[data-theme="gdimp"] header {
      background: linear-gradient(
        90deg,
        #ff3b3b 0%,
        #ff8a00 16%,
        #ffe45c 32%,
        #44d66b 48%,
        #23b5ff 64%,
        #6a5cff 80%,
        #d34bff 100%
      );
      background-size: 100% 100%;
      animation: gdimp-rainbow-shift 85s ease-in-out infinite;
      border-bottom-color: rgba(0, 0, 0, 0.4);
    }
    @keyframes gdimp-rainbow-shift {
      0% { background-position: 0% 50%; }
      50% { background-position: 4% 50%; }
      100% { background-position: 0% 50%; }
    }
    html, body {
      max-width: 100%;
      overflow-x: hidden;
    }
    header {
      position: sticky;
      top: 0;
      z-index: 20;
      padding: 20px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid var(--border);
      background: linear-gradient(120deg, rgba(8, 10, 16, 0.92), rgba(10, 14, 22, 0.86));
      backdrop-filter: blur(20px);
    }
    .fx-layer {
      position: fixed;
      inset: 0;
      pointer-events: none;
      overflow: hidden;
      z-index: 1;
    }
    .fx-dot, .fx-comet, .fx-spark {
      position: absolute;
      border-radius: 999px;
      opacity: 0;
      will-change: transform, opacity;
    }
    .fx-dot {
      width: 3px;
      height: 3px;
      background: var(--fx-dot, rgba(255, 255, 255, 0.75));
      box-shadow: 0 0 10px var(--fx-dot-glow, rgba(255, 255, 255, 0.45));
      animation: fx-twinkle 7s linear infinite;
    }
    .fx-comet {
      width: 120px;
      height: 2px;
      background: linear-gradient(90deg, rgba(255, 255, 255, 0), var(--fx-comet, rgba(255, 175, 72, 0.95)), rgba(255, 255, 255, 0));
      filter: drop-shadow(0 0 8px var(--fx-comet-glow, rgba(255, 174, 66, 0.55)));
      animation: fx-comet 8s linear infinite;
    }
    .fx-spark {
      width: 2px;
      height: 2px;
      background: var(--fx-spark, rgba(255, 225, 170, 0.9));
      box-shadow: 0 0 8px var(--fx-spark-glow, rgba(255, 196, 112, 0.5));
      animation: fx-float 6s ease-in-out infinite;
    }
    @keyframes fx-twinkle {
      0%, 100% { opacity: 0.1; transform: translateY(0) scale(0.8); }
      50% { opacity: 0.85; transform: translateY(-6px) scale(1.2); }
    }
    @keyframes fx-float {
      0%, 100% { opacity: 0; transform: translate3d(0, 0, 0); }
      25% { opacity: 0.8; }
      75% { opacity: 0.4; transform: translate3d(16px, -22px, 0); }
    }
    @keyframes fx-comet {
      0% { opacity: 0; transform: translate3d(-15vw, -5vh, 0) rotate(-18deg); }
      8% { opacity: 0.95; }
      35% { opacity: 0.8; transform: translate3d(40vw, 18vh, 0) rotate(-18deg); }
      100% { opacity: 0; transform: translate3d(120vw, 58vh, 0) rotate(-18deg); }
    }
    .brand {
      font-family: "Space Grotesk", "Manrope", sans-serif;
      font-size: 22px;
      font-weight: 700;
      letter-spacing: 0.02em;
    }
    .sub {
      font-size: 12px;
      color: var(--muted);
      margin-top: 4px;
    }
    .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
    [data-tip] { position: relative; }
    [data-tip]::after {
      content: attr(data-tip);
      position: absolute;
      left: 50%;
      bottom: calc(100% + 8px);
      transform: translateX(-50%) translateY(4px);
      background: rgba(6, 10, 18, 0.95);
      color: #e2e8f0;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 6px 8px;
      font-size: 11px;
      line-height: 1.25;
      width: max-content;
      max-width: 240px;
      text-align: center;
      opacity: 0;
      pointer-events: none;
      z-index: 120;
      box-shadow: 0 10px 20px rgba(0, 0, 0, 0.35);
      transition: opacity 120ms ease, transform 120ms ease;
      white-space: normal;
    }
    [data-tip]:hover::after, [data-tip]:focus-visible::after {
      opacity: 1;
      transform: translateX(-50%) translateY(0);
    }
    .btn {
      border: none;
      padding: 9px 16px;
      border-radius: 999px;
      background: linear-gradient(120deg, var(--accent), var(--accent-2));
      color: #041015;
      font-weight: 700;
      cursor: pointer;
      box-shadow: 0 10px 24px var(--glow);
      transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .btn:hover { transform: translateY(-1px); }
    .btn.secondary {
      background: linear-gradient(120deg, var(--accent-2), var(--accent));
    }
    .btn.ghost {
      background: transparent;
      color: var(--muted);
      border: 1px solid var(--border);
      box-shadow: none;
    }
    :root[data-theme="gdimp"] .btn,
    :root[data-theme="gdimp"] .panel,
    :root[data-theme="gdimp"] input,
    :root[data-theme="gdimp"] select,
    :root[data-theme="gdimp"] textarea {
      border-radius: 0;
      box-shadow: none;
    }
    .help-btn {
      width: 36px;
      height: 36px;
      border-radius: 12px;
      font-weight: 700;
      padding: 0;
    }
    .conn-toast {
      position: fixed;
      top: 14px;
      left: 50%;
      transform: translateX(-50%) translateY(-8px);
      padding: 9px 12px;
      border-radius: 10px;
      border: 1px solid var(--border);
      font-size: 13px;
      font-weight: 700;
      display: none;
      align-items: center;
      gap: 8px;
      z-index: 140;
      opacity: 0;
      transition: opacity 160ms ease, transform 160ms ease;
      box-shadow: 0 10px 22px rgba(0, 0, 0, 0.35);
      backdrop-filter: blur(10px);
      max-width: calc(100vw - 20px);
      white-space: normal;
      overflow-wrap: anywhere;
    }
    .conn-toast.show {
      display: inline-flex;
      opacity: 1;
      transform: translateX(-50%) translateY(0);
    }
    .conn-toast.ok {
      background: rgba(22, 163, 74, 0.18);
      border-color: rgba(34, 197, 94, 0.55);
      color: #86efac;
    }
    .conn-toast.err {
      background: rgba(220, 38, 38, 0.18);
      border-color: rgba(248, 113, 113, 0.55);
      color: #fca5a5;
    }
    .conn-toast .icon {
      font-size: 14px;
      font-weight: 900;
      line-height: 1;
    }
    .gdimp-video-wrap {
      position: fixed;
      right: 12px;
      bottom: 98px;
      width: min(420px, calc(100vw - 24px));
      border: 2px solid rgba(0, 0, 0, 0.45);
      background: rgba(255, 255, 255, 0.9);
      box-shadow: 0 12px 24px rgba(0, 0, 0, 0.35);
      display: none;
      z-index: 35;
    }
    .gdimp-video-wrap.hidden-by-user {
      display: none !important;
    }
    .gdimp-video-close {
      position: absolute;
      top: 6px;
      right: 6px;
      width: 28px;
      height: 28px;
      border: 1px solid rgba(0, 0, 0, 0.45);
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.94);
      color: #111;
      font-weight: 900;
      cursor: pointer;
      z-index: 2;
      line-height: 1;
    }
    .gdimp-video-frame {
      width: 100%;
      aspect-ratio: 16 / 9;
      border: 0;
      display: block;
    }
    :root[data-theme="gdimp"] .gdimp-video-wrap {
      display: block;
    }
    .layout {
      display: grid;
      grid-template-columns: 300px 1fr 290px 320px;
      gap: 20px;
      padding: 24px;
      position: relative;
      z-index: 2;
      max-width: 100%;
    }
    .mobile-switcher {
      display: none;
    }
    .mobile-header-menu-btn {
      display: none;
    }
    .desktop-header-actions {
      display: inline-flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }
    .mobile-actions-drawer {
      display: none;
    }
    .mobile-switcher button .icon {
      display: none;
      font-size: 14px;
      line-height: 1;
    }
    .mobile-switcher button .label {
      display: inline;
    }
    .panel {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 16px;
      box-shadow: 0 14px 30px var(--shadow);
      animation: panel-rise 420ms ease both;
    }
    .layout > .panel:nth-child(2) { animation-delay: 70ms; }
    .layout > .panel:nth-child(3) { animation-delay: 140ms; }
    .layout > .panel:nth-child(4) { animation-delay: 210ms; }
    .layout > .panel:nth-child(5) { animation-delay: 250ms; }
    .layout .panel.security-panel { grid-column: 1 / -1; }
    @keyframes panel-rise {
      from { opacity: 0; transform: translateY(12px) scale(0.99); }
      to { opacity: 1; transform: translateY(0) scale(1); }
    }
    .panel h3, .panel h2 { margin: 0 0 10px; font-family: "Space Grotesk", sans-serif; }
    .list {
      display: flex;
      flex-direction: column;
      gap: 10px;
      max-height: 70vh;
      overflow-y: auto;
      overflow-x: hidden;
    }
    .oc-card {
      position: relative;
      padding: 12px;
      border-radius: 14px;
      border: 1px solid var(--border);
      background: var(--surface);
      transition: transform 0.24s ease, border 0.24s ease, box-shadow 0.24s ease;
      z-index: 1;
    }
    .oc-card.menu-open { z-index: 30; }
    .oc-card:hover { transform: translateY(-2px); }
    .oc-card.active {
      border-color: var(--accent);
      box-shadow: 0 10px 18px rgba(31, 191, 147, 0.2);
    }
    .oc-link {
      text-decoration: none;
      color: var(--ink);
      display: block;
      padding-right: 34px;
    }
    .oc-card .title { font-weight: 700; }
    .oc-card.pinned .title::before {
      content: "Pinned â€¢ ";
      color: var(--accent);
      font-weight: 700;
    }
    .oc-card .meta {
      font-size: 12px;
      color: var(--muted);
    }
    .oc-menu-wrap {
      position: absolute;
      top: 8px;
      right: 8px;
    }
    .oc-menu-btn {
      border: 1px solid var(--border);
      background: rgba(255, 255, 255, 0.02);
      color: var(--muted);
      width: 28px;
      height: 28px;
      border-radius: 9px;
      cursor: pointer;
      display: grid;
      place-items: center;
      transition: border-color 0.2s ease, color 0.2s ease, transform 0.2s ease;
      font-size: 16px;
      line-height: 1;
    }
    .oc-menu-btn:hover {
      border-color: var(--accent);
      color: var(--ink);
      transform: translateY(-1px);
    }
    .oc-menu {
      position: absolute;
      top: 32px;
      right: 0;
      width: 156px;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: var(--card);
      box-shadow: 0 12px 22px var(--shadow);
      display: none;
      overflow: hidden;
      transform: translateY(6px) scale(0.98);
      transform-origin: top right;
      opacity: 0;
      transition: opacity 0.18s ease, transform 0.18s ease;
      z-index: 50;
    }
    .oc-menu.open {
      display: block;
      opacity: 1;
      transform: translateY(0) scale(1);
    }
    .oc-menu button {
      width: 100%;
      border: none;
      background: transparent;
      color: var(--ink);
      text-align: left;
      padding: 9px 11px;
      cursor: pointer;
      font-family: inherit;
      font-size: 13px;
    }
    .oc-menu button:hover { background: var(--soft); }
    .oc-menu button[data-action="delete"] { color: #fb7185; }
    label {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 6px;
      display: block;
    }
    input, select, textarea {
      width: 100%;
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: var(--input-bg);
      color: var(--ink);
      font-family: inherit;
      font-size: 14px;
    }
    input:focus, select:focus, textarea:focus {
      outline: none;
      border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--focus);
    }
    textarea { min-height: 120px; resize: vertical; }
    .grid {
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    }
    .mode-toggle {
      display: inline-flex;
      padding: 4px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: var(--soft);
      gap: 4px;
    }
    .mode-toggle button {
      border: none;
      background: transparent;
      color: var(--muted);
      padding: 6px 14px;
      border-radius: 999px;
      cursor: pointer;
      font-weight: 600;
    }
    .mode-toggle button.active {
      background: linear-gradient(120deg, var(--accent), var(--accent-2));
      color: #041015;
    }
    .status {
      font-size: 12px;
      color: var(--muted);
    }
    .result {
      min-height: 240px;
      font-family: "Fira Code", ui-monospace, monospace;
      white-space: pre-wrap;
    }
    .theme-control {
      display: flex;
      align-items: center;
      gap: 12px;
      position: relative;
      min-width: 0;
    }
    .theme-picker {
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      padding: 6px;
      border-radius: 12px;
      background: var(--soft);
      border: 1px solid var(--border);
    }
    .theme-option {
      width: 26px;
      height: 26px;
      border-radius: 8px;
      border: 1px solid transparent;
      padding: 0;
      background: transparent;
      cursor: pointer;
      display: grid;
      place-items: center;
    }
    .theme-option.active { border-color: var(--accent); }
    .theme-swatch {
      width: 16px;
      height: 16px;
      border-radius: 999px;
      background: linear-gradient(120deg, var(--accent), var(--accent-2));
    }
    .theme-swatch[data-theme="system"] { background: linear-gradient(120deg, #94a3b8, #cbd5f5); }
    .theme-swatch[data-theme="light"] { background: linear-gradient(120deg, #d1fae5, #dbeafe); }
    .theme-swatch[data-theme="dark"] { background: linear-gradient(120deg, #0f172a, #334155); }
    .theme-swatch[data-theme="stellar"] { background: linear-gradient(120deg, #6ee7ff, #e879f9); }
    .theme-swatch[data-theme="nebula"] { background: linear-gradient(120deg, #fb7185, #38bdf8); }
    .theme-swatch[data-theme="solar"] { background: linear-gradient(120deg, #fbbf24, #fb7185); }
    .theme-swatch[data-theme="lunar"] { background: linear-gradient(120deg, #a5b4fc, #22d3ee); }
    .theme-swatch[data-theme="eclipse"] { background: linear-gradient(120deg, #34d399, #6366f1); }
    .theme-swatch[data-theme="aurora"] { background: linear-gradient(120deg, #34d399, #38bdf8); }
    .theme-swatch[data-theme="cosmos"] { background: linear-gradient(120deg, #f472b6, #60a5fa); }
    .theme-swatch[data-theme="supernova"] { background: linear-gradient(120deg, #f97316, #f43f5e); }
    .theme-swatch[data-theme="void"] { background: linear-gradient(120deg, #22d3ee, #a78bfa); }
    .theme-swatch[data-theme="orbit"] { background: linear-gradient(120deg, #60a5fa, #34d399); }
    .theme-swatch[data-theme="plasma"] { background: linear-gradient(120deg, #ec4899, #f59e0b); }
    .theme-swatch[data-theme="meteor"] { background: linear-gradient(120deg, #f97316, #22d3ee); }
    .theme-swatch[data-theme="gdimp"] {
      background-image: url("/assets/GDIMP.jpg");
      background-size: cover;
      background-position: center;
      border-radius: 4px;
    }
    .theme-select { display: none; }
    .effects-menu-btn {
      border: 1px solid var(--border);
      background: var(--soft);
      color: var(--muted);
      border-radius: 10px;
      padding: 7px 10px;
      cursor: pointer;
      font-size: 12px;
      font-weight: 700;
    }
    .effects-panel {
      position: absolute;
      right: 0;
      top: 42px;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      min-width: 220px;
      padding: 12px;
      box-shadow: 0 14px 28px var(--shadow);
      display: none;
      z-index: 60;
    }
    .effects-panel.open { display: block; }
    .effects-line {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      font-size: 13px;
      color: var(--muted);
    }
    .effects-line input { width: auto; }
    .hint { font-size: 12px; color: var(--muted); }
    .provider-list {
      display: flex;
      flex-direction: column;
      gap: 8px;
      margin-top: 12px;
      max-height: 38vh;
      overflow: auto;
    }
    .provider-card {
      border: 1px solid var(--border);
      border-radius: 12px;
      background: var(--surface);
      padding: 10px;
      display: grid;
      gap: 8px;
      transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .provider-card:hover {
      transform: translateY(-1px);
      box-shadow: 0 10px 16px rgba(0, 0, 0, 0.2);
    }
    .provider-card .meta {
      font-size: 12px;
      color: var(--muted);
    }
    .provider-row {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }
    .provider-row .btn {
      box-shadow: none;
      padding: 7px 10px;
      font-size: 12px;
    }
    .ai-tab-toggle {
      margin-bottom: 12px;
    }
    .ai-tab-content {
      display: none;
    }
    .ai-tab-content.active {
      display: block;
    }
    .security-users {
      display: flex;
      flex-direction: column;
      gap: 8px;
      margin-top: 10px;
      max-height: 34vh;
      overflow: auto;
    }
    .security-user-card {
      border: 1px solid var(--border);
      border-radius: 12px;
      background: var(--surface);
      padding: 10px;
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
    }
    .security-user-card .meta {
      font-size: 12px;
      color: var(--muted);
    }
    .security-user-card .btn {
      box-shadow: none;
      padding: 6px 10px;
      font-size: 12px;
    }
    .modal {
      position: fixed;
      inset: 0;
      display: none;
      align-items: center;
      justify-content: center;
      background: rgba(8, 10, 15, 0.6);
      z-index: 50;
    }
    .modal.open { display: flex; animation: fade-in 200ms ease both; }
    @keyframes fade-in {
      from { opacity: 0; }
      to { opacity: 1; }
    }
    .modal-card {
      width: min(980px, 92vw);
      max-height: 90vh;
      overflow: auto;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 20px;
      padding: 20px;
      box-shadow: 0 20px 40px var(--shadow);
      position: relative;
      animation: modal-rise 220ms ease both;
    }
    @keyframes modal-rise {
      from { opacity: 0; transform: translateY(16px) scale(0.99); }
      to { opacity: 1; transform: translateY(0) scale(1); }
    }
    .help-grid {
      display: grid;
      gap: 16px;
      grid-template-columns: 1.1fr 1fr;
    }
    .faq-card, .help-chat {
      padding: 16px;
      border-radius: 16px;
      border: 1px solid var(--border);
      background: var(--surface);
    }
    .help-chat-log {
      min-height: 220px;
      max-height: 280px;
      overflow: auto;
      padding: 10px;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: rgba(7, 10, 16, 0.4);
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .help-chat-bubble {
      padding: 10px 12px;
      border-radius: 12px;
      font-size: 13px;
      line-height: 1.4;
    }
    .help-chat-bubble.user {
      align-self: flex-end;
      background: rgba(31, 191, 147, 0.2);
      border: 1px solid rgba(31, 191, 147, 0.3);
    }
    .help-chat-bubble.bot {
      align-self: flex-start;
      background: rgba(65, 182, 255, 0.18);
      border: 1px solid rgba(65, 182, 255, 0.3);
    }
    .help-mode-toggle {
      display: inline-flex;
      gap: 6px;
      padding: 4px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: var(--soft);
    }
    .help-mode-toggle button {
      border: none;
      padding: 6px 12px;
      border-radius: 999px;
      cursor: pointer;
      background: transparent;
      color: var(--muted);
      font-weight: 600;
    }
    .help-mode-toggle button.active {
      background: linear-gradient(120deg, var(--accent), var(--accent-2));
      color: #041015;
    }
    .onboarding-modal .modal-card { max-width: 860px; }
    .onboard-hero {
      padding: 20px;
      border-radius: 18px;
      background: linear-gradient(130deg, rgba(31, 191, 147, 0.22), rgba(65, 182, 255, 0.2));
      display: grid;
      gap: 12px;
    }
    .onboard-hero h2 { margin: 0; font-family: "Space Grotesk", sans-serif; }
    .onboard-step-card {
      padding: 16px;
      border-radius: 14px;
      border: 1px solid var(--border);
      background: var(--surface);
      min-height: 140px;
    }
    .onboard-progress {
      height: 8px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.08);
      overflow: hidden;
    }
    .onboard-progress-bar {
      height: 100%;
      width: 0%;
      background: linear-gradient(120deg, var(--accent), var(--accent-2));
      transition: width 0.3s ease;
    }
    @media (max-width: 1600px) {
      .layout { grid-template-columns: 280px 1fr 300px; }
      .layout .panel.provider-panel { grid-column: 1 / -1; }
      .layout .panel.security-panel { grid-column: 1 / -1; }
    }
    @media (max-width: 1200px) {
      .layout { grid-template-columns: 280px 1fr; }
      .layout .panel.library-panel { order: 3; }
      .layout .panel.provider-panel { order: 4; grid-column: 1 / -1; }
      .layout .panel.security-panel { order: 5; grid-column: 1 / -1; }
    }
    @media (max-width: 980px) {
      header {
        padding: 12px;
        gap: 10px;
        flex-direction: column;
        align-items: stretch;
      }
      .sub {
        font-size: 11px;
      }
      header > .row {
        justify-content: space-between;
      }
      .desktop-header-actions {
        display: none;
      }
      .mobile-header-menu-btn {
        display: inline-flex;
        align-items: center;
        justify-content: center;
      }
      .mobile-actions-drawer {
        position: fixed;
        inset: 0;
        z-index: 40;
        display: none;
        background: rgba(5, 8, 14, 0.64);
        align-items: flex-start;
        justify-content: center;
        padding: 14px 12px;
        overflow-y: auto;
        overflow-x: hidden;
      }
      .mobile-actions-drawer.open {
        display: flex;
      }
      .mobile-actions-card {
        width: 100%;
        max-width: 520px;
        border-radius: 14px;
        border: 1px solid var(--border);
        background: var(--card);
        box-shadow: 0 16px 28px rgba(0, 0, 0, 0.4);
        padding: 10px;
        display: grid;
        gap: 8px;
        overflow-x: hidden;
      }
      .mobile-actions-row {
        display: grid;
        gap: 8px;
        grid-template-columns: 1fr;
      }
      .mobile-actions-row > * {
        min-width: 0;
      }
      .mobile-actions-card .btn,
      .mobile-actions-card form .btn {
        width: 100%;
        justify-content: center;
        white-space: normal;
      }
      .mobile-actions-card form {
        margin: 0;
        width: 100%;
      }
      .theme-control {
        width: 100%;
        justify-content: space-between;
      }
      .theme-picker {
        max-width: 100%;
        overflow-x: hidden;
        flex-wrap: wrap;
      }
      .theme-option {
        flex: 0 0 auto;
      }
      .layout { grid-template-columns: 1fr; }
      .help-grid { grid-template-columns: 1fr; }
      .layout .panel.provider-panel { grid-column: auto; }
      .layout .panel.security-panel { grid-column: auto; }
      .layout {
        padding: 12px;
        gap: 12px;
      }
      .layout > .panel {
        display: none;
      }
      .layout > .panel.mobile-active {
        display: block;
      }
      .mobile-switcher {
        position: fixed;
        left: 10px;
        right: 10px;
        bottom: 10px;
        z-index: 30;
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 8px;
        padding: 8px;
        border-radius: 14px;
        border: 1px solid var(--border);
        background: rgba(7, 10, 16, 0.9);
        backdrop-filter: blur(12px);
        box-shadow: 0 10px 26px rgba(0, 0, 0, 0.35);
      }
      .mobile-switcher button {
        border: 1px solid var(--border);
        background: transparent;
        color: var(--muted);
        border-radius: 10px;
        padding: 8px 6px;
        font-size: 12px;
        font-weight: 700;
        cursor: pointer;
      }
      .mobile-switcher button.active {
        color: #041015;
        border-color: transparent;
        background: linear-gradient(120deg, var(--accent), var(--accent-2));
      }
      body {
        padding-bottom: 84px;
      }
    }
    @media (max-width: 420px) {
      .brand {
        font-size: 19px;
      }
      .mobile-switcher {
        left: 8px;
        right: 8px;
        bottom: 8px;
        gap: 6px;
        padding: 6px;
      }
      .mobile-switcher button {
        padding: 8px 4px;
        font-size: 11px;
      }
      .mobile-switcher button .icon {
        display: inline;
      }
      .mobile-switcher button .label {
        display: none;
      }
      .row {
        gap: 8px;
      }
      .btn {
        padding: 8px 12px;
      }
      .panel {
        padding: 12px;
      }
      .gdimp-video-wrap {
        right: 8px;
        bottom: 84px;
        width: calc(100vw - 16px);
      }
    }
    @media (prefers-reduced-motion: reduce) {
      * {
        animation: none !important;
        transition: none !important;
      }
    }
  </style>
</head>
<body>
  <div id="fx_layer" class="fx-layer" aria-hidden="true"></div>
  <div class="gdimp-video-wrap" aria-label="GDIMP theme video">
    <button type="button" class="gdimp-video-close" id="gdimp_video_close_btn" aria-label="Close theme video">X</button>
    <iframe
      id="gdimp_video_frame"
      class="gdimp-video-frame"
      data-src="https://www.youtube.com/embed/_VRTa81j_cY?autoplay=1&mute=1&loop=1&playlist=_VRTa81j_cY"
      title="Graphics Design Is My Passion"
      allow="autoplay; encrypted-media; picture-in-picture"
      allowfullscreen
    ></iframe>
  </div>
  <div id="conn_toast" class="conn-toast" aria-live="polite">
    <span id="conn_toast_icon" class="icon">âœ“</span>
    <span id="conn_toast_text"></span>
  </div>
  <header>
    <div>
      <div class="brand">{{ app_title }}</div>
      <div class="sub">Build playable OC personas from scratch or refine an existing OC text block.</div>
    </div>
    <div class="row">
      <div class="theme-control">
        <div class="theme-picker" id="theme_picker" role="radiogroup" aria-label="Theme" data-tip="Pick a color theme for the app UI.">
          <button type="button" class="theme-option" data-theme="system"><span class="theme-swatch" data-theme="system"></span></button>
          <button type="button" class="theme-option" data-theme="light"><span class="theme-swatch" data-theme="light"></span></button>
          <button type="button" class="theme-option" data-theme="dark"><span class="theme-swatch" data-theme="dark"></span></button>
          <button type="button" class="theme-option" data-theme="stellar"><span class="theme-swatch" data-theme="stellar"></span></button>
          <button type="button" class="theme-option" data-theme="nebula"><span class="theme-swatch" data-theme="nebula"></span></button>
          <button type="button" class="theme-option" data-theme="solar"><span class="theme-swatch" data-theme="solar"></span></button>
          <button type="button" class="theme-option" data-theme="lunar"><span class="theme-swatch" data-theme="lunar"></span></button>
          <button type="button" class="theme-option" data-theme="eclipse"><span class="theme-swatch" data-theme="eclipse"></span></button>
          <button type="button" class="theme-option" data-theme="aurora"><span class="theme-swatch" data-theme="aurora"></span></button>
          <button type="button" class="theme-option" data-theme="cosmos"><span class="theme-swatch" data-theme="cosmos"></span></button>
          <button type="button" class="theme-option" data-theme="supernova"><span class="theme-swatch" data-theme="supernova"></span></button>
          <button type="button" class="theme-option" data-theme="void"><span class="theme-swatch" data-theme="void"></span></button>
          <button type="button" class="theme-option" data-theme="orbit"><span class="theme-swatch" data-theme="orbit"></span></button>
          <button type="button" class="theme-option" data-theme="plasma"><span class="theme-swatch" data-theme="plasma"></span></button>
          <button type="button" class="theme-option" data-theme="meteor"><span class="theme-swatch" data-theme="meteor"></span></button>
          <button type="button" class="theme-option" data-theme="gdimp"><span class="theme-swatch" data-theme="gdimp"></span></button>
        </div>
        <select id="theme_select" class="theme-select" aria-hidden="true" tabindex="-1">
          <option value="system">System</option>
          <option value="light">Light</option>
          <option value="dark">Dark</option>
          <option value="stellar">Stellar</option>
          <option value="nebula">Nebula</option>
          <option value="solar">Solar</option>
          <option value="lunar">Lunar</option>
          <option value="eclipse">Eclipse</option>
          <option value="aurora">Aurora</option>
          <option value="cosmos">Cosmos</option>
          <option value="supernova">Supernova</option>
          <option value="void">Void</option>
          <option value="orbit">Orbit</option>
          <option value="plasma">Plasma</option>
          <option value="meteor">Meteor</option>
          <option value="gdimp">GDIMP</option>
        </select>
        <button class="effects-menu-btn" id="effects_menu_btn" type="button" data-tip="Toggle ambient visual effects for the selected theme.">Effects</button>
        <div class="effects-panel" id="effects_panel">
          <div class="effects-line">
            <span>Theme effects</span>
            <input id="effects_enabled" type="checkbox" checked />
          </div>
          <div class="hint" style="margin-top: 8px;">Warm themes use comets and sparks. Cool themes use stars.</div>
        </div>
      </div>
      <div class="desktop-header-actions">
        <button class="btn ghost" type="button" onclick="toggleOnboard(true)">Guide</button>
        <button class="btn ghost help-btn" type="button" onclick="toggleHelp(true)" title="Help">?</button>
        <button class="btn ghost" type="button" onclick="saveOC()" data-tip="Save your current OC fields now.">Save</button>
        <form method="post" action="{{ url_for('create_oc_route') }}">
          <button class="btn secondary" type="submit">New OC</button>
        </form>
        {% if security_enabled and current_username %}
        <form method="post" action="{{ url_for('logout_route') }}">
          <button class="btn ghost" type="submit">Logout ({{ current_username }})</button>
        </form>
        {% endif %}
      </div>
      <button class="btn ghost mobile-header-menu-btn" id="mobile_actions_btn" type="button" aria-expanded="false" aria-controls="mobile_actions_drawer">More</button>
    </div>
  </header>

  <div class="mobile-actions-drawer" id="mobile_actions_drawer" aria-hidden="true">
    <div class="mobile-actions-card">
      <div class="row" style="justify-content: space-between;">
        <strong>Quick Actions</strong>
        <button class="btn ghost" type="button" id="mobile_actions_close_btn">Close</button>
      </div>
      <div class="mobile-actions-row">
        <button class="btn ghost" type="button" onclick="toggleOnboard(true); setMobileActionsOpen(false);">Guide</button>
        <button class="btn ghost" type="button" onclick="toggleHelp(true); setMobileActionsOpen(false);">Help</button>
        <button class="btn ghost" type="button" onclick="saveOC(); setMobileActionsOpen(false);">Save</button>
        <form method="post" action="{{ url_for('create_oc_route') }}">
          <button class="btn secondary" type="submit">New OC</button>
        </form>
        {% if security_enabled and current_username %}
        <form method="post" action="{{ url_for('logout_route') }}">
          <button class="btn ghost" type="submit">Logout ({{ current_username }})</button>
        </form>
        {% endif %}
      </div>
    </div>
  </div>

  <nav class="mobile-switcher" id="mobile_switcher" aria-label="Mobile quick sections">
    <button type="button" data-mobile-target="editor" aria-label="Editor panel"><span class="icon">&#9998;</span><span class="label">Editor</span></button>
    <button type="button" data-mobile-target="ai" aria-label="AI settings panel"><span class="icon">&#9881;</span><span class="label">AI</span></button>
    <button type="button" data-mobile-target="library" aria-label="OC library panel"><span class="icon">&#9776;</span><span class="label">Library</span></button>
    <button type="button" data-mobile-target="vault" aria-label="Provider vault panel"><span class="icon">&#8993;</span><span class="label">Vault</span></button>
    <button type="button" data-mobile-target="security" aria-label="Security panel"><span class="icon">&#128274;</span><span class="label">Secure</span></button>
  </nav>

  <div class="layout">
    <aside class="panel" id="panel_ai" data-panel-name="ai">
      <h3>AI Settings</h3>
      <div class="mode-toggle ai-tab-toggle" id="ai_tab_toggle">
        <button type="button" data-ai-tab="provider">Provider</button>
        <button type="button" data-ai-tab="precut">Pre-Cut</button>
      </div>

      <div class="ai-tab-content active" data-ai-tab-content="provider">
        <div class="grid">
          <div>
            <label>Saved Profile</label>
            <select id="ai_profile_select" data-tip="Select a saved provider profile to quickly fill AI settings.">
              <option value="">Manual / Current Settings</option>
              {% for profile in provider_profiles %}
              <option value="{{ profile.id }}">{{ profile.name or 'Unnamed Provider' }} ({{ profile.provider }})</option>
              {% endfor %}
            </select>
          </div>
          <div>
            <label>Provider</label>
            <select id="ai_provider" data-tip="Choose which AI provider endpoint to call for generation.">
              {% for provider in providers %}
              <option value="{{ provider.id }}" {% if settings.provider == provider.id %}selected{% endif %}>{{ provider.label }}</option>
              {% endfor %}
            </select>
          </div>
          <div>
            <label>Model</label>
            <input id="ai_model" value="{{ settings.model }}" />
          </div>
          <div>
            <label>API Key</label>
            <input id="ai_key" type="password" value="{{ settings.api_key }}" placeholder="Enter API key" />
          </div>
          <div>
            <label>Base URL</label>
            <input id="ai_base_url" value="{{ settings.base_url }}" />
          </div>
          <div>
            <label>Temperature</label>
            <input id="ai_temp" type="number" min="0" max="2" step="0.1" value="{{ settings.temperature }}" />
          </div>
          <div>
            <label>Max Tokens</label>
            <input id="ai_tokens" type="number" step="1" value="{{ settings.max_tokens }}" />
          </div>
          <div>
            <button class="btn ghost" type="button" onclick="saveSettings()" data-tip="Store provider, model, key, and tuning values in your current user folder.">Save AI Settings</button>
          </div>
          <div>
            <button class="btn ghost" type="button" onclick="applySelectedProfileToSettings()" data-tip="Copy the selected saved profile into the AI Settings fields.">Apply Selected Profile</button>
          </div>
          <div>
            <button class="btn ghost" type="button" onclick="testConnection()" data-tip="Run a quick provider ping to confirm key/model/base URL works.">Test Connection</button>
          </div>
        </div>
      </div>

      <div class="ai-tab-content" data-ai-tab-content="precut">
        <div class="grid">
          <div>
            <label>Enable String Truncate</label>
            <select id="ai_pre_cut_enabled">
              <option value="false" {% if not settings.pre_cut_enabled %}selected{% endif %}>Off</option>
              <option value="true" {% if settings.pre_cut_enabled %}selected{% endif %}>On</option>
            </select>
          </div>
          <div style="grid-column: 1 / -1;">
            <label>Truncate From These Phrases (one per line)</label>
            <textarea id="ai_pre_cut_markers" placeholder="Want the best AI models?&#10;Sponsored">{{ settings.pre_cut_markers }}</textarea>
          </div>
          <div style="grid-column: 1 / -1;">
            <div class="hint">If any phrase appears in a response, text is cut from that phrase onward.</div>
          </div>
          <div>
            <label>Enable Word Replacer</label>
            <select id="ai_pre_cut_replace_enabled">
              <option value="false" {% if not settings.pre_cut_replace_enabled %}selected{% endif %}>Off</option>
              <option value="true" {% if settings.pre_cut_replace_enabled %}selected{% endif %}>On</option>
            </select>
          </div>
          <div style="grid-column: 1 / -1;">
            <label>Replacement Rules (one per line)</label>
            <textarea id="ai_pre_cut_replace_rules" placeholder="bad phrase => better phrase&#10;unfavorable => neutral">{{ settings.pre_cut_replace_rules }}</textarea>
          </div>
          <div>
            <button class="btn ghost" type="button" onclick="saveSettings()">Save Pre-Cut Settings</button>
          </div>
        </div>
      </div>

      <div style="margin-top: 16px;">
        <h3>Status</h3>
        <div class="status" style="margin-bottom: 10px;">
          <div class="hint">Progress</div>
          <div id="status_progress">Ready.</div>
        </div>
        <div class="status">
          <div class="hint">Save</div>
          <div id="status_save">Not saved yet.</div>
        </div>
      </div>
    </aside>

    <main class="panel" id="panel_editor" data-panel-name="editor">
      <div class="grid">
        <div>
          <label>OC Name</label>
          <input id="oc_name" value="{{ oc.name }}" placeholder="OC name" />
        </div>
        <div>
          <label>Tab Name</label>
          <input id="oc_tab_name" value="{{ oc.tab_name }}" placeholder="Name shown in OC Library" />
        </div>
        <div>
          <label>Mode</label>
          <div class="mode-toggle" id="mode_toggle">
            <button type="button" data-mode="scratch">Create</button>
            <button type="button" data-mode="existing">Modify</button>
            <button type="button" data-mode="enhance">Enhance</button>
          </div>
        </div>
        <div>
          <label>Template</label>
          <select id="template_mode">
            <option value="default">Default Template</option>
            <option value="custom">Custom Template</option>
            <option value="none">No Template</option>
          </select>
        </div>
      </div>

      <div class="grid" style="margin-top: 12px;">
        <div>
          <label>Prompt (single text entry)</label>
          <textarea id="oc_prompt" placeholder="Describe the OC idea or prompt.">{{ oc.prompt }}</textarea>
        </div>
        <div>
          <label>Existing OC Text</label>
          <textarea id="oc_existing" placeholder="Paste a full OC persona text block.">{{ oc.existing_text }}</textarea>
        </div>
      </div>

      <div class="grid" style="margin-top: 12px;">
        <div>
          <label>Enhancement Notes</label>
          <textarea id="oc_enhance" placeholder="What should be improved, expanded, or corrected?">{{ oc.enhance_notes }}</textarea>
        </div>
        <div>
          <label>Custom Template</label>
          <textarea id="oc_template" placeholder="{{ default_template }}">{{ oc.custom_template }}</textarea>
        </div>
      </div>

      <div class="row" style="margin-top: 12px;">
        <button class="btn" type="button" onclick="generateOC()" data-tip="Generate OC output using your current mode and settings.">Generate</button>
        <button class="btn ghost" type="button" onclick="quickFixOC()" data-tip="Fix grammar/spelling/structure without adding new lore.">Quick Fix</button>
        <button class="btn ghost" type="button" onclick="applyResult()" data-tip="Copy result text into Existing OC Text for another pass.">Apply Result</button>
      </div>

      <div style="margin-top: 12px;">
        <label>Result (single text persona)</label>
        <textarea id="oc_result" class="result" readonly>{{ oc.result_text }}</textarea>
      </div>
    </main>

    <aside class="panel library-panel" id="panel_library" data-panel-name="library">
      <h3>OC Library</h3>
      <div class="list" style="margin-bottom: 16px;">
        {% for item in ocs %}
        <div class="oc-card {% if item.id == oc.id %}active{% endif %} {% if item.pinned %}pinned{% endif %}" data-oc-id="{{ item.id }}">
          <a class="oc-link" href="{{ url_for('edit_oc', oc_id=item.id) }}">
            <div class="title">{{ item.tab_name or item.name or 'Untitled OC' }}</div>
            <div class="meta">{{ item.updated_at or 'New' }}</div>
          </a>
          <div class="oc-menu-wrap">
            <button class="oc-menu-btn" type="button" title="OC actions" data-menu-trigger aria-label="Open OC menu">â˜°</button>
            <div class="oc-menu" data-menu>
              <button type="button" data-action="duplicate">Duplicate</button>
              <button type="button" data-action="pin">{% if item.pinned %}Unpin{% else %}Pin{% endif %}</button>
              <button type="button" data-action="export_txt">Export TXT</button>
              <button type="button" data-action="export_json">Export JSON</button>
              <button type="button" data-action="delete">Delete</button>
            </div>
          </div>
        </div>
        {% endfor %}
      </div>
    </aside>

    <aside class="panel provider-panel" id="panel_vault" data-panel-name="vault">
      <h3>Provider Vault</h3>
      <div class="grid">
        <input type="hidden" id="vault_id" />
        <div>
          <label>Profile Name</label>
          <input id="vault_name" placeholder="My OpenRouter Key" />
        </div>
        <div>
          <label>Provider</label>
          <select id="vault_provider">
            {% for provider in providers %}
            <option value="{{ provider.id }}">{{ provider.label }}</option>
            {% endfor %}
          </select>
        </div>
        <div>
          <label>Model</label>
          <input id="vault_model" placeholder="Model override (optional)" />
        </div>
        <div>
          <label>Base URL</label>
          <input id="vault_base_url" placeholder="Base URL override (optional)" />
        </div>
        <div>
          <label>API Key</label>
          <input id="vault_key" type="password" placeholder="Stored in your user folder" />
        </div>
        <div>
          <label>Notes</label>
          <input id="vault_notes" placeholder="Optional notes" />
        </div>
      </div>
      <div class="row" style="margin-top: 10px;">
        <button class="btn ghost" type="button" onclick="saveProviderProfile()" data-tip="Save these provider credentials/details in the local vault.">Save Profile</button>
        <button class="btn ghost" type="button" onclick="clearProviderForm()">Clear</button>
      </div>
      <div class="hint" id="provider_status" style="margin-top: 8px;">Profiles are saved in your current user folder.</div>
      <div class="provider-list" id="provider_list">
        {% for profile in provider_profiles %}
        <div class="provider-card" data-profile-id="{{ profile.id }}">
          <div>
            <strong>{{ profile.name or 'Unnamed Provider' }}</strong>
            <div class="meta">{{ profile.provider }}{% if profile.model %} &bull; {{ profile.model }}{% endif %}</div>
          </div>
          <div class="provider-row">
            <button class="btn ghost" type="button" onclick="useProviderProfile('{{ profile.id }}')" data-tip="Load this profile into AI Settings.">Use</button>
            <button class="btn ghost" type="button" onclick="editProviderProfile('{{ profile.id }}')" data-tip="Load this profile into the vault form for editing.">Edit</button>
            <button class="btn ghost" type="button" onclick="deleteProviderProfile('{{ profile.id }}')" data-tip="Delete this saved provider profile.">Delete</button>
          </div>
        </div>
        {% endfor %}
      </div>
    </aside>

    <aside class="panel security-panel" id="panel_security" data-panel-name="security">
      <h3>Secure Login</h3>
      <div class="grid">
        <div>
          <label>Enable Secure Login</label>
          <select id="security_enabled">
            <option value="false" {% if not security_enabled %}selected{% endif %}>Off</option>
            <option value="true" {% if security_enabled %}selected{% endif %}>On</option>
          </select>
        </div>
        <div>
          <label>Active User</label>
          <input id="security_current_user" value="{{ current_username or 'Not signed in' }}" readonly />
        </div>
        <div>
          <label>New Password (Current User)</label>
          <input id="security_password" type="password" placeholder="At least 8 characters" />
        </div>
      </div>
      <div class="row" style="margin-top: 10px;">
        <button class="btn ghost" type="button" onclick="saveSecurityPassword()">Update My Password</button>
        <button class="btn ghost" type="button" onclick="saveSecurityConfig()">Apply Security Toggle</button>
        <button class="btn ghost" type="button" onclick="refreshSecurityStatus()">Refresh</button>
      </div>
      <div class="hint" id="security_status" style="margin-top: 8px;">
        Secure login is {% if security_enabled %}enabled{% else %}disabled{% endif %}. Accounts are private and not listed to other users.
      </div>
      <div class="hint" style="margin-top: 4px;">
        New devices are prompted to Sign In or Sign Up here: <a href="{{ url_for('secure_login_route') }}" style="color: var(--accent-2);" target="_self">Open Secure Login</a>
      </div>
    </aside>
  </div>

  <div class="modal" id="help_modal" aria-hidden="true">
    <div class="modal-card">
      <div class="row" style="justify-content: space-between; margin-bottom: 12px;">
        <h2>Help Desk</h2>
        <button class="btn ghost" type="button" onclick="toggleHelp(false)">Close</button>
      </div>
      <div class="help-grid">
        <div class="faq-card">
          <h3>FAQ</h3>
          <p><strong>How do I start fast?</strong> Pick Create mode, add a short prompt, Generate, then Apply Result to refine.</p>
          <p><strong>What is Modify?</strong> Paste an existing OC block and clean it up while keeping identity consistent.</p>
          <p><strong>What is Enhance?</strong> Provide existing text plus improvements to expand details or fix gaps.</p>
          <p><strong>Template tips?</strong> Default templates keep labeled sections. Custom templates follow your labels exactly.</p>
          <p><strong>Output seems short?</strong> Raise Max Tokens in AI Settings or use a larger model.</p>
        </div>
        <div class="help-chat">
          <div class="row" style="justify-content: space-between;">
            <div class="help-mode-toggle" id="help_mode_toggle">
              <button type="button" data-mode="create">Create</button>
              <button type="button" data-mode="modify">Modify</button>
              <button type="button" data-mode="enhance">Enhance</button>
            </div>
            <select id="help_model">
              <option value="grok-4-fast">grok-4-fast</option>
              <option value="gpt-5-nano">gpt-5-nano</option>
              <option value="claude-sonnet-4.5">claude-sonnet-4.5</option>
              <option value="gemini-2.5-flash">gemini-2.5-flash</option>
            </select>
          </div>
          <div class="hint" id="help_status_hint" style="margin-top: 8px;">Uses Puter.js. Chat history stays in your browser.</div>
          <div id="help_chat_log" class="help-chat-log"></div>
          <textarea id="help_chat_input" class="help-chat-input" placeholder="Ask anything about OCreator..."></textarea>
          <div class="row" style="margin-top: 10px;">
            <button class="btn" type="button" onclick="sendHelpChat()">Ask</button>
            <button class="btn ghost" type="button" onclick="clearHelpChat()">Clear</button>
            <button class="btn ghost" type="button" onclick="signInHelpChat()">Sign in</button>
          </div>
        </div>
      </div>
    </div>
  </div>

  <div class="modal onboarding-modal" id="onboard_modal" aria-hidden="true">
    <div class="modal-card">
      <div class="row" style="justify-content: space-between; margin-bottom: 12px;">
        <h2>Quick Start Guide</h2>
        <button class="btn ghost" type="button" onclick="toggleOnboard(false)">Close</button>
      </div>
      <div class="onboard-hero">
        <span class="hint">Fast path from idea to a playable persona.</span>
        <h2>Make an OC in minutes</h2>
        <div class="onboard-progress">
          <div class="onboard-progress-bar" id="onboard_progress"></div>
        </div>
      </div>
      <div class="onboard-step-card" id="onboard_step_card" style="margin-top: 16px;">
        <h3 id="onboard_step_title">Pick a mode</h3>
        <p id="onboard_step_body">Create for new characters, Modify for cleanup, Enhance for targeted improvements.</p>
      </div>
      <div class="row" style="justify-content: space-between; margin-top: 12px;">
        <div class="hint" id="onboard_step_count">Step 1 of 5</div>
        <div class="row">
          <button class="btn ghost" type="button" id="onboard_back_btn" onclick="prevOnboardStep()">Back</button>
          <button class="btn" type="button" id="onboard_next_btn" onclick="nextOnboardStep()">Next</button>
        </div>
      </div>
    </div>
  </div>

  <div class="modal" id="confirm_modal" aria-hidden="true">
    <div class="modal-card" style="max-width: 460px;">
      <div class="row" style="justify-content: space-between; margin-bottom: 8px;">
        <h2 id="confirm_title">Confirm</h2>
      </div>
      <p id="confirm_body" class="hint" style="font-size: 14px; margin-top: 0;"></p>
      <div class="row" style="justify-content: flex-end; margin-top: 12px;">
        <button class="btn ghost" type="button" id="confirm_cancel_btn">Cancel</button>
        <button class="btn" type="button" id="confirm_ok_btn">Delete</button>
      </div>
    </div>
  </div>

  <script src="https://js.puter.com/v2/"></script>
  <script>
    const OC_ID = "{{ oc.id }}";
    const modeToggle = document.getElementById('mode_toggle');
    const templateMode = document.getElementById('template_mode');
    const statusProgressEl = document.getElementById('status_progress');
    const statusSaveEl = document.getElementById('status_save');
    const aiProfileSelectEl = document.getElementById('ai_profile_select');
    const aiTabToggleEl = document.getElementById('ai_tab_toggle');
    const preCutEnabledEl = document.getElementById('ai_pre_cut_enabled');
    const preCutMarkersEl = document.getElementById('ai_pre_cut_markers');
    const preCutReplaceEnabledEl = document.getElementById('ai_pre_cut_replace_enabled');
    const preCutReplaceRulesEl = document.getElementById('ai_pre_cut_replace_rules');
    const helpModalEl = document.getElementById('help_modal');
    const helpModeToggleEl = document.getElementById('help_mode_toggle');
    const helpModelEl = document.getElementById('help_model');
    const helpChatLogEl = document.getElementById('help_chat_log');
    const helpChatInputEl = document.getElementById('help_chat_input');
    const helpStatusHintEl = document.getElementById('help_status_hint');
    const themeSelect = document.getElementById('theme_select');
    const themeButtons = document.querySelectorAll('.theme-option');
    const mobileActionsBtnEl = document.getElementById('mobile_actions_btn');
    const mobileActionsDrawerEl = document.getElementById('mobile_actions_drawer');
    const mobileActionsCloseBtnEl = document.getElementById('mobile_actions_close_btn');
    const mobileSwitcherEl = document.getElementById('mobile_switcher');
    const connToastEl = document.getElementById('conn_toast');
    const connToastIconEl = document.getElementById('conn_toast_icon');
    const connToastTextEl = document.getElementById('conn_toast_text');
    const gdimpVideoWrapEl = document.querySelector('.gdimp-video-wrap');
    const gdimpVideoFrameEl = document.getElementById('gdimp_video_frame');
    const gdimpVideoCloseBtnEl = document.getElementById('gdimp_video_close_btn');
    const fxLayerEl = document.getElementById('fx_layer');
    const effectsMenuBtnEl = document.getElementById('effects_menu_btn');
    const effectsPanelEl = document.getElementById('effects_panel');
    const effectsEnabledEl = document.getElementById('effects_enabled');
    const onboardModalEl = document.getElementById('onboard_modal');
    const confirmModalEl = document.getElementById('confirm_modal');
    const confirmTitleEl = document.getElementById('confirm_title');
    const confirmBodyEl = document.getElementById('confirm_body');
    const confirmOkBtnEl = document.getElementById('confirm_ok_btn');
    const confirmCancelBtnEl = document.getElementById('confirm_cancel_btn');
    const onboardProgressEl = document.getElementById('onboard_progress');
    const onboardStepTitleEl = document.getElementById('onboard_step_title');
    const onboardStepBodyEl = document.getElementById('onboard_step_body');
    const onboardStepCountEl = document.getElementById('onboard_step_count');
    const onboardNextBtnEl = document.getElementById('onboard_next_btn');
    const onboardBackBtnEl = document.getElementById('onboard_back_btn');
    const providerStatusEl = document.getElementById('provider_status');
    const securityEnabledEl = document.getElementById('security_enabled');
    const securityCurrentUserEl = document.getElementById('security_current_user');
    const securityPasswordEl = document.getElementById('security_password');
    const securityStatusEl = document.getElementById('security_status');
    const providerDefaults = {{ provider_defaults|tojson }};
    const providerProfiles = {{ provider_profiles|tojson }};
    const initialSecurityEnabled = {{ security_enabled|tojson }};
    const initialCurrentUsername = {{ current_username|tojson }};

    const HELP_HISTORY_KEY = 'ocreator_help_history';
    const HELP_MODE_KEY = 'ocreator_help_mode';
    const HELP_MODEL_KEY = 'ocreator_help_model';
    const EFFECTS_ENABLED_KEY = 'ocreator_effects_enabled';
    const MOBILE_PANEL_KEY = 'ocreator_mobile_panel';
    const AI_TAB_KEY = 'ocreator_ai_tab';

    let activeThemeName = 'system';
    let confirmResolver = null;
    let connToastTimer = null;

    const ONBOARD_STEPS = [
      { title: 'Pick a mode', body: 'Create for new characters, Modify for cleanup, Enhance for targeted improvements.' },
      { title: 'Add your input', body: 'Use a short prompt or paste a full OC text block depending on your mode.' },
      { title: 'Choose a template', body: 'Default uses labels, Custom uses your exact format, None creates a clean block.' },
      { title: 'Generate', body: 'Hit Generate to get a clean persona output based on your instructions.' },
      { title: 'Apply and refine', body: 'Use Apply Result to copy the output into Existing for a second pass.' },
    ];

    let onboardIndex = 0;

    function setStatus(text) {
      if (statusProgressEl) statusProgressEl.textContent = text;
    }

    function setSaveStatus(text) {
      if (statusSaveEl) statusSaveEl.textContent = text;
    }

    function setMode(mode) {
      document.querySelectorAll('#mode_toggle button').forEach((btn) => {
        btn.classList.toggle('active', btn.dataset.mode === mode);
      });
      localStorage.setItem('ocreator_mode', mode);
    }

    function initMode() {
      const current = "{{ oc.mode }}" || localStorage.getItem('ocreator_mode') || 'scratch';
      setMode(current);
    }

    function setAITab(name) {
      const next = name === 'precut' ? 'precut' : 'provider';
      document.querySelectorAll('[data-ai-tab-content]').forEach((panel) => {
        panel.classList.toggle('active', panel.dataset.aiTabContent === next);
      });
      if (aiTabToggleEl) {
        aiTabToggleEl.querySelectorAll('button[data-ai-tab]').forEach((btn) => {
          btn.classList.toggle('active', btn.dataset.aiTab === next);
        });
      }
      localStorage.setItem(AI_TAB_KEY, next);
    }

    function initAITabs() {
      if (!aiTabToggleEl) return;
      const saved = localStorage.getItem(AI_TAB_KEY) || 'provider';
      setAITab(saved);
      aiTabToggleEl.querySelectorAll('button[data-ai-tab]').forEach((btn) => {
        btn.addEventListener('click', () => setAITab(btn.dataset.aiTab || 'provider'));
      });
    }

    function setMobilePanel(name) {
      const target = name || 'editor';
      document.querySelectorAll('.layout > .panel').forEach((panel) => {
        panel.classList.toggle('mobile-active', panel.dataset.panelName === target);
      });
      if (!mobileSwitcherEl) return;
      mobileSwitcherEl.querySelectorAll('button[data-mobile-target]').forEach((btn) => {
        btn.classList.toggle('active', btn.dataset.mobileTarget === target);
      });
    }

    function initMobilePanels() {
      const media = window.matchMedia('(max-width: 980px)');
      if (mobileSwitcherEl) {
        mobileSwitcherEl.querySelectorAll('button[data-mobile-target]').forEach((btn) => {
          btn.addEventListener('click', () => {
            const next = btn.dataset.mobileTarget || 'editor';
            localStorage.setItem(MOBILE_PANEL_KEY, next);
            setMobilePanel(next);
          });
        });
      }
      const apply = () => {
        if (media.matches) {
          const saved = localStorage.getItem(MOBILE_PANEL_KEY) || 'editor';
          setMobilePanel(saved);
          return;
        }
        document.querySelectorAll('.layout > .panel').forEach((panel) => panel.classList.remove('mobile-active'));
      };
      apply();
      if (media.addEventListener) {
        media.addEventListener('change', apply);
      } else if (media.addListener) {
        media.addListener(apply);
      }
    }

    function initTemplate() {
      templateMode.value = "{{ oc.template_mode }}";
    }

    function setMobileActionsOpen(show) {
      if (!mobileActionsDrawerEl || !mobileActionsBtnEl) return;
      mobileActionsDrawerEl.classList.toggle('open', show);
      mobileActionsDrawerEl.setAttribute('aria-hidden', show ? 'false' : 'true');
      mobileActionsBtnEl.setAttribute('aria-expanded', show ? 'true' : 'false');
    }

    function initMobileActions() {
      if (!mobileActionsBtnEl || !mobileActionsDrawerEl) return;
      mobileActionsBtnEl.addEventListener('click', () => {
        const open = !mobileActionsDrawerEl.classList.contains('open');
        setMobileActionsOpen(open);
      });
      if (mobileActionsCloseBtnEl) {
        mobileActionsCloseBtnEl.addEventListener('click', () => setMobileActionsOpen(false));
      }
      mobileActionsDrawerEl.addEventListener('click', (event) => {
        if (event.target === mobileActionsDrawerEl) setMobileActionsOpen(false);
      });
      document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape' && mobileActionsDrawerEl.classList.contains('open')) {
          setMobileActionsOpen(false);
        }
      });
    }

    function collectPayload() {
      return {
        name: document.getElementById('oc_name').value,
        tab_name: document.getElementById('oc_tab_name').value,
        mode: document.querySelector('#mode_toggle button.active')?.dataset.mode || 'scratch',
        prompt: document.getElementById('oc_prompt').value,
        existing_text: document.getElementById('oc_existing').value,
        enhance_notes: document.getElementById('oc_enhance').value,
        template_mode: templateMode.value,
        custom_template: document.getElementById('oc_template').value,
        result_text: document.getElementById('oc_result').value,
      };
    }

    async function saveOC(silent) {
      if (!silent) {
        setSaveStatus('Saving...');
      }
      const payload = collectPayload();
      const res = await fetch(`/oc/${OC_ID}/save`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) {
        if (!silent) {
          setStatus(data.error || 'Save failed');
        }
        setSaveStatus(data.error || 'Save failed');
        return;
      }
      const now = new Date();
      const stamp = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      setSaveStatus(`Autosaved at ${stamp}`);
      if (!silent) {
        setSaveStatus('Saved.');
      }
    }

    async function saveSettings() {
      const payload = clampSettingsPayload({
        provider: document.getElementById('ai_provider').value,
        model: document.getElementById('ai_model').value,
        api_key: document.getElementById('ai_key').value,
        base_url: document.getElementById('ai_base_url').value,
        temperature: parseFloat(document.getElementById('ai_temp').value || '0.7'),
        max_tokens: parseInt(document.getElementById('ai_tokens').value || '1200', 10),
        pre_cut_enabled: preCutEnabledEl ? preCutEnabledEl.value === 'true' : false,
        pre_cut_markers: preCutMarkersEl ? preCutMarkersEl.value : '',
        pre_cut_replace_enabled: preCutReplaceEnabledEl ? preCutReplaceEnabledEl.value === 'true' : false,
        pre_cut_replace_rules: preCutReplaceRulesEl ? preCutReplaceRulesEl.value : '',
      });
      document.getElementById('ai_temp').value = String(payload.temperature);
      document.getElementById('ai_tokens').value = String(payload.max_tokens);
      if (preCutEnabledEl) preCutEnabledEl.value = payload.pre_cut_enabled ? 'true' : 'false';
      if (preCutMarkersEl) preCutMarkersEl.value = payload.pre_cut_markers || '';
      if (preCutReplaceEnabledEl) preCutReplaceEnabledEl.value = payload.pre_cut_replace_enabled ? 'true' : 'false';
      if (preCutReplaceRulesEl) preCutReplaceRulesEl.value = payload.pre_cut_replace_rules || '';
      const res = await fetch('/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) {
        setStatus('Settings save failed.');
        return;
      }
      if (data.settings) {
        document.getElementById('ai_temp').value = String(data.settings.temperature);
        document.getElementById('ai_tokens').value = String(data.settings.max_tokens);
        if (preCutEnabledEl) preCutEnabledEl.value = data.settings.pre_cut_enabled ? 'true' : 'false';
        if (preCutMarkersEl) preCutMarkersEl.value = data.settings.pre_cut_markers || '';
        if (preCutReplaceEnabledEl) preCutReplaceEnabledEl.value = data.settings.pre_cut_replace_enabled ? 'true' : 'false';
        if (preCutReplaceRulesEl) preCutReplaceRulesEl.value = data.settings.pre_cut_replace_rules || '';
      }
      setStatus('Settings saved.');
    }

    function clampSettingsPayload(payload) {
      const safe = { ...payload };
      const temp = Number.isFinite(safe.temperature) ? safe.temperature : 0.7;
      safe.temperature = Math.max(0, Math.min(2, temp));
      safe.max_tokens = Number.isFinite(safe.max_tokens) ? Math.round(safe.max_tokens) : 1200;
      safe.pre_cut_enabled = Boolean(safe.pre_cut_enabled);
      safe.pre_cut_replace_enabled = Boolean(safe.pre_cut_replace_enabled);
      safe.pre_cut_markers = String(safe.pre_cut_markers || '').slice(0, 12000);
      safe.pre_cut_replace_rules = String(safe.pre_cut_replace_rules || '').slice(0, 12000);
      return safe;
    }

    function applyProfileToSettings(profile) {
      if (!profile) return;
      document.getElementById('ai_provider').value = profile.provider || 'openai';
      document.getElementById('ai_model').value = profile.model || '';
      document.getElementById('ai_base_url').value = profile.base_url || '';
      document.getElementById('ai_key').value = profile.api_key || '';
      setStatus(`Loaded profile "${profile.name || 'Unnamed Provider'}".`);
    }

    function applySelectedProfileToSettings() {
      if (!aiProfileSelectEl) return;
      const profileId = aiProfileSelectEl.value;
      if (!profileId) {
        setStatus('No saved profile selected.');
        return;
      }
      const profile = providerProfiles.find((item) => item.id === profileId);
      if (!profile) {
        setStatus('Selected profile no longer exists.');
        return;
      }
      applyProfileToSettings(profile);
    }

    async function testConnection() {
      const payload = clampSettingsPayload({
        provider: document.getElementById('ai_provider').value,
        model: document.getElementById('ai_model').value,
        api_key: document.getElementById('ai_key').value,
        base_url: document.getElementById('ai_base_url').value,
        temperature: parseFloat(document.getElementById('ai_temp').value || '0.7'),
        max_tokens: parseInt(document.getElementById('ai_tokens').value || '1200', 10),
      });
      const res = await fetch('/settings/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok || !data.ok) {
        showConnToast(false, data.error || 'Connection test failed.');
        return;
      }
      showConnToast(true, `${data.provider} / ${data.model}`);
    }

    function showConnToast(ok, text) {
      if (!connToastEl || !connToastTextEl || !connToastIconEl) return;
      connToastEl.classList.remove('ok', 'err');
      connToastEl.classList.add(ok ? 'ok' : 'err');
      connToastIconEl.textContent = ok ? 'âœ“' : 'âœ•';
      connToastTextEl.textContent = text || (ok ? 'Connection OK' : 'Connection failed');
      connToastEl.classList.add('show');
      if (connToastTimer) {
        clearTimeout(connToastTimer);
        connToastTimer = null;
      }
      connToastTimer = setTimeout(() => {
        connToastEl.classList.remove('show');
      }, ok ? 2600 : 3600);
    }

    function setProviderStatus(text) {
      if (providerStatusEl) providerStatusEl.textContent = text;
    }

    function fillProviderForm(profile) {
      document.getElementById('vault_id').value = profile.id || '';
      document.getElementById('vault_name').value = profile.name || '';
      document.getElementById('vault_provider').value = profile.provider || 'openai';
      document.getElementById('vault_model').value = profile.model || '';
      document.getElementById('vault_base_url').value = profile.base_url || '';
      document.getElementById('vault_key').value = profile.api_key || '';
      document.getElementById('vault_notes').value = profile.notes || '';
    }

    function clearProviderForm() {
      fillProviderForm({
        id: '',
        name: '',
        provider: document.getElementById('ai_provider').value || 'openai',
        model: '',
        base_url: '',
        api_key: '',
        notes: '',
      });
    }

    function readProviderForm() {
      const provider = document.getElementById('vault_provider').value || 'openai';
      const defaults = providerDefaults[provider] || providerDefaults.openai || { model: '', base_url: '' };
      return {
        id: document.getElementById('vault_id').value.trim(),
        name: document.getElementById('vault_name').value.trim(),
        provider,
        model: document.getElementById('vault_model').value.trim() || defaults.model || '',
        base_url: document.getElementById('vault_base_url').value.trim() || defaults.base_url || '',
        api_key: document.getElementById('vault_key').value.trim(),
        notes: document.getElementById('vault_notes').value.trim(),
      };
    }

    function useProviderProfile(profileId) {
      const profile = providerProfiles.find((item) => item.id === profileId);
      if (!profile) {
        setProviderStatus('Provider profile not found.');
        return;
      }
      applyProfileToSettings(profile);
      if (aiProfileSelectEl) aiProfileSelectEl.value = profile.id;
      setProviderStatus(`Loaded "${profile.name}" into AI Settings.`);
    }

    function editProviderProfile(profileId) {
      const profile = providerProfiles.find((item) => item.id === profileId);
      if (!profile) {
        setProviderStatus('Provider profile not found.');
        return;
      }
      fillProviderForm(profile);
      setProviderStatus(`Editing "${profile.name}".`);
    }

    async function saveProviderProfile() {
      const payload = readProviderForm();
      if (!payload.name) {
        setProviderStatus('Profile name is required.');
        return;
      }
      const res = await fetch('/providers/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) {
        setProviderStatus(data.error || 'Failed to save provider profile.');
        return;
      }
      setProviderStatus('Provider profile saved.');
      window.location.reload();
    }

    async function deleteProviderProfile(profileId) {
      const profile = providerProfiles.find((item) => item.id === profileId);
      if (!profile) return;
      const confirmed = await askConfirm(
        'Delete Provider Profile',
        `Delete provider profile "${profile.name}"? This cannot be undone.`
      );
      if (!confirmed) return;
      const res = await fetch(`/providers/${profileId}`, { method: 'DELETE' });
      const data = await res.json();
      if (!res.ok) {
        setProviderStatus(data.error || 'Failed to delete provider profile.');
        return;
      }
      setProviderStatus('Provider profile deleted.');
      window.location.reload();
    }

    function setSecurityStatus(text) {
      if (securityStatusEl) securityStatusEl.textContent = text;
    }

    async function refreshSecurityStatus() {
      if (!securityEnabledEl) return;
      const res = await fetch('/security/status');
      const data = await res.json();
      if (!res.ok) {
        setSecurityStatus(data.error || 'Failed to read security status.');
        return;
      }
      securityEnabledEl.value = data.enabled ? 'true' : 'false';
      if (securityCurrentUserEl) {
        securityCurrentUserEl.value = data.current_user || 'Not signed in';
      }
      setSecurityStatus(data.enabled ? 'Secure login is enabled.' : 'Secure login is disabled.');
    }

    async function saveSecurityConfig() {
      if (!securityEnabledEl) return;
      const enabled = securityEnabledEl.value === 'true';
      const res = await fetch('/security/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled }),
      });
      const data = await res.json();
      if (!res.ok) {
        setSecurityStatus(data.error || 'Failed to update security toggle.');
        return;
      }
      setSecurityStatus(data.enabled ? 'Secure login enabled.' : 'Secure login disabled.');
      if (!data.enabled && securityCurrentUserEl) {
        securityCurrentUserEl.value = 'Not signed in';
      }
    }

    async function saveSecurityPassword() {
      if (!securityPasswordEl) return;
      const password = securityPasswordEl.value;
      if (!password) {
        setSecurityStatus('Enter a new password first.');
        return;
      }
      const res = await fetch('/security/users/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ password }),
      });
      const data = await res.json();
      if (!res.ok) {
        setSecurityStatus(data.error || 'Failed to update password.');
        return;
      }
      securityPasswordEl.value = '';
      setSecurityStatus('Password updated for current user.');
      await refreshSecurityStatus();
    }

    function initSecurityPanel() {
      if (!securityEnabledEl) return;
      securityEnabledEl.value = initialSecurityEnabled ? 'true' : 'false';
      if (securityCurrentUserEl) {
        securityCurrentUserEl.value = initialCurrentUsername || 'Not signed in';
      }
      if (securityEnabledEl) {
        securityEnabledEl.addEventListener('change', () => {
          setSecurityStatus(`Security toggle set to ${securityEnabledEl.value === 'true' ? 'On' : 'Off'}. Click Apply Security Toggle to save.`);
        });
      }
    }

    async function duplicateOC(ocId) {
      const res = await fetch(`/oc/${ocId}/duplicate`, { method: 'POST' });
      const data = await res.json();
      if (!res.ok) {
        setStatus(data.error || 'Duplicate failed.');
        return;
      }
      window.location.href = `/oc/${data.oc_id}`;
    }

    async function togglePinOC(ocId) {
      const res = await fetch(`/oc/${ocId}/pin`, { method: 'POST' });
      const data = await res.json();
      if (!res.ok) {
        setStatus(data.error || 'Pin update failed.');
        return;
      }
      window.location.reload();
    }

    async function deleteOC(ocId) {
      const confirmed = await askConfirm('Delete OC', 'Delete this OC? This cannot be undone.');
      if (!confirmed) return;
      const res = await fetch(`/oc/${ocId}/delete`, { method: 'POST' });
      const data = await res.json();
      if (!res.ok) {
        setStatus(data.error || 'Delete failed.');
        return;
      }
      window.location.href = `/oc/${data.next_oc_id}`;
    }

    function exportOC(ocId, format) {
      const next = format === 'json' ? 'json' : 'txt';
      window.location.href = `/oc/${ocId}/export?format=${next}`;
    }

    function closeOCMenus() {
      document.querySelectorAll('[data-menu].open').forEach((menu) => menu.classList.remove('open'));
      document.querySelectorAll('.oc-card.menu-open').forEach((card) => card.classList.remove('menu-open'));
    }

    function initOCMenus() {
      document.querySelectorAll('[data-menu-trigger]').forEach((btn) => {
        btn.addEventListener('click', (event) => {
          event.preventDefault();
          event.stopPropagation();
          const card = btn.closest('.oc-card');
          if (!card) return;
          const menu = card.querySelector('[data-menu]');
          if (!menu) return;
          const isOpen = menu.classList.contains('open');
          closeOCMenus();
          if (!isOpen) {
            menu.classList.add('open');
            card.classList.add('menu-open');
          }
        });
      });
      document.querySelectorAll('.oc-menu button').forEach((button) => {
        button.addEventListener('click', (event) => {
          event.preventDefault();
          event.stopPropagation();
          const card = button.closest('.oc-card');
          if (!card) return;
          const ocId = card.dataset.ocId;
          if (!ocId) return;
          const action = button.dataset.action;
          closeOCMenus();
          if (action === 'duplicate') duplicateOC(ocId);
          if (action === 'pin') togglePinOC(ocId);
          if (action === 'export_txt') exportOC(ocId, 'txt');
          if (action === 'export_json') exportOC(ocId, 'json');
          if (action === 'delete') deleteOC(ocId);
        });
      });
      document.addEventListener('click', () => closeOCMenus());
      document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape') closeOCMenus();
        if (event.key === 'Escape' && confirmModalEl?.classList.contains('open')) resolveConfirm(false);
      });
    }

    async function generateOC() {
      await saveOC();
      setStatus('Generating...');
      const res = await fetch(`/oc/${OC_ID}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ notes: '' }),
      });
      const data = await res.json();
      if (!res.ok) {
        setStatus(data.error || 'Generation failed.');
        return;
      }
      document.getElementById('oc_result').value = data.result || '';
      if (data.name) {
        document.getElementById('oc_name').value = data.name;
      }
      setStatus(data.filtered ? 'Done. String filter applied.' : 'Done.');
    }

    async function quickFixOC() {
      await saveOC();
      setStatus('Quick fixing...');
      const res = await fetch(`/oc/${OC_ID}/quick-fix`, { method: 'POST' });
      const data = await res.json();
      if (!res.ok) {
        setStatus(data.error || 'Quick fix failed.');
        return;
      }
      document.getElementById('oc_result').value = data.result || '';
      setStatus(data.filtered ? 'Quick fix complete (string filter applied).' : 'Quick fix complete.');
    }

    function applyResult() {
      const result = document.getElementById('oc_result').value;
      if (!result) return;
      document.getElementById('oc_existing').value = result;
      setStatus('Applied to Existing OC.');
    }

    function toggleHelp(show) {
      if (!helpModalEl) return;
      helpModalEl.classList.toggle('open', show);
      helpModalEl.setAttribute('aria-hidden', show ? 'false' : 'true');
    }

    function toggleOnboard(show) {
      if (!onboardModalEl) return;
      onboardModalEl.classList.toggle('open', show);
      onboardModalEl.setAttribute('aria-hidden', show ? 'false' : 'true');
      if (show) renderOnboardStep(onboardIndex);
    }

    function setConfirmOpen(show) {
      if (!confirmModalEl) return;
      confirmModalEl.classList.toggle('open', show);
      confirmModalEl.setAttribute('aria-hidden', show ? 'false' : 'true');
    }

    function resolveConfirm(value) {
      if (confirmResolver) {
        const next = confirmResolver;
        confirmResolver = null;
        next(Boolean(value));
      }
      setConfirmOpen(false);
    }

    function askConfirm(title, body) {
      if (!confirmModalEl || !confirmTitleEl || !confirmBodyEl) {
        return Promise.resolve(false);
      }
      confirmTitleEl.textContent = title || 'Confirm';
      confirmBodyEl.textContent = body || 'Are you sure?';
      setConfirmOpen(true);
      return new Promise((resolve) => {
        confirmResolver = resolve;
      });
    }

    function renderOnboardStep(index) {
      onboardIndex = Math.min(Math.max(index, 0), ONBOARD_STEPS.length - 1);
      const step = ONBOARD_STEPS[onboardIndex];
      if (onboardStepTitleEl) onboardStepTitleEl.textContent = step.title;
      if (onboardStepBodyEl) onboardStepBodyEl.textContent = step.body;
      if (onboardStepCountEl) {
        onboardStepCountEl.textContent = `Step ${onboardIndex + 1} of ${ONBOARD_STEPS.length}`;
      }
      if (onboardProgressEl) {
        onboardProgressEl.style.width = `${Math.round(((onboardIndex + 1) / ONBOARD_STEPS.length) * 100)}%`;
      }
      if (onboardNextBtnEl) {
        onboardNextBtnEl.textContent = onboardIndex === ONBOARD_STEPS.length - 1 ? 'Finish' : 'Next';
      }
      if (onboardBackBtnEl) {
        onboardBackBtnEl.disabled = onboardIndex === 0;
      }
    }

    function nextOnboardStep() {
      if (onboardIndex >= ONBOARD_STEPS.length - 1) {
        toggleOnboard(false);
        return;
      }
      renderOnboardStep(onboardIndex + 1);
    }

    function prevOnboardStep() {
      renderOnboardStep(onboardIndex - 1);
    }

    function effectsEnabled() {
      return localStorage.getItem(EFFECTS_ENABLED_KEY) !== '0';
    }

    function clearThemeEffects() {
      if (fxLayerEl) fxLayerEl.innerHTML = '';
    }

    function spawnEffectNodes(kind, count) {
      if (!fxLayerEl) return;
      for (let i = 0; i < count; i += 1) {
        const node = document.createElement('span');
        node.className = kind;
        node.style.left = `${Math.random() * 100}%`;
        node.style.top = `${Math.random() * 100}%`;
        node.style.animationDelay = `${(Math.random() * 6).toFixed(2)}s`;
        node.style.animationDuration = `${(5 + Math.random() * 8).toFixed(2)}s`;
        fxLayerEl.appendChild(node);
      }
    }

    function applyThemeEffects(theme) {
      clearThemeEffects();
      if (!fxLayerEl) return;
      if (!effectsEnabled()) return;
      const rootStyle = getComputedStyle(document.documentElement);
      const accent = (rootStyle.getPropertyValue('--accent') || '#60a5fa').trim();
      const accent2 = (rootStyle.getPropertyValue('--accent-2') || '#a78bfa').trim();
      fxLayerEl.style.setProperty('--fx-dot', accent2);
      fxLayerEl.style.setProperty('--fx-dot-glow', accent2);
      fxLayerEl.style.setProperty('--fx-comet', accent);
      fxLayerEl.style.setProperty('--fx-comet-glow', accent);
      fxLayerEl.style.setProperty('--fx-spark', accent);
      fxLayerEl.style.setProperty('--fx-spark-glow', accent2);
      const warmThemes = new Set(['solar', 'supernova', 'meteor', 'plasma']);
      const coolThemes = new Set(['stellar', 'nebula', 'lunar', 'cosmos', 'void', 'orbit', 'dark', 'eclipse', 'aurora']);
      if (warmThemes.has(theme)) {
        spawnEffectNodes('fx-comet', 5);
        spawnEffectNodes('fx-spark', 26);
        return;
      }
      if (coolThemes.has(theme)) {
        spawnEffectNodes('fx-dot', 46);
        return;
      }
      spawnEffectNodes('fx-dot', 22);
    }

    function applyTheme(next) {
      activeThemeName = next || 'system';
      if (next === 'system') {
        document.documentElement.removeAttribute('data-theme');
      } else {
        document.documentElement.setAttribute('data-theme', next);
      }
      if (gdimpVideoFrameEl) {
        const videoSrc = gdimpVideoFrameEl.dataset.src || '';
        if (next === 'gdimp') {
          if (gdimpVideoWrapEl) gdimpVideoWrapEl.classList.remove('hidden-by-user');
          if (gdimpVideoFrameEl.src !== videoSrc) gdimpVideoFrameEl.src = videoSrc;
        } else {
          gdimpVideoFrameEl.src = '';
        }
      }
      localStorage.setItem('theme', next);
      if (themeSelect) themeSelect.value = next;
      themeButtons.forEach((btn) => {
        btn.classList.toggle('active', btn.dataset.theme === next);
      });
      applyThemeEffects(activeThemeName);
    }

    function initTheme() {
      const saved = localStorage.getItem('theme') || 'system';
      applyTheme(saved);
      if (themeSelect) {
        themeSelect.addEventListener('change', (event) => applyTheme(event.target.value));
      }
      themeButtons.forEach((btn) => {
        btn.addEventListener('click', () => applyTheme(btn.dataset.theme || 'system'));
      });
    }

    function initEffectsPanel() {
      if (!effectsMenuBtnEl || !effectsPanelEl || !effectsEnabledEl) return;
      effectsEnabledEl.checked = effectsEnabled();
      effectsMenuBtnEl.addEventListener('click', (event) => {
        event.stopPropagation();
        effectsPanelEl.classList.toggle('open');
      });
      effectsEnabledEl.addEventListener('change', () => {
        localStorage.setItem(EFFECTS_ENABLED_KEY, effectsEnabledEl.checked ? '1' : '0');
        applyThemeEffects(activeThemeName);
      });
      document.addEventListener('click', (event) => {
        if (!effectsPanelEl.classList.contains('open')) return;
        if (effectsPanelEl.contains(event.target)) return;
        if (effectsMenuBtnEl.contains(event.target)) return;
        effectsPanelEl.classList.remove('open');
      });
    }

    function initProviderVault() {
      const providerSelect = document.getElementById('vault_provider');
      if (!providerSelect) return;
      providerSelect.addEventListener('change', () => {
        const provider = providerSelect.value || 'openai';
        const defaults = providerDefaults[provider] || providerDefaults.openai || { model: '', base_url: '' };
        const modelEl = document.getElementById('vault_model');
        const baseEl = document.getElementById('vault_base_url');
        if (modelEl && !modelEl.value.trim()) modelEl.value = defaults.model || '';
        if (baseEl && !baseEl.value.trim()) baseEl.value = defaults.base_url || '';
      });
    }

    function initAIProfiles() {
      if (!aiProfileSelectEl) return;
      const provider = document.getElementById('ai_provider').value || '';
      const model = document.getElementById('ai_model').value || '';
      const baseUrl = document.getElementById('ai_base_url').value || '';
      const matched = providerProfiles.find((item) => (
        (item.provider || '') === provider
        && (item.model || '') === model
        && (item.base_url || '') === baseUrl
      ));
      if (matched) {
        aiProfileSelectEl.value = matched.id;
      }
      aiProfileSelectEl.addEventListener('change', () => {
        if (!aiProfileSelectEl.value) return;
        applySelectedProfileToSettings();
      });
    }

    function setHelpMode(mode) {
      if (!helpModeToggleEl) return;
      helpModeToggleEl.querySelectorAll('button[data-mode]').forEach((btn) => {
        btn.classList.toggle('active', btn.dataset.mode === mode);
      });
      localStorage.setItem(HELP_MODE_KEY, mode);
    }

    function initHelpMode() {
      const saved = localStorage.getItem(HELP_MODE_KEY) || 'create';
      setHelpMode(saved);
      if (helpModeToggleEl) {
        helpModeToggleEl.querySelectorAll('button[data-mode]').forEach((btn) => {
          btn.addEventListener('click', () => setHelpMode(btn.dataset.mode));
        });
      }
    }

    function initHelpModel() {
      const saved = localStorage.getItem(HELP_MODEL_KEY);
      if (saved && helpModelEl) helpModelEl.value = saved;
      if (helpModelEl) {
        helpModelEl.addEventListener('change', () => {
          localStorage.setItem(HELP_MODEL_KEY, helpModelEl.value);
        });
      }
    }

    function loadHelpHistory() {
      if (!helpChatLogEl) return;
      helpChatLogEl.innerHTML = '';
      const raw = localStorage.getItem(HELP_HISTORY_KEY);
      const history = raw ? JSON.parse(raw) : [];
      history.forEach((item) => {
        const bubble = document.createElement('div');
        bubble.className = `help-chat-bubble ${item.role}`;
        bubble.textContent = item.content;
        helpChatLogEl.appendChild(bubble);
      });
      helpChatLogEl.scrollTop = helpChatLogEl.scrollHeight;
    }

    function saveHelpHistory(history) {
      localStorage.setItem(HELP_HISTORY_KEY, JSON.stringify(history));
    }

    function appendHelpBubble(role, content) {
      if (!helpChatLogEl) return;
      const bubble = document.createElement('div');
      bubble.className = `help-chat-bubble ${role}`;
      bubble.textContent = content;
      helpChatLogEl.appendChild(bubble);
      helpChatLogEl.scrollTop = helpChatLogEl.scrollHeight;
    }

    function formatHelpError(err) {
      if (!err) return 'Help chat failed.';
      return err.message || String(err);
    }

    async function signInHelpChat() {
      if (!helpStatusHintEl) return;
      if (!window.puter?.auth?.signIn) {
        helpStatusHintEl.textContent = 'Puter.js auth is not available.';
        return;
      }
      try {
        helpStatusHintEl.textContent = 'Requesting Puter auth...';
        await window.puter.auth.signIn();
        helpStatusHintEl.textContent = 'Signed in. Ask a question.';
      } catch (err) {
        helpStatusHintEl.textContent = formatHelpError(err);
      }
    }

    async function sendHelpChat() {
      if (!helpChatInputEl || !helpChatLogEl) return;
      const text = helpChatInputEl.value.trim();
      if (!text) return;
      helpChatInputEl.value = '';
      appendHelpBubble('user', text);

      const raw = localStorage.getItem(HELP_HISTORY_KEY);
      const history = raw ? JSON.parse(raw) : [];
      history.push({ role: 'user', content: text });
      saveHelpHistory(history);

      if (!window.puter?.ai?.chat) {
        if (helpStatusHintEl) helpStatusHintEl.textContent = 'Puter.js not loaded yet.';
        appendHelpBubble('bot', 'Help chat failed. Puter.js not loaded.');
        return;
      }

      try {
        if (helpStatusHintEl) helpStatusHintEl.textContent = 'Thinking...';
        const mode = localStorage.getItem(HELP_MODE_KEY) || 'create';
        const systemPrompt = [
          'You are the embedded help assistant for the OCreator app.',
          'Answer with short, actionable steps.',
          `Focus on the "${mode}" mode.`,
          'The app generates playable OC personas; avoid bot personality guidance.',
        ].join(' ');
        const model = helpModelEl ? helpModelEl.value : 'grok-4-fast';
        const result = await window.puter.ai.chat({
          model,
          messages: [
            { role: 'system', content: systemPrompt },
            ...history,
          ],
        });
        const reply = result?.message?.content || result?.message || 'Help chat failed.';
        appendHelpBubble('bot', reply);
        history.push({ role: 'assistant', content: reply });
        saveHelpHistory(history);
        if (helpStatusHintEl) helpStatusHintEl.textContent = 'Ready.';
      } catch (err) {
        const message = formatHelpError(err);
        appendHelpBubble('bot', message);
        if (helpStatusHintEl) helpStatusHintEl.textContent = message;
      }
    }

    function clearHelpChat() {
      localStorage.removeItem(HELP_HISTORY_KEY);
      if (helpChatLogEl) helpChatLogEl.innerHTML = '';
      if (helpStatusHintEl) helpStatusHintEl.textContent = 'Cleared.';
    }

    document.querySelectorAll('#mode_toggle button').forEach((btn) => {
      btn.addEventListener('click', () => setMode(btn.dataset.mode));
    });

    if (helpChatInputEl) {
      helpChatInputEl.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
          event.preventDefault();
          sendHelpChat();
        }
      });
    }

    if (confirmOkBtnEl) confirmOkBtnEl.addEventListener('click', () => resolveConfirm(true));
    if (confirmCancelBtnEl) confirmCancelBtnEl.addEventListener('click', () => resolveConfirm(false));
    if (confirmModalEl) {
      confirmModalEl.addEventListener('click', (event) => {
        if (event.target === confirmModalEl) resolveConfirm(false);
      });
    }
    if (gdimpVideoCloseBtnEl) {
      gdimpVideoCloseBtnEl.addEventListener('click', () => {
        if (gdimpVideoFrameEl) gdimpVideoFrameEl.src = '';
        if (gdimpVideoWrapEl) gdimpVideoWrapEl.classList.add('hidden-by-user');
      });
    }

    function startAutosave() {
      setInterval(() => {
        if (document.hidden) return;
        saveOC(true);
      }, 20000);
    }

    initMode();
    initAITabs();
    initMobileActions();
    initMobilePanels();
    initTemplate();
    initTheme();
    initEffectsPanel();
    initAIProfiles();
    initProviderVault();
    initHelpMode();
    initHelpModel();
    initSecurityPanel();
    initOCMenus();
    refreshSecurityStatus();
    loadHelpHistory();
    renderOnboardStep(0);
    startAutosave();
  </script>
</body>
</html>
"""


def main() -> None:
    ensure_dirs()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8934")), debug=False)
    ## this is just so its not on exactly 8000 :)


if __name__ == "__main__":
    main()

