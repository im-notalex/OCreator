import subprocess
import json
import os
import secrets
import shutil
import socket
import sys
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
    render_template,
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


def print_non_windows_ipv4_summary(port: int) -> None:
    host = socket.gethostname()
    ipv4_candidates: List[str] = []

    # Best-effort primary LAN IP (works on most Linux/macOS/Android environments).
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            primary_ip = sock.getsockname()[0]
            if primary_ip and not primary_ip.startswith("127."):
                ipv4_candidates.append(primary_ip)
    except Exception:
        pass

    # Add any extra IPv4 addresses resolved from hostname.
    try:
        for entry in socket.getaddrinfo(host, None, family=socket.AF_INET, type=socket.SOCK_STREAM):
            ip = str(entry[4][0])
            if ip and not ip.startswith("127.") and ip not in ipv4_candidates:
                ipv4_candidates.append(ip)
    except Exception:
        pass

    is_android = bool(os.environ.get("ANDROID_ROOT") or os.environ.get("ANDROID_DATA"))
    platform_label = "Android" if is_android else ("macOS" if sys.platform == "darwin" else "Linux/Unix")

    print("--- Network Summary ---")
    print(f"Platform: {platform_label}")
    if not ipv4_candidates:
        print("IPv4: could not detect a LAN address (check network/interface state).")
    else:
        for ip in ipv4_candidates:
            print(f"IPv4: {ip}")
            print(f"Local URL: http://{ip}:{port}")
    print("-----------------------")


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

    return render_template("secure_login.html",
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
            "current_user_id": str(current.get("id", "") if current else ""),
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
    return render_template("app.html",
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
        current_user_id=str(current_user.get("id", "") if current_user else ""),
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

def main() -> None:
    ensure_dirs()
    port = int(os.environ.get("PORT", "8934"))
    if os.name == "nt":
        get_ip_script = Path(__file__).resolve().parent / "get_ip.bat"
        if get_ip_script.exists():
            print("Processing complete. Fetching network info...")
            subprocess.run([str(get_ip_script)], check=False)
    else:
        print_non_windows_ipv4_summary(port)
    app.run(host="0.0.0.0", port=port, debug=False)
    ## this is just so its not on exactly 8000 :)


if __name__ == "__main__":
    main()
