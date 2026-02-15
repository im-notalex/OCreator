import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from flask import Flask, jsonify, redirect, render_template_string, request, url_for

APP_TITLE = "OCreator"
DATA_DIR = Path(__file__).resolve().parent / "data"
DB_PATH = DATA_DIR / "ocs.json"
SETTINGS_PATH = DATA_DIR / "settings.json"
PROVIDERS_PATH = DATA_DIR / "providers.json"

PROVIDERS = [
    {"id": "openai", "label": "OpenAI"},
    {"id": "gemini", "label": "Gemini"},
    {"id": "anthropic", "label": "Anthropic"},
    {"id": "grok", "label": "Grok"},
    {"id": "openrouter", "label": "OpenRouter"},
    {"id": "openai_compatible", "label": "OpenAI Compatible"},
]

PROVIDER_DEFAULTS = {
    "openai": {"model": "gpt-4o-mini", "base_url": "https://api.openai.com/v1"},
    "gemini": {"model": "gemini-1.5-pro", "base_url": "https://generativelanguage.googleapis.com/v1beta"},
    "anthropic": {"model": "claude-3-5-sonnet-latest", "base_url": "https://api.anthropic.com/v1"},
    "grok": {"model": "grok-2-latest", "base_url": "https://api.x.ai/v1"},
    "openrouter": {"model": "openai/gpt-4o-mini", "base_url": "https://openrouter.ai/api/v1"},
    "openai_compatible": {"model": "your-model", "base_url": "http://localhost:1234/v1"},
}

DEFAULT_SETTINGS = {
    "provider": "openai",
    "api_key": "",
    "model": PROVIDER_DEFAULTS["openai"]["model"],
    "base_url": PROVIDER_DEFAULTS["openai"]["base_url"],
    "temperature": 0.7,
    "max_tokens": 1200,
}

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


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_settings() -> Dict[str, Any]:
    settings = load_json(SETTINGS_PATH, DEFAULT_SETTINGS.copy())
    merged = DEFAULT_SETTINGS.copy()
    if isinstance(settings, dict):
        merged.update({k: v for k, v in settings.items() if v is not None})
    try:
        merged["temperature"] = float(merged.get("temperature", DEFAULT_SETTINGS["temperature"]))
    except Exception:
        merged["temperature"] = DEFAULT_SETTINGS["temperature"]
    try:
        merged["max_tokens"] = int(merged.get("max_tokens", DEFAULT_SETTINGS["max_tokens"]))
    except Exception:
        merged["max_tokens"] = DEFAULT_SETTINGS["max_tokens"]
    provider = merged.get("provider", "openai")
    defaults = PROVIDER_DEFAULTS.get(provider, PROVIDER_DEFAULTS["openai"])
    if not merged.get("model"):
        merged["model"] = defaults["model"]
    if not merged.get("base_url"):
        merged["base_url"] = defaults["base_url"]
    return merged


def save_settings(settings: Dict[str, Any]) -> None:
    save_json(SETTINGS_PATH, settings)


def load_provider_profiles() -> List[Dict[str, Any]]:
    raw = load_json(PROVIDERS_PATH, {"providers": []})
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
    save_json(PROVIDERS_PATH, {"providers": profiles})


def load_db() -> Dict[str, Any]:
    db = load_json(DB_PATH, {"ocs": []})
    if "ocs" not in db or not isinstance(db["ocs"], list):
        db["ocs"] = []
    return db


def save_db(db: Dict[str, Any]) -> None:
    save_json(DB_PATH, db)


def ensure_oc_defaults(oc: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in DEFAULT_OC.items():
        oc.setdefault(key, value)
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
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join([p.get("text", "") for p in parts if isinstance(p, dict)])


def call_llm(settings: Dict[str, Any], prompt: str) -> str:
    provider = settings.get("provider", "openai")
    if provider in {"openai", "openrouter", "grok", "openai_compatible"}:
        return call_openai_like(settings, prompt)
    if provider == "anthropic":
        return call_anthropic(settings, prompt)
    if provider == "gemini":
        return call_gemini(settings, prompt)
    raise RuntimeError("Unknown provider.")


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
    )


@app.route("/bot/<oc_id>")
def legacy_bot_route(oc_id: str):
    return redirect(url_for("edit_oc", oc_id=oc_id))


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
    oc["result_text"] = response
    if not oc.get("name") or oc["name"] == "Untitled OC":
        first_line = response.splitlines()[0].strip()
        if first_line.lower().startswith("name:"):
            candidate = first_line.split(":", 1)[-1].strip()
            if candidate:
                oc["name"] = candidate
    oc["updated_at"] = now_iso()
    save_db(db)
    return jsonify({"result": response, "name": oc.get("name", "")})


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
    settings.update({k: v for k, v in payload.items() if k in settings})
    save_settings(settings)
    return jsonify({"ok": True})


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
    .help-btn {
      width: 36px;
      height: 36px;
      border-radius: 12px;
      font-weight: 700;
      padding: 0;
    }
    .layout {
      display: grid;
      grid-template-columns: 300px 1fr 290px 320px;
      gap: 20px;
      padding: 24px;
      position: relative;
      z-index: 2;
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
      overflow: auto;
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
      content: "Pinned • ";
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
    }
    @media (max-width: 1200px) {
      .layout { grid-template-columns: 280px 1fr; }
      .layout .panel.library-panel { order: 3; }
      .layout .panel.provider-panel { order: 4; grid-column: 1 / -1; }
    }
    @media (max-width: 980px) {
      .layout { grid-template-columns: 1fr; }
      .help-grid { grid-template-columns: 1fr; }
      .layout .panel.provider-panel { grid-column: auto; }
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
  <header>
    <div>
      <div class="brand">{{ app_title }}</div>
      <div class="sub">Build playable OC personas from scratch or refine an existing OC text block.</div>
    </div>
    <div class="row">
      <div class="theme-control">
        <div class="theme-picker" id="theme_picker" role="radiogroup" aria-label="Theme">
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
        </select>
        <button class="effects-menu-btn" id="effects_menu_btn" type="button">Effects</button>
        <div class="effects-panel" id="effects_panel">
          <div class="effects-line">
            <span>Theme effects</span>
            <input id="effects_enabled" type="checkbox" checked />
          </div>
          <div class="hint" style="margin-top: 8px;">Warm themes use comets and sparks. Cool themes use stars.</div>
        </div>
      </div>
      <button class="btn ghost" type="button" onclick="toggleOnboard(true)">Guide</button>
      <button class="btn ghost help-btn" type="button" onclick="toggleHelp(true)" title="Help">?</button>
      <button class="btn ghost" type="button" onclick="saveOC()">Save</button>
      <form method="post" action="{{ url_for('create_oc_route') }}">
        <button class="btn secondary" type="submit">New OC</button>
      </form>
    </div>
  </header>

  <div class="layout">
    <aside class="panel">
      <h3>AI Settings</h3>
      <div class="grid">
        <div>
          <label>Provider</label>
          <select id="ai_provider">
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
          <input id="ai_tokens" type="number" min="128" max="4096" step="64" value="{{ settings.max_tokens }}" />
        </div>
        <div>
          <button class="btn ghost" type="button" onclick="saveSettings()">Save AI Settings</button>
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

    <main class="panel">
      <div class="grid">
        <div>
          <label>OC Name</label>
          <input id="oc_name" value="{{ oc.name }}" placeholder="OC name" />
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
        <button class="btn" type="button" onclick="generateOC()">Generate</button>
        <button class="btn ghost" type="button" onclick="applyResult()">Apply Result</button>
      </div>

      <div style="margin-top: 12px;">
        <label>Result (single text persona)</label>
        <textarea id="oc_result" class="result" readonly>{{ oc.result_text }}</textarea>
      </div>
    </main>

    <aside class="panel library-panel">
      <h3>OC Library</h3>
      <div class="list" style="margin-bottom: 16px;">
        {% for item in ocs %}
        <div class="oc-card {% if item.id == oc.id %}active{% endif %} {% if item.pinned %}pinned{% endif %}" data-oc-id="{{ item.id }}">
          <a class="oc-link" href="{{ url_for('edit_oc', oc_id=item.id) }}">
            <div class="title">{{ item.name or 'Untitled OC' }}</div>
            <div class="meta">{{ item.updated_at or 'New' }}</div>
          </a>
          <div class="oc-menu-wrap">
            <button class="oc-menu-btn" type="button" title="OC actions" data-menu-trigger aria-label="Open OC menu">☰</button>
            <div class="oc-menu" data-menu>
              <button type="button" data-action="duplicate">Duplicate</button>
              <button type="button" data-action="pin">{% if item.pinned %}Unpin{% else %}Pin{% endif %}</button>
              <button type="button" data-action="delete">Delete</button>
            </div>
          </div>
        </div>
        {% endfor %}
      </div>
    </aside>

    <aside class="panel provider-panel">
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
          <input id="vault_key" type="password" placeholder="Stored in data/providers.json" />
        </div>
        <div>
          <label>Notes</label>
          <input id="vault_notes" placeholder="Optional notes" />
        </div>
      </div>
      <div class="row" style="margin-top: 10px;">
        <button class="btn ghost" type="button" onclick="saveProviderProfile()">Save Profile</button>
        <button class="btn ghost" type="button" onclick="clearProviderForm()">Clear</button>
      </div>
      <div class="hint" id="provider_status" style="margin-top: 8px;">Profiles are saved to data/providers.json.</div>
      <div class="provider-list" id="provider_list">
        {% for profile in provider_profiles %}
        <div class="provider-card" data-profile-id="{{ profile.id }}">
          <div>
            <strong>{{ profile.name or 'Unnamed Provider' }}</strong>
            <div class="meta">{{ profile.provider }}{% if profile.model %} • {{ profile.model }}{% endif %}</div>
          </div>
          <div class="provider-row">
            <button class="btn ghost" type="button" onclick="useProviderProfile('{{ profile.id }}')">Use</button>
            <button class="btn ghost" type="button" onclick="editProviderProfile('{{ profile.id }}')">Edit</button>
            <button class="btn ghost" type="button" onclick="deleteProviderProfile('{{ profile.id }}')">Delete</button>
          </div>
        </div>
        {% endfor %}
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

  <script src="https://js.puter.com/v2/"></script>
  <script>
    const OC_ID = "{{ oc.id }}";
    const modeToggle = document.getElementById('mode_toggle');
    const templateMode = document.getElementById('template_mode');
    const statusProgressEl = document.getElementById('status_progress');
    const statusSaveEl = document.getElementById('status_save');
    const helpModalEl = document.getElementById('help_modal');
    const helpModeToggleEl = document.getElementById('help_mode_toggle');
    const helpModelEl = document.getElementById('help_model');
    const helpChatLogEl = document.getElementById('help_chat_log');
    const helpChatInputEl = document.getElementById('help_chat_input');
    const helpStatusHintEl = document.getElementById('help_status_hint');
    const themeSelect = document.getElementById('theme_select');
    const themeButtons = document.querySelectorAll('.theme-option');
    const fxLayerEl = document.getElementById('fx_layer');
    const effectsMenuBtnEl = document.getElementById('effects_menu_btn');
    const effectsPanelEl = document.getElementById('effects_panel');
    const effectsEnabledEl = document.getElementById('effects_enabled');
    const onboardModalEl = document.getElementById('onboard_modal');
    const onboardProgressEl = document.getElementById('onboard_progress');
    const onboardStepTitleEl = document.getElementById('onboard_step_title');
    const onboardStepBodyEl = document.getElementById('onboard_step_body');
    const onboardStepCountEl = document.getElementById('onboard_step_count');
    const onboardNextBtnEl = document.getElementById('onboard_next_btn');
    const onboardBackBtnEl = document.getElementById('onboard_back_btn');
    const providerStatusEl = document.getElementById('provider_status');
    const providerDefaults = {{ provider_defaults|tojson }};
    const providerProfiles = {{ provider_profiles|tojson }};

    const HELP_HISTORY_KEY = 'ocreator_help_history';
    const HELP_MODE_KEY = 'ocreator_help_mode';
    const HELP_MODEL_KEY = 'ocreator_help_model';
    const EFFECTS_ENABLED_KEY = 'ocreator_effects_enabled';

    let activeThemeName = 'system';

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

    function initTemplate() {
      templateMode.value = "{{ oc.template_mode }}";
    }

    function collectPayload() {
      return {
        name: document.getElementById('oc_name').value,
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
      const payload = {
        provider: document.getElementById('ai_provider').value,
        model: document.getElementById('ai_model').value,
        api_key: document.getElementById('ai_key').value,
        base_url: document.getElementById('ai_base_url').value,
        temperature: parseFloat(document.getElementById('ai_temp').value || '0.7'),
        max_tokens: parseInt(document.getElementById('ai_tokens').value || '1200', 10),
      };
      const res = await fetch('/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        setStatus('Settings save failed.');
        return;
      }
      setStatus('Settings saved.');
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
      document.getElementById('ai_provider').value = profile.provider || 'openai';
      document.getElementById('ai_model').value = profile.model || '';
      document.getElementById('ai_base_url').value = profile.base_url || '';
      document.getElementById('ai_key').value = profile.api_key || '';
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
      if (!window.confirm(`Delete provider profile "${profile.name}"?`)) return;
      const res = await fetch(`/providers/${profileId}`, { method: 'DELETE' });
      const data = await res.json();
      if (!res.ok) {
        setProviderStatus(data.error || 'Failed to delete provider profile.');
        return;
      }
      setProviderStatus('Provider profile deleted.');
      window.location.reload();
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
      if (!window.confirm('Delete this OC?')) return;
      const res = await fetch(`/oc/${ocId}/delete`, { method: 'POST' });
      const data = await res.json();
      if (!res.ok) {
        setStatus(data.error || 'Delete failed.');
        return;
      }
      window.location.href = `/oc/${data.next_oc_id}`;
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
          if (action === 'delete') deleteOC(ocId);
        });
      });
      document.addEventListener('click', () => closeOCMenus());
      document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape') closeOCMenus();
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
      setStatus('Done.');
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

    function startAutosave() {
      setInterval(() => {
        if (document.hidden) return;
        saveOC(true);
      }, 20000);
    }

    initMode();
    initTemplate();
    initTheme();
    initEffectsPanel();
    initProviderVault();
    initHelpMode();
    initHelpModel();
    initOCMenus();
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
