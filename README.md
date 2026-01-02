# OCreator

OCreator is a local-first OC persona builder. Create a playable character from a prompt, clean up an existing persona, or enhance a draft — all as a single text output for platforms like Chub or Janitor.

## Quick start (main QoL)
1) Double-click `quickstart.bat`
2) The script installs requirements if needed and launches the app

Open `http://localhost:8000` in your browser.

## Features
- Create / Modify / Enhance modes
- Single text output (no JSON)
- Default template or custom template
- OpenAI, Anthropic, Gemini, Grok, OpenRouter, or OpenAI-compatible endpoints
- Local storage in `data/`

## Templates
Default template focuses on identity facts and avoids personality:
```
Name:
Age:
Species:
Appearance:
Origin:
Occupation:
Skills:
Abilities:
Gear:
Relationships:
Goals:
Limits/Boundaries:
Notes:
```

## Requirements
- Python 3.10+
- `flask` and `requests` via `requirements.txt`

## License
See `LICENSE`.
