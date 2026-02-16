# ***OCreator***

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-App-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-22c55e.svg)](LICENSE)
![Local First](https://img.shields.io/badge/Storage-Local%20First-0ea5e9)

**OCreator is a local-first OC persona builder. Create a playable character from a prompt, clean up an existing persona, or enhance a draft, all as a single text output for platforms like Chub or Janitor.**

![main](https://raw.githubusercontent.com/im-notalex/storage-n-stuff/refs/heads/main/OCreator/Main.png)

## ***Quick start (main QoL)***
1) Double-click `quickstart.bat`
2) The script installs requirements if needed and launches the app

Open `http://localhost:8000` in your browser.

## ***Features***
- Create / Modify / Enhance modes
- Single text output (no JSON)
- Default template or custom template
- OpenAI, Anthropic, Gemini, Grok, OpenRouter, or OpenAI-compatible endpoints
- Local storage in `data/`

## ***Templates***
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

## ***Requirements***
- Python 3.10+
- `flask` and `requests` via `requirements.txt`

## ***Security***
- API keys are stored in plaintext in `data/settings.json` and `data/providers.json`.
- Keep `data/` out of git and treat it as sensitive local data.

## License
See `LICENSE`.

## ***Why AGPL?***
**This project uses AGPL-3.0 to ensure that if anyone hosts a modified 
version as a public service, users have access to that modified source 
code. Since these tools handle personal creative content, transparency 
is important. Run it locally and modify freely - the license only 
requires sharing if you make it a public web service. Keep it local, or keep it open.**

## ***Notes***

***the readme was done by AI, i am a little chud and didnt want to write it myself as you can see by emdashes :)***
***also to clarify cause i dont say it a lot***
***i build the shell of the project, Codex and so on decorates it so its visually appealing (cause i dont know a lot of HTML) :]***
***and i forgot to mention, if you didnt see the description of the repo this and botsmyth are freelance comissions cause i wanted to mess with python in my freetime.***
***a little note as of now, the Why AGPL is by me. i want to be transparent on my code, and i want everyone else to be too.***
