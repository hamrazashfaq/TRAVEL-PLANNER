# TRAVEL-PLANNER
# 🧭 AI Travel Itinerary Planner

A Streamlit app that creates a personalised travel itinerary with Google's Gemini Flash model. You enter a destination, budget, dates, interests and trip length, and it returns:

- A day-by-day plan with attractions and activities
- Transportation options (getting there and getting around)
- Accommodation suggestions (types and areas)
- Estimated expenses with a budget check
- Travel tips and a "verify before you book" checklist
- Downloads as Markdown or JSON

## Honest-by-design

The AI has **no live access** to prices, opening hours, visa rules or events. The app is built so it doesn't pretend otherwise:

| Risk | How the app handles it |
|---|---|
| Made-up "current" prices | Every price is a **range** labelled as an *estimate*, with a confidence level (low/medium/high) |
| Bad arithmetic from the AI | Totals are **recalculated in Python** from the AI's line items |
| Invented hotels, URLs, phone numbers | The prompt forbids them. Only accommodation *types and areas* are suggested |
| Stale visa / safety / schedule claims | The AI is told to point you to official sources instead of stating rules |
| Unrealistic budgets | The AI must say so plainly, and the app compares the estimate against your budget |
| Wrong dates | Dates are assigned by the app, not the AI |
| Prompt injection via input fields | User input is passed as JSON data and the AI is told to treat it as data only |

Always verify prices, visas, opening hours and transport before booking.

## Project structure

```
travel-planner/
├── app.py            # The whole Streamlit app
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

## Run locally

**Requirements:** Python 3.10+ and a free Gemini API key from <https://aistudio.google.com/apikey>.

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key (pick ONE option)
```

**Option A, secrets file (recommended).** Create `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your-key-here"
```

**Option B, environment variable.**

```bash
export GEMINI_API_KEY="your-key-here"     # macOS / Linux
set GEMINI_API_KEY=your-key-here          # Windows (cmd)
$env:GEMINI_API_KEY="your-key-here"       # Windows (PowerShell)
```

**Option C, paste it in the sidebar** when the app is running.

```bash
# 4. Run
streamlit run app.py
```

The app opens at <http://localhost:8501>.

## Configuration

| Setting | Where | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | secrets / env var | Your Gemini API key |
| `GEMINI_MODEL` | secrets / env var (optional) | Override the model, e.g. `gemini-3.8-flash`. The default is `gemini-3.8-flash` with `gemini-3.6-flash` as a fallback |

Gemini model names change often. If you see "None of the configured Gemini models are available", set `GEMINI_MODEL` to a current name from <https://ai.google.dev/gemini-api/docs/models>.

## Push to GitHub

1. **Create a `.gitignore`** so your API key is never uploaded. Do this *before* your first commit:

   ```bash
   printf "venv/\n__pycache__/\n.streamlit/secrets.toml\n.env\n" > .gitignore
   ```

2. **Create an empty repository** on <https://github.com/new>. Name it e.g. `travel-planner`. Don't add a README or .gitignore there, because you already have them.

3. **Initialise and push** from your project folder:

   ```bash
   git init
   git add .
   git status                       # check that secrets.toml is NOT listed
   git commit -m "Initial commit: AI travel planner"
   git branch -M main
   git remote add origin https://github.com/<your-username>/travel-planner.git
   git push -u origin main
   ```

   When asked for a password, use a **Personal Access Token** (GitHub → Settings → Developer settings → Personal access tokens), not your account password. Alternatively, use SSH or the GitHub CLI (`gh auth login`).

> ⚠️ If you ever commit your API key by accident, **revoke it** in Google AI Studio and create a new one. Deleting it from a later commit is not enough, because it stays in the git history.

## Deploy on Streamlit Community Cloud

1. Go to <https://share.streamlit.io> and sign in with your GitHub account. Authorise access to your repositories when asked.
2. Click **Create app** → **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `<your-username>/travel-planner`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Advanced settings** and:
   - Choose a Python version (3.11 or 3.12 works well).
   - Paste your secret into the **Secrets** box (TOML format):

     ```toml
     GEMINI_API_KEY = "your-key-here"
     ```
5. Click **Deploy**. The first build takes a few minutes while it installs `requirements.txt`.
6. Your app gets a public URL like `https://<your-app-name>.streamlit.app`.

**Updating:** every `git push` to `main` redeploys automatically. To change secrets later, open the app → **⋮ → Settings → Secrets**.

### Protect your quota on a public app

Anyone with the link can trigger requests using *your* key. To limit the risk:

- Set quotas and billing alerts for your key in Google AI Studio or Google Cloud.
- Or leave `GEMINI_API_KEY` out of the secrets, so each visitor must paste their own key in the sidebar.
- Or deploy as a private app and invite only specific people (app **Settings → Sharing**).

## Troubleshooting

| Problem | Fix |
|---|---|
| "The Gemini API key was rejected" | Check the key, and that the Gemini API is enabled for it |
| "rate limit or quota reached" | Wait a minute, or check your quota in Google AI Studio |
| "None of the configured Gemini models are available" | Set `GEMINI_MODEL` to a current model name |
| "unexpected format" | Click Generate again; the app already retries once automatically |
| App crashes on Streamlit Cloud | Open **Manage app → Logs** and check `requirements.txt` and the secrets format |

## Ideas for extending it

- Add Google Search grounding so the model can cite fresh sources
- Pull real exchange rates and weather from public APIs
- Export a PDF, or show attractions on a map
- Add user accounts and saved trips

## Tech stack

[Streamlit](https://streamlit.io) · [Google Gen AI SDK](https://pypi.org/project/google-genai/) · Gemini Flash

> Tested with Streamlit 1.66 and google-genai 2.32.
