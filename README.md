# Senegal French Semantic Shift Finder

Paste Senegalese news URLs (Le Soleil, Seneweb, APS) -> the tool extracts words, looks up
standard French definitions (Wiktionnaire), and flags words whose real usage differs from the dictionary sense.

## Run on GitHub Codespaces (recommended)
1. Push this folder to a GitHub repo.
2. Click **Code -> Codespaces -> Create codespace on main**.
3. Wait for setup (about 3-5 min). The app starts automatically; open port 8501 if the browser tab doesn't pop up.
4. (Optional) For Claude explanations: repo **Settings -> Secrets and variables -> Codespaces -> New secret**, name `ANTHROPIC_API_KEY`.

## Run on Streamlit Community Cloud (free, public link)
1. Push to GitHub, then go to share.streamlit.io -> New app -> select repo, main file `app.py`.
2. (Optional) App settings -> Secrets: `ANTHROPIC_API_KEY = "sk-..."`.
Note: free tier has ~1 GB RAM; the embedding model may be tight. If it crashes, use Codespaces.

## Run locally
    pip install -r requirements.txt
    streamlit run app.py
