"""
Senegal French Semantic Shift Finder
뉴스 URL -> 본문 추출 -> 표제어 추출 -> 사전 정의 조회 -> 문맥/정의 비교 -> 의미 차이 후보 표시
"""
import os, re, json, hashlib, collections
import requests
import pandas as pd
import streamlit as st

CACHE_DIR = ".cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# 연구 방법론에서 쓴 '문화 표지어' (필요하면 자유롭게 추가)
CULTURAL_MARKERS = {
    "teranga", "village", "villageois", "villageoise", "communauté", "communautaire",
    "famille", "familial", "tontine", "collectif", "collective", "entraide",
    "ndigël", "kersa", "sutura", "dahira", "daara", "marabout", "griot", "chef",
    "sage", "sagesse", "doyen", "ancien", "tradition", "traditionnel", "cérémonie",
    "baptême", "mariage", "cotisation", "financière", "survie", "dette", "devoir",
}


# ---------- 1. 기사 수집 ----------
@st.cache_data(show_spinner=False)
def fetch_article(url):
    import trafilatura
    html = trafilatura.fetch_url(url)
    if not html:
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        html = r.text
    text = trafilatura.extract(html, include_comments=False) or ""
    meta = trafilatura.extract_metadata(html)
    return {"url": url, "title": getattr(meta, "title", "") or "",
            "date": getattr(meta, "date", "") or "", "text": text}


# ---------- 2. NLP ----------
@st.cache_resource(show_spinner=False)
def load_nlp():
    import spacy
    try:
        return spacy.load("fr_core_news_md")
    except OSError:
        from spacy.cli import download
        download("fr_core_news_md")
        return spacy.load("fr_core_news_md")


@st.cache_resource(show_spinner=False)
def load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def analyze(articles, window=5):
    """lemma -> {contexts, collocates, articles}"""
    nlp = load_nlp()
    data = collections.defaultdict(lambda: {"contexts": [], "colloc": collections.Counter(), "urls": set()})
    for art in articles:
        doc = nlp(art["text"][:100000])
        for sent in doc.sents:
            toks = [t for t in sent if not t.is_punct and not t.is_space]
            for i, t in enumerate(toks):
                if t.pos_ not in ("NOUN", "ADJ", "VERB") or t.is_stop or len(t.lemma_) < 4:
                    continue
                lem = t.lemma_.lower()
                d = data[lem]
                d["contexts"].append(sent.text.strip())
                d["urls"].add(art["url"])
                for n in toks[max(0, i - window): i + window + 1]:
                    if n.i != t.i and not n.is_stop and len(n.lemma_) > 3:
                        d["colloc"][n.lemma_.lower()] += 1
    return data


# ---------- 3. 사전 (Wiktionnaire REST API) ----------
def lookup_definitions(word):
    path = os.path.join(CACHE_DIR, hashlib.md5(word.encode()).hexdigest() + ".json")
    if os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    defs = []
    try:
        r = requests.get(f"https://fr.wiktionary.org/api/rest_v1/page/definition/{word}",
                         timeout=15, headers={"User-Agent": "semantic-shift-research/1.0"})
        if r.status_code == 200:
            for entry in r.json().get("fr", []):
                for d in entry.get("definitions", []):
                    txt = re.sub(r"<[^>]+>", "", d.get("definition", "")).strip()
                    if txt:
                        defs.append(f"({entry.get('partOfSpeech','')}) {txt}")
    except Exception:
        pass
    json.dump(defs, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    return defs


# ---------- 4. 의미 차이 점수 ----------
def score_word(word, info, defs, model):
    from sentence_transformers import util
    ctx = info["contexts"][:30]
    if not defs:
        return {"fit": None, "marker_hits": 0}
    d_emb = model.encode([f"{word}: {d}" for d in defs], convert_to_tensor=True)
    c_emb = model.encode(ctx, convert_to_tensor=True)
    sims = util.cos_sim(c_emb, d_emb).max(dim=1).values  # 문맥별 가장 가까운 정의와의 유사도
    fit = float(sims.mean())
    markers = sum(c for w, c in info["colloc"].items() if w in CULTURAL_MARKERS)
    return {"fit": fit, "marker_hits": markers}


def explain_with_claude(word, defs, contexts, collocs):
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        try:
            key = st.secrets.get("ANTHROPIC_API_KEY")
        except Exception:
            key = None
    if not key:
        return None
    import anthropic
    client = anthropic.Anthropic(api_key=key)
    prompt = (
        f"French word: {word}\nStandard dictionary definitions:\n- " + "\n- ".join(defs[:5]) +
        "\n\nSentences from Senegalese news:\n- " + "\n- ".join(contexts[:6]) +
        f"\n\nFrequent collocates: {', '.join(collocs)}\n\n"
        "In 3-4 sentences (English), state whether the word seems to carry a meaning in Senegal "
        "different from the dictionary sense (e.g., via Wolof culture). Quote evidence from the sentences. "
        "If there is no clear shift, say so. Do not speculate beyond the evidence."
    )
    msg = client.messages.create(model="claude-sonnet-5-5", max_tokens=400,
                                 messages=[{"role": "user", "content": prompt}])
    return msg.content[0].text


# ---------- UI ----------
st.set_page_config(page_title="Semantic Shift Finder", layout="wide")
st.title("세네갈 프랑스어 의미 변화 탐지기")

urls_text = st.text_area("세네갈 뉴스 URL (한 줄에 하나: Le Soleil, Seneweb, APS 등)", height=140)
c1, c2, c3 = st.columns(3)
threshold = c1.slider("의미 일치도 임계값 (낮을수록 엄격)", 0.2, 0.8, 0.45, 0.01)
min_freq = c2.number_input("최소 출현 횟수", 1, 20, 2)
targets = c3.text_input("특정 단어만 보기 (쉼표 구분, 예: hospitalité,solidarité,aîné)")
use_llm = st.checkbox("Claude로 의미 차이 설명 생성 (ANTHROPIC_API_KEY 필요)")

if st.button("분석 시작") and urls_text.strip():
    urls = [u.strip() for u in urls_text.splitlines() if u.strip()]
    with st.spinner("기사 수집 중..."):
        arts = [a for a in (fetch_article(u) for u in urls) if a["text"]]
    st.write(f"본문 추출 성공: {len(arts)} / {len(urls)}")
    if not arts:
        st.stop()

    with st.spinner("형태소 분석 중..."):
        data = analyze(arts)
    want = {w.strip().lower() for w in targets.split(",") if w.strip()}
    cands = [w for w, d in data.items() if len(d["contexts"]) >= min_freq and (not want or w in want)]

    model = load_embedder()
    rows, details = [], {}
    prog = st.progress(0.0)
    for i, w in enumerate(cands):
        defs = lookup_definitions(w)
        sc = score_word(w, data[w], defs, model)
        if sc["fit"] is None:
            status = "사전 항목 없음 (차용어/신조어 후보)"
        elif sc["fit"] < threshold or sc["marker_hits"] >= 3:
            status = "의미 차이 후보"
        else:
            status = "사전 의미와 일치"
        rows.append({"단어": w, "빈도": len(data[w]["contexts"]),
                     "정의-문맥 일치도": None if sc["fit"] is None else round(sc["fit"], 3),
                     "문화 표지어 공기 수": sc["marker_hits"], "판정": status})
        details[w] = defs
        prog.progress((i + 1) / len(cands))

    df = pd.DataFrame(rows)
    flagged = df[df["판정"] != "사전 의미와 일치"].sort_values("정의-문맥 일치도", na_position="first")
    st.subheader(f"의미 차이 후보 {len(flagged)}개 / 전체 {len(df)}개")
    st.dataframe(flagged, use_container_width=True)
    st.download_button("CSV 저장", flagged.to_csv(index=False).encode("utf-8-sig"), "shift_candidates.csv")

    for _, r in flagged.head(30).iterrows():
        w = r["단어"]
        with st.expander(f"{w}  |  {r['판정']}"):
            st.markdown("**사전 정의**")
            for d in details[w][:5] or ["(없음)"]:
                st.write("-", d)
            st.markdown("**실제 사용 문장**")
            for s in data[w]["contexts"][:5]:
                st.write("-", s)
            top = [c for c, _ in data[w]["colloc"].most_common(10)]
            st.markdown("**주요 공기어(collocates):** " + ", ".join(top))
            if use_llm:
                exp = explain_with_claude(w, details[w], data[w]["contexts"], top)
                st.info(exp or "ANTHROPIC_API_KEY가 설정되지 않았어요.")
