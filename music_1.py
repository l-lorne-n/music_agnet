# app.py —— API易 版本
import os, json
import streamlit as st
from openai import OpenAI
from ddgs import DDGS
from rapidfuzz import fuzz

# ====== 配置区 ======
# 优先用系统环境变量 OPENAI_API_KEY；如果没有，就用下面这个占位符（请替换成你的密钥）
APIYI_KEY = os.getenv("OPENAI_API_KEY") or "sk-EbAhbINv19eVTaX4CeAb989924A540A1952611Ec45867b81"
APIYI_BASE_URL = "https://api.apiyi.com/v1"
MODEL = "gpt-5"   # 你的代理可用的模型名；不通就换成 gpt-4o / gpt-4o-mini 试试

client = OpenAI(api_key=APIYI_KEY, base_url=APIYI_BASE_URL)

# ====== 核心函数 ======
PROMPT_SYSTEM = """
你是一个严谨的音乐考据型助手。用户会提供“歌曲名 - 艺术家”，请仅输出 JSON，字段：
- title, artist
- tags: 5-12 个风格/子风格标签（如 fusion, minimalism, uk jazz）
- instruments: 主要乐器
- rhythm: straight/swing/syncopation/shuffle/polyrhythm
- time_signature: 4/4, 3/4, 5/4, 7/8 ...
- tempo_bpm: 区间字符串（如 "90-110"）或 slow/medium/fast
- region, era
- label: 唱片厂牌（若可得）
- awards: 重要奖项（含年份，若可得）
- similar_artists: 3-8 个
- evidence_terms: 8-15 个检索关键词（英/中混合）
- confidence: 为每个字段给出 0~1 置信度（如 { "tags":0.8, "rhythm":0.6, ... }）
严格核对“艺术家名 + 歌曲名”一致性；不确定时 conf<0.5。仅输出 JSON。
""".strip()

def profile_song(seed: str) -> dict:
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": PROMPT_SYSTEM},
            {"role": "user", "content": seed}
        ]
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return {"error": "parse_failed", "raw": resp.choices[0].message.content}

def build_queries(p: dict) -> list:
    bits = []
    bits += p.get("tags", [])[:6]
    bits += p.get("instruments", [])[:3]
    for k in ["rhythm", "time_signature", "region", "era"]:
        if p.get(k): bits.append(str(p[k]))
    if p.get("tempo_bpm"): bits.append(str(p["tempo_bpm"]))
    base = " ".join(bits)
    artist = p.get("artist", "")
    title = p.get("title", "")

    qs = [
        f'{artist} "{title}" review site:allmusic.com',
        f'{base} site:bandcamp.com',
        f'"for fans of" {artist} site:bandcamp.com',
        f'{base} similar artists',
        f'{base} playlist',
        f'{artist} {title} 奖项 OR 获奖',
        f'{artist} 类似 音乐'
    ]
    # 去重
    seen, out = set(), []
    for q in qs:
        if q and q not in seen:
            out.append(q); seen.add(q)
    return out

def ddg_search(queries: list, k=36) -> list:
    rows, seen = [], set()
    per = max(2, k // max(1, len(queries)))
    try:
        with DDGS() as ddgs:
            for q in queries:
                # 兼容不同版本：优先用 query=，失败再试 keywords=
                try:
                    it = ddgs.text(
                        query=q,             # 关键：用 query
                        max_results=per,
                        region="wt-wt",
                        safesearch="off",
                        # timelimit="y",     # 需要时再开；部分版本不支持这个参数
                    )
                except TypeError:
                    it = ddgs.text(
                        keywords=q,         # 旧版本兜底
                        max_results=per,
                        region="wt-wt",
                        safesearch="off",
                    )

                for r in it:
                    href = (r.get("href") or "").strip()
                    if href and href not in seen:
                        seen.add(href)
                        rows.append(r)
    except Exception as e:
        import streamlit as st
        st.error(f"DuckDuckGo 检索失败：{type(e).__name__}: {e}")
    return rows


PRIORITY = [
    "bandcamp.com","rateyourmusic.com","allmusic.com","pitchfork.com",
    "youtube.com","open.spotify.com","music.apple.com","tidal.com"
]

def rerank(profile: dict, results: list, topn=12):
    tags = set(x.lower() for x in profile.get("tags", []))
    inst = set(x.lower() for x in profile.get("instruments", []))
    rhy  = (profile.get("rhythm") or "").lower()
    ts   = (profile.get("time_signature") or "").lower()
    infl = [x.lower() for x in profile.get("similar_artists", [])] + \
           [x.lower() for x in profile.get("evidence_terms", [])]

    scored = []
    for r in results:
        text = f'{r.get("title","")} {r.get("body","")} {r.get("href","")}'.lower()
        s, why = 0.0, []
        th = sum(1 for t in tags if t and t in text); s += 0.6 * th
        if th: why.append(f"风格命中×{th}")
        ih = sum(1 for i in inst if i and i in text); s += 0.5 * ih
        if ih: why.append(f"乐器吻合×{ih}")
        if rhy and rhy in text: s += 0.5; why.append("节奏吻合")
        if ts and ts in text:  s += 0.4; why.append("拍号吻合")
        if infl:
            best = max(fuzz.token_set_ratio(k, text) for k in infl)
            s += best / 150
        dom_bonus = next((0.8 - 0.1*i for i, d in enumerate(PRIORITY) if d in text), 0)
        s += dom_bonus
        scored.append((s, why, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:topn]

# ====== Streamlit UI ======
st.set_page_config(page_title="音乐画像 → 联网相似检索", layout="wide")
st.title("🎧 歌曲画像 → 相似检索")

seed = st.text_input(
    "输入格式：歌曲名 - 艺术家（最好带年份/专辑提高准确率）",
    "Umbra - GoGo Penguin (2025)"
)
if st.button("分析并检索", use_container_width=True):
    # 1) 画像
    with st.spinner("生成画像…"):
        prof = profile_song(seed)
    st.subheader("画像（JSON）"); st.json(prof, expanded=False)

    # 2) 查询
    qs = build_queries(prof)
    st.caption("自动生成的查询：")
    for q in qs: st.code(q)

    # 3) 检索 + 重排
    with st.spinner("DuckDuckGo 搜索中…"):
        results = ddg_search(qs, k=36)
    st.caption(f"检索到 {len(results)} 条原始结果")
    ranked = rerank(prof, results, topn=12)

    st.subheader("Top 匹配与理由")
    for sc, why, r in ranked:
        st.markdown(
            f"**[{r.get('title','(无标题)')}]({r.get('href','#')})**  \n"
            f"分数: `{sc:.2f}`  \n{r.get('body','')}\n"
            f"_理由：{'; '.join(why) or '关键词相关'}_"
        )
