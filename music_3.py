# app.py —— 音乐画像 → DuckDuckGo 相似检索（API易网关 + 可调权重 + 可播放过滤）
import os, json
import streamlit as st
from openai import OpenAI
from ddgs import DDGS
from rapidfuzz import fuzz

# 只能调用一次，且必须最先出现
st.set_page_config(page_title="🎧 音乐画像 → 联网相似检索", layout="wide")

# ============== 基础配置 ==============
API_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.apiyi.com/v1")
API_KEY = os.getenv("OPENAI_API_KEY") or "sk-EbAhbINv19eVTaX4CeAb989924A540A1952611Ec45867b81"
MODEL = os.getenv("OPENAI_MODEL", "gpt-5")  # 视你的网关映射改为 gpt-4o / gpt-5-thinking 等

if not API_KEY:
    st.warning("未检测到 OPENAI_API_KEY 环境变量。Windows: setx OPENAI_API_KEY \"你的密钥\"")

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

# ============== 侧边栏：权重与参数 ==============
def get_preferences():
    with st.sidebar:
        st.header("⚙️ 打分偏好")
        preset = st.selectbox("预设", ["均衡", "偏风格", "偏乐器", "偏节奏/拍号"], index=0)
        defaults_map = {
            "均衡":       dict(genre=0.6, inst=0.5, rhythm=0.5, timesig=0.4, influence=0.4, domain=0.3),
            "偏风格":     dict(genre=1.0, inst=0.4, rhythm=0.3, timesig=0.2, influence=0.3, domain=0.3),
            "偏乐器":     dict(genre=0.4, inst=1.0, rhythm=0.3, timesig=0.2, influence=0.3, domain=0.3),
            "偏节奏/拍号": dict(genre=0.4, inst=0.4, rhythm=1.0, timesig=0.8, influence=0.3, domain=0.3),
        }
        d = defaults_map[preset]
        w_genre   = st.slider("风格/标签 权重",      0.0, 1.5, d["genre"],   0.05)
        w_inst    = st.slider("乐器匹配 权重",        0.0, 1.5, d["inst"],    0.05)
        w_rhythm  = st.slider("节奏类型 权重",        0.0, 1.5, d["rhythm"],  0.05)
        w_timesig = st.slider("拍号 权重",            0.0, 1.5, d["timesig"], 0.05)
        w_infl    = st.slider("影响/相似艺人 权重",   0.0, 1.5, d["influence"], 0.05)
        w_domain  = st.slider("权威站点加权",         0.0, 1.0, d["domain"],  0.05)

        topn = st.slider("返回数量 Top N", 5, 30, 12, 1)
        mode = st.radio("查询模式", ["严格（更精准）", "宽松（更易召回）"], horizontal=True, index=0)

        st.markdown("---")
        st.header("🔗 链接偏好")
        playable_only = st.checkbox("只显示可播放链接", True)
        allowed_domains = st.multiselect(
            "允许的可播放域名",
            ["youtube.com","youtu.be","open.spotify.com","music.apple.com",
             "bandcamp.com","soundcloud.com","tidal.com","deezer.com"],
            default=["youtube.com","youtu.be"]  # 你现在只要 YouTube
        )
        # 若只选了 YouTube，两端都做 site 限定（检索更干净）
        limit_to_youtube = playable_only and set(allowed_domains).issubset({"youtube.com","youtu.be"})

        weights = dict(
            genre=w_genre, inst=w_inst, rhythm=w_rhythm,
            timesig=w_timesig, influence=w_infl, domain=w_domain
        )
    return weights, topn, mode, playable_only, allowed_domains, limit_to_youtube

weights, topn, mode, playable_only, allowed_domains, limit_to_youtube = get_preferences()

# ============== 画像（GPT） ==============
PROMPT_SYSTEM = """
你是一个严谨的音乐考据型助手。用户会提供“歌曲名 - 艺术家”，请仅输出 JSON，字段：
- title, artist
- tags: 5-12 个风格/子风格标签
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
        model=MODEL, temperature=0.2,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": PROMPT_SYSTEM},
                  {"role": "user", "content": seed}]
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return {"error":"parse_failed","raw":resp.choices[0].message.content}

# ============== 查询构造 ==============
def build_queries(p: dict, mode: str, limit_to_youtube: bool) -> list:
    artist = (p.get("artist") or "").strip()
    title  = (p.get("title") or "").strip()

    bits = []
    bits += [t for t in p.get("tags", [])[:6] if t]
    bits += [t for t in p.get("instruments", [])[:3] if t]
    if mode.startswith("严格"):
        for k in ["rhythm","time_signature","region","era"]:
            if p.get(k): bits.append(str(p[k]))
        if p.get("tempo_bpm"): bits.append(str(p["tempo_bpm"]))
    base = " ".join(bits).strip()

    qs = []
    if artist and title:
        qs += [f'{artist} "{title}" review site:allmusic.com',
               f'{artist} "{title}" site:pitchfork.com',
               f'{artist} "{title}" site:bandcamp.com']
    if base:
        qs += [f'{base} site:bandcamp.com',
               f'{base} similar artists',
               f'{base} playlist']
    if artist:
        qs += [f'"for fans of" {artist} site:bandcamp.com',
               f'{artist} 类似 音乐',
               f'{artist} {title} 奖项 OR 获奖' if title else f'{artist} 奖项 OR 获奖']

    # 宽松兜底
    if artist or title:
        qs.append(f'{artist} {title} similar songs OR 相似 歌曲')

    # 如果用户仅想要 YouTube，这里直接加站点限定
    if limit_to_youtube:
        qs = [f'{q} site:youtube.com OR site:youtu.be' for q in qs]

    seen, out = set(), []
    for q in qs:
        q = q.strip()
        if q and q not in seen:
            out.append(q); seen.add(q)
    return out

# ============== DuckDuckGo 搜索 ==============
def ddg_search(queries: list, k=36) -> list:
    rows, seen = [], set()
    per = max(2, k // max(1, len(queries)))
    try:
        with DDGS() as ddgs:
            for q in queries:
                try:
                    it = ddgs.text(query=q, max_results=per, region="wt-wt", safesearch="off")
                except TypeError:
                    it = ddgs.text(keywords=q, max_results=per, region="wt-wt", safesearch="off")
                for r in it:
                    href = (r.get("href") or "").strip()
                    if href and href not in seen:
                        seen.add(href); rows.append(r)
    except Exception as e:
        st.error(f"DuckDuckGo 检索失败：{type(e).__name__}: {e}")
    return rows

# 按“可播放体验”优先排序（YouTube/Spotify…靠前）
PRIORITY = [
    "youtube.com","youtu.be",
    "open.spotify.com","music.apple.com",
    "bandcamp.com","soundcloud.com","tidal.com","deezer.com",
    "rateyourmusic.com","allmusic.com","pitchfork.com"
]

# ============== 重排（使用权重 + 贡献明细） ==============
def rerank(profile: dict, results: list, weights: dict, topn=12):
    tags = set((x or "").lower() for x in profile.get("tags",[]))
    inst = set((x or "").lower() for x in profile.get("instruments",[]))
    rhy  = (profile.get("rhythm") or "").lower()
    ts   = (profile.get("time_signature") or "").lower()
    infl = [ (x or "").lower() for x in profile.get("similar_artists",[]) ] + \
           [ (x or "").lower() for x in profile.get("evidence_terms",[]) ]

    out=[]
    for r in results:
        text = f'{r.get("title","")} {r.get("body","")} {r.get("href","")}'.lower()
        contrib, score = {}, 0.0

        tag_hits = sum(1 for t in tags if t and t in text)
        contrib["风格"] = tag_hits * weights["genre"]; score += contrib["风格"]

        inst_hits = sum(1 for i in inst if i and i in text)
        contrib["乐器"] = inst_hits * weights["inst"]; score += contrib["乐器"]

        contrib["节奏"] = weights["rhythm"] if (rhy and rhy in text) else 0.0
        contrib["拍号"] = weights["timesig"] if (ts  and ts  in text) else 0.0
        score += contrib["节奏"] + contrib["拍号"]

        infl_best = max((fuzz.token_set_ratio(k, text) for k in infl), default=0)
        contrib["影响"] = (infl_best/100.0) * weights["influence"]; score += contrib["影响"]

        dom_bonus = 0.0
        for i,dom in enumerate(PRIORITY):
            if dom in text:
                dom_bonus = weights["domain"] * (1.0 - 0.1*i); break
        contrib["站点"] = max(dom_bonus, 0.0); score += contrib["站点"]

        out.append({
            "score": round(score, 3),
            "why": contrib,
            "title": r.get("title","(无标题)"),
            "href": r.get("href","#"),
            "body": r.get("body","")
        })

    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:topn]

# ============== UI 主体 ==============
st.title("🎧 歌曲画像 → 相似检索")
seed = st.text_input("输入：歌曲名 - 艺术家（尽量精确，含年份/专辑更好）",
                     "Umbra - GoGo Penguin (2025)")

if st.button("分析并检索", use_container_width=True):
    # 1) 画像
    with st.spinner("生成画像…"):
        prof = profile_song(seed)
    st.subheader("画像（JSON）"); st.json(prof, expanded=False)
    if "error" in prof:
        st.error("画像解析失败："); st.code(prof.get("raw","")); st.stop()

    # 2) 查询
    qs = build_queries(prof, mode, limit_to_youtube)
    st.caption("自动生成的查询：")
    for q in qs: st.code(q)

    # 3) 检索
    with st.spinner("DuckDuckGo 搜索中…"):
        results = ddg_search(qs, k=max(topn*3, 24))
    st.caption(f"检索到 {len(results)} 条原始结果")

    # 4) 只保留可播放域名（如果用户勾选）
    if playable_only:
        def allowed(url: str) -> bool:
            u = (url or "").lower()
            return any(dom in u for dom in allowed_domains)
        results = [r for r in results if allowed(r.get("href",""))]
        st.caption(f"筛选后剩余 {len(results)} 条可播放链接")

    # 5) 重排 + 展示
    ranked = rerank(prof, results, weights, topn=topn)
    st.subheader("Top 匹配与理由")
    if not ranked:
        st.info("没有命中结果。试试切到“宽松”或放开更多可播放域名。")
    for item in ranked:
        st.markdown(f"**[{item['title']}]({item['href']})**  \n分数: `{item['score']:.2f}`  \n{item['body']}")
        with st.expander("贡献明细"):
            st.write({k: round(v, 3) for k, v in item["why"].items()})

