import json
from datetime import date, datetime, timedelta
from typing import List, Dict, Any

import streamlit as st
import openai
openai.api_key = st.secrets["OPENAI_API_KEY"]

# ---------------- CONFIG BÁSICA ----------------
st.set_page_config(page_title="ContentForge v9.2", layout="wide")

client = OpenAI()  # OPENAI_API_KEY vem das secrets/env


# ---------------- ESTADO INICIAL ----------------
def init_state():
    today = date.today().isoformat()
    if "gen_date" not in st.session_state:
        st.session_state.gen_date = today
        st.session_state.gen_count = 0
    if st.session_state.gen_date != today:
        st.session_state.gen_date = today
        st.session_state.gen_count = 0

    if "planner_events" not in st.session_state:
        # cada evento: {id, day, time, title, platform, caption, hashtags, completed, score}
        st.session_state.planner_events: List[Dict[str, Any]] = []

    if "week_anchor" not in st.session_state:
        st.session_state.week_anchor = date.today()

    if "generated_variations" not in st.session_state:
        st.session_state.generated_variations: List[Dict[str, Any]] = []

    if "plan" not in st.session_state:
        st.session_state.plan = "Starter"


init_state()


# ---------------- PLANOS ----------------
PLAN_CONFIG = {
    "Starter": {
        "daily_generations": 5,
        "analysis": False,
        "performance": False,
    },
    "Pro": {
        "daily_generations": 50,
        "analysis": True,
        "performance": True,
    },
}


def get_plan_limits(plan: str) -> Dict[str, Any]:
    return PLAN_CONFIG.get(plan, PLAN_CONFIG["Starter"])


# ---------------- HELPERS GERAIS ----------------
def week_bounds(anchor: date):
    monday = anchor - timedelta(days=anchor.weekday())
    sunday = monday + timedelta(days=6)
    return monday, sunday


def score_caption(caption: str) -> Dict[str, float]:
    """Pseudo-análise local para o Pro (sem segunda chamada à API)."""
    text = caption.lower()
    length = len(caption)

    has_offer = any(k in text for k in ["desconto", "%", "promo", "oferta", "só hoje"])
    has_cta = any(
        k in text
        for k in [
            "link na bio",
            "clica",
            "envia mensagem",
            "comenta",
            "guarda",
            "partilha",
            "compartilha",
        ]
    )
    has_emotion = any(
        k in text
        for k in ["história", "sonho", "confiança", "incrível", "mudança", "transformação"]
    )

    clarity = 7.0
    if 80 <= length <= 260:
        clarity += 2
    elif length < 60:
        clarity -= 1
    elif length > 400:
        clarity -= 1.5

    conversion = 6.0 + (1.5 if has_offer else 0) + (1.5 if has_cta else 0)
    engagement = 6.0 + (1.5 if has_emotion else 0)
    emotion = 6.0 + (2.0 if has_emotion else 0)
    platform_fit = 7.0  # podia ser adaptado por plataforma mais tarde

    def clamp(x: float) -> float:
        return max(0.0, min(10.0, x))

    metrics = {
        "clarity": round(clamp(clarity), 1),
        "conversion": round(clamp(conversion), 1),
        "engagement": round(clamp(engagement), 1),
        "emotion": round(clamp(emotion), 1),
        "platform_fit": round(clamp(platform_fit), 1),
    }
    final = (
        metrics["conversion"] * 0.3
        + metrics["engagement"] * 0.25
        + metrics["clarity"] * 0.15
        + metrics["platform_fit"] * 0.15
        + metrics["emotion"] * 0.15
    )
    metrics["final_score"] = round(clamp(final), 1)
    return metrics


# ---------------- PROMPT GERAÇÃO ----------------
def build_generation_prompt(
    brand: str,
    niche: str,
    tone: str,
    platform: str,
    copy_mode: str,
    goal: str,
    extra: str,
    plan: str,
) -> str:
    tone_map = {
        "profissional": "profissional, objetivo mas humano",
        "premium": "premium, elegante, linguagem cuidada",
        "emocional": "emocional, próximo e empático",
        "casual": "casual, descontraído, estilo conversa",
    }
    tone_txt = tone_map.get(tone, "profissional, humano")

    mode_map = {
        "Venda": "foco em conversão e vendas",
        "Storytelling": "foco em história e ligação emocional",
        "Educacional": "foco em ensinar algo útil e aplicável",
    }
    mode_txt = mode_map.get(copy_mode, "equilíbrio entre valor e conversão")

    pro_txt = (
        "Estás no modo PRO: o utilizador é exigente, o texto tem de parecer escrito por um copywriter sénior."
        if plan == "Pro"
        else "Estás no modo Starter: mantém texto simples mas profissional."
    )

    return f"""Quero que cries 3 VARIAÇÕES de legendas em PT-PT para redes sociais.

Marca: {brand}
Nicho: {niche}
Plataforma: {platform}
Tom de voz: {tone_txt}
Modo de copy: {mode_txt}
Objetivo do dia: {goal or "não especificado"}
Informação extra relevante: {extra or "nenhuma informação extra"}
{pro_txt}

Regras:
- NÃO copies literalmente frases do utilizador (especialmente coisas como "quero levar as pessoas ao site"). Reescreve de forma profissional.
- Frases curtas, respiráveis, boas para ler no telemóvel.
- Usa emojis com intenção (máx. 3–4 por legenda).
- Inclui SEMPRE um CTA no fim (mas não repitas o mesmo CTA nas 3 variações).
- Adapta o estilo à plataforma (Instagram = mais visual/emocional).

Para cada variação (A, B, C) devolve:
- "id": "A" ou "B" ou "C"
- "titulo": título curto para o planner (máx. 60 caracteres)
- "legenda": texto completo (inclui o CTA no fim)
- "hashtags": lista com 10–15 hashtags relevantes (sem #love, #insta, etc.)
- "cta": a frase final de chamada à ação
- "angulo": descrição rápida do ângulo (ex: urgência, bastidores, story, prova social)

Formata a resposta EXCLUSIVAMENTE como JSON com esta estrutura:

{{
  "variacoes": [
    {{
      "id": "A",
      "titulo": "...",
      "legenda": "...",
      "hashtags": ["#exemplo", "..."],
      "cta": "...",
      "angulo": "..."
    }},
    {{
      "id": "B",
      "titulo": "...",
      "legenda": "...",
      "hashtags": ["#exemplo", "..."],
      "cta": "...",
      "angulo": "..."
    }},
    {{
      "id": "C",
      "titulo": "...",
      "legenda": "...",
      "hashtags": ["#exemplo", "..."],
      "cta": "...",
      "angulo": "..."
    }}
  ]
}}"""


def call_openai_json(prompt: str) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.9,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "És um copywriter sénior de social media que escreve como um humano, em PT-PT.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    content = resp.choices[0].message.content
    return json.loads(content)


# ---------------- UI HELPERS ----------------
def render_auto_analysis(stats: Dict[str, float], is_recommended: bool, plan: str):
    """Renderiza o bloco de análise automática com emojis e badge."""
    st.markdown("**🔍 Análise automática**")

    if plan == "Pro" and is_recommended:
        st.markdown(
            """
            <div style="display:inline-block; padding:4px 10px; border-radius:999px;
                        background-color:#facc15; color:#1a1a1a; font-size:0.8rem;
                        font-weight:600; margin:4px 0 8px 0;">
                🌟 Nossa recomendação
            </div>
            """,
            unsafe_allow_html=True,
        )

    if plan != "Pro":
        st.markdown(
            "<span style='font-size:0.85rem;'>🔒 Análise automática completa disponível no plano <strong>Pro</strong>.</span>",
            unsafe_allow_html=True,
        )
        return

    final_score = stats.get("final_score", 0.0)
    clarity = stats.get("clarity", 0.0)
    engagement = stats.get("engagement", 0.0)
    conversion = stats.get("conversion", 0.0)
    emotion = stats.get("emotion", 0.0)
    platform_fit = stats.get("platform_fit", 0.0)

    st.markdown(
        f"✨ **Score final:** **{final_score:.1f} / 10**",
        unsafe_allow_html=True,
    )
    st.markdown(
        (
            f"🧠 **Clareza:** {clarity:.1f} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"💬 **Engajamento:** {engagement:.1f}<br>"
            f"💰 **Conversão:** {conversion:.1f} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"❤️ **Emoção:** {emotion:.1f}<br>"
            f"📱 **Adequação à plataforma:** {platform_fit:.1f}"
        ),
        unsafe_allow_html=True,
    )


# ---------------- SIDEBAR ----------------
def sidebar_profile():
    st.sidebar.header("Plano e perfil")

    plan = st.sidebar.selectbox(
        "Plano",
        ["Starter", "Pro"],
        index=0 if st.session_state.plan == "Starter" else 1,
        key="plan",
    )
    limits = get_plan_limits(plan)

    st.sidebar.markdown(
        f"**Gerações hoje:** {st.session_state.gen_count}/{limits['daily_generations']}"
    )

    st.sidebar.markdown("---")

    brand = st.sidebar.text_input("Marca", value="Loukisses")
    niche = st.sidebar.text_input("Nicho/tema", value="Moda feminina")
    tone = st.sidebar.selectbox(
        "Tom de voz",
        ["profissional", "premium", "emocional", "casual"],
        index=1,
    )
    copy_mode = st.sidebar.selectbox(
        "Modo de copy", ["Venda", "Storytelling", "Educacional"], index=0
    )

    return plan, brand, niche, tone, copy_mode


# ---------------- PÁGINA GERAR ----------------
def page_generate(plan: str, brand: str, niche: str, tone: str, copy_mode: str):
    limits = get_plan_limits(plan)

    st.subheader("⚡ Geração inteligente de conteúdo")

    col1, col2 = st.columns(2)
    with col1:
        goal = st.text_input(
            "O que queres comunicar hoje?",
            value="Lançamento da nova coleção de Outono",
        )
    with col2:
        extra = st.text_area(
            "Informação extra (opcional)",
            value="Desconto de 10% no site até domingo.",
            height=70,
        )

    platform = st.selectbox("Plataforma principal", ["Instagram", "TikTok"], index=0)

    can_generate = st.session_state.gen_count < limits["daily_generations"]
    gen_btn = st.button("⚡ Gerar agora", disabled=not can_generate)

    if not can_generate:
        st.info("Atingiste o limite de gerações de hoje para o teu plano.")

    if gen_btn and can_generate:
        with st.spinner("A gerar variações com IA..."):
            prompt = build_generation_prompt(
                brand, niche, tone, platform, copy_mode, goal, extra, plan
            )
            data = call_openai_json(prompt)
            variations = data.get("variacoes", [])

            # Análise local (apenas Pro)
            if limits["analysis"]:
                for v in variations:
                    v["analysis"] = score_caption(v["legenda"])
                best = max(
                    variations,
                    key=lambda v: v["analysis"]["final_score"],
                    default=None,
                )
                if best:
                    best["recommended"] = True

            st.session_state.generated_variations = variations
            st.session_state.gen_count += 1

    variations = st.session_state.generated_variations
    if not variations:
        st.info("Gera conteúdo para veres as variações aqui em baixo.")
        return

    st.markdown("### Resultados")

    cols = st.columns(3)
    for col, var in zip(cols, variations):
        with col:
            vid = var.get("id", "?")
            st.markdown(f"**Variação {vid}**")

            is_rec = bool(var.get("recommended", False))

            if is_rec and plan != "Pro":
                st.markdown(
                    """
                    <div style="display:inline-block; padding:4px 10px; border-radius:999px;
                                background-color:#e5e7eb; color:#111827; font-size:0.8rem;
                                font-weight:600; margin:4px 0 8px 0;">
                        🌟 Nossa recomendação (Pro)
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown(f"**Título (planner):** {var['titulo']}")
            st.write(var["legenda"])

            st.markdown("**Hashtags:**")
            st.code(" ".join(var.get("hashtags", [])))

            # Análise automática
            stats = var.get("analysis")
            render_auto_analysis(stats or {}, is_rec, plan)

            st.markdown("---")
            st.markdown("**Adicionar ao planner**")
            d_col, h_col = st.columns(2)
            with d_col:
                day = st.date_input(
                    f"Dia {vid}", value=date.today(), key=f"day_{vid}"
                )
            with h_col:
                time_val = st.time_input(
                    f"Hora {vid}",
                    value=datetime.strptime("18:00", "%H:%M").time(),
                    key=f"time_{vid}",
                )
                time_str = time_val.strftime("%H:%M")

            if st.button("➕ Adicionar", key=f"add_{vid}"):
                st.session_state.planner_events.append(
                    {
                        "id": f"{datetime.utcnow().timestamp()}_{vid}",
                        "day": day.isoformat(),
                        "time": time_str,
                        "title": var["titulo"],
                        "platform": platform,
                        "caption": var["legenda"],
                        "hashtags": var.get("hashtags", []),
                        "completed": False,
                        "score": (stats or {}).get("final_score") if stats else None,
                    }
                )
                st.success("Adicionado ao planner ✅")


# ---------------- PÁGINA PLANNER (v9.2 simples e estável) ----------------
def page_planner(plan: str):
    st.subheader("📅 Planner semanal")

    # navegação de semanas
    col_prev, col_center, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("« Semana anterior"):
            st.session_state.week_anchor -= timedelta(days=7)
    with col_next:
        if st.button("Semana seguinte »"):
            st.session_state.week_anchor += timedelta(days=7)
    with col_center:
        anchor_ui = st.date_input("Semana de referência", value=st.session_state.week_anchor)
        if anchor_ui != st.session_state.week_anchor:
            st.session_state.week_anchor = anchor_ui

    week_start, week_end = week_bounds(st.session_state.week_anchor)
    st.caption(
        f"Semana de {week_start.strftime('%d/%m')} a {week_end.strftime('%d/%m')}"
    )

    events = st.session_state.planner_events
    days = [week_start + timedelta(days=i) for i in range(7)]
    by_day: Dict[str, List[Dict[str, Any]]] = {d.isoformat(): [] for d in days}
    for ev in events:
        if week_start.isoformat() <= ev["day"] <= week_end.isoformat():
            by_day.setdefault(ev["day"], []).append(ev)

    cols = st.columns(7)
    day_labels = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]

    for idx, d in enumerate(days):
        d_iso = d.isoformat()
        posts = by_day.get(d_iso, [])
        with cols[idx]:
            st.markdown(
                f"<div style='text-align:center; font-weight:600;'>{day_labels[idx]}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='text-align:center; color:gray; margin-bottom:8px;'>{d.strftime('%d/%m')}</div>",
                unsafe_allow_html=True,
            )

            if not posts:
                st.markdown(
                    "<div style='text-align:center; font-size:0.8rem; color:#888;'>Sem tarefas</div>",
                    unsafe_allow_html=True,
                )
                continue

            for ev in sorted(posts, key=lambda e: e["time"]):
                completed = bool(ev.get("completed"))
                bg = "#E8FDF1" if completed else "#f7f7f7"
                status_txt = "Concluído ✅" if completed else "Pendente"
                status_color = "#00c46b" if completed else "#666666"

                card_html = f"""
                <div style="
                    background:{bg};
                    border-radius:12px;
                    padding:8px 10px;
                    margin:0 auto 8px auto;
                    border:1px solid #ddd;
                    text-align:left;
                    max-width:220px;
                ">
                  <div style="font-size:0.8rem; color:#000;">{ev['time']} · {ev['platform']}</div>
                  <div style="font-weight:600; font-size:0.85rem; color:#000;">{ev['title']}</div>
                  <div style="font-size:0.75rem; color:{status_color}; margin-top:4px;">{status_txt}</div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)

                with st.expander("Ver detalhes", expanded=False):
                    st.markdown(f"**Legenda:**\n\n{ev['caption']}")
                    if ev.get("hashtags"):
                        st.markdown("**Hashtags:**")
                        st.code(" ".join(ev["hashtags"]))
                    if ev.get("score") is not None:
                        st.markdown(f"**Score previsto:** {ev['score']}/10")

                    col_a, col_b = st.columns(2)
                    with col_a:
                        if not completed:
                            if st.button(
                                "✔ Concluir",
                                key=f"done_{ev['id']}",
                            ):
                                ev["completed"] = True
                                st.success("Tarefa marcada como concluída ✅")
                                st.experimental_rerun()
                        else:
                            st.markdown("Concluído ✅")
                    with col_b:
                        if st.button(
                            "🗑 Remover",
                            key=f"del_{ev['id']}",
                        ):
                            st.session_state.planner_events = [
                                e for e in st.session_state.planner_events if e["id"] != ev["id"]
                            ]
                            st.warning("Tarefa removida.")
                            st.experimental_rerun()


# ---------------- PÁGINA PERFORMANCE (v9.2) ----------------
def page_performance(plan: str):
    st.subheader("📊 Performance (Pro)")

    if not PLAN_CONFIG[plan]["performance"]:
        st.info("🔒 A aba de performance detalhada é exclusiva do plano Pro.")
        return

    completed = [e for e in st.session_state.planner_events if e.get("completed")]
    if not completed:
        st.info("Ainda não tens tarefas concluídas.")
        return

    # KPIs
    total_posts = len(st.session_state.planner_events)
    concluidos = len(completed)
    taxa = (concluidos / total_posts * 100) if total_posts > 0 else 0.0
    scores = [e["score"] for e in completed if isinstance(e.get("score"), (int, float))]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    with kpi_col1:
        st.metric("Posts no planner", total_posts)
    with kpi_col2:
        st.metric("Concluídos", concluidos)
    with kpi_col3:
        st.metric("Taxa conclusão", f"{taxa:.1f}%")
    with kpi_col4:
        st.metric("Score médio previsto", f"{avg_score:.1f}/10")

    # Hora recomendada
    hour_scores: Dict[str, List[float]] = {}
    for ev in completed:
        h = ev["time"]
        s = ev.get("score")
        if isinstance(s, (int, float)):
            hour_scores.setdefault(h, []).append(float(s))

    if hour_scores:
        best_hour = None
        best_score = -1.0
        for h, vals in hour_scores.items():
            m = sum(vals) / len(vals)
            if m > best_score:
                best_score = m
                best_hour = h
        precision_label = "Baixa"
        if concluidos > 5:
            precision_label = "Alta"
        elif concluidos > 1:
            precision_label = "Média"

        st.markdown("---")
        st.markdown("### 🕒 Hora recomendada para postar")
        st.markdown(f"**{best_hour}**")
        st.markdown(f"*Precisão: {precision_label}*")
        st.markdown("*Precisão da IA aumenta com o nº de postagens.*")

    # Lista das últimas publicações concluídas
    st.markdown("---")
    st.markdown("### 📋 Publicações concluídas")

    for ev in sorted(completed, key=lambda e: (e["day"], e["time"]), reverse=True):
        linha = f"- {ev['day']} {ev['time']} · {ev['platform']} · **{ev['title']}**"
        if ev.get("score") is not None:
            linha += f" ({ev['score']}/10)"
        st.markdown(linha)


# ---------------- MAIN ----------------
def main():
    plan, brand, niche, tone, copy_mode = sidebar_profile()

    st.title("ContentForge v9.2")
    st.caption(
        "Gera conteúdo com IA, organiza num planner semanal e acompanha a performance (Pro)."
    )

    tab_gen, tab_plan, tab_perf = st.tabs(["⚡ Gerar", "📅 Planner", "📊 Performance"])

    with tab_gen:
        page_generate(plan, brand, niche, tone, copy_mode)
    with tab_plan:
        page_planner(plan)
    with tab_perf:
        page_performance(plan)


if __name__ == "__main__":
    main()
