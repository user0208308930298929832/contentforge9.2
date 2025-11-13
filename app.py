import streamlit as st
from datetime import datetime, date, time, timedelta
from typing import List, Dict, Any, Optional
from openai import OpenAI
import json
import uuid
import statistics

# -----------------------------
# CONFIGURAÇÃO GERAL
# -----------------------------
st.set_page_config(
    page_title="ContentForge v9.2",
    layout="wide",
    page_icon="🍏",
)

st.markdown(
    """
    <style>
    /* Fonte e feeling mais “Apple” */
    html, body, [class*="css"]  {
        font-family: -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
    }
    /* Cards do planner */
    .cf-card {
        border-radius: 14px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
        background: #111111;
        border: 1px solid #333333;
        color: #f5f5f5;
    }
    .cf-card-done {
        background: #d9fdd3 !important;
        border-color: #9be69b !important;
        color: #111111 !important;
    }
    .cf-badge-reco {
        display: inline-flex;
        align-items: center;
        padding: 0.15rem 0.5rem;
        border-radius: 999px;
        background: #f7e49c;
        color: #3a2c00;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }
    .cf-badge-lock {
        display:inline-flex;
        align-items:center;
        padding:0.4rem 0.8rem;
        border-radius:999px;
        background:#3e3a19;
        color:#f5f5d7;
        font-size:0.85rem;
        margin-top:0.3rem;
    }
    .cf-subtle {
        font-size: 0.8rem;
        opacity: 0.7;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# CLIENTE OPENAI (SDK NOVA)
# -----------------------------
def get_openai_client() -> OpenAI:
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


# -----------------------------
# ESTADO INICIAL
# -----------------------------
if "planner_items" not in st.session_state:
    st.session_state.planner_items: List[Dict[str, Any]] = []

if "anchor_date" not in st.session_state:
    st.session_state.anchor_date: date = date.today()

if "selected_task_id" not in st.session_state:
    st.session_state.selected_task_id: Optional[str] = None

if "geracoes_hoje" not in st.session_state:
    st.session_state.geracoes_hoje: int = 0


# -----------------------------
# FUNÇÕES AUXILIARES
# -----------------------------
def gerar_variacoes_legenda(
    marca: str,
    nicho: str,
    tom: str,
    modo_copy: str,
    plataforma: str,
    mensagem: str,
    extra: Optional[str] = "",
) -> List[Dict[str, Any]]:
    """
    Chama o modelo da OpenAI e devolve uma lista de 3 variações:
    [
      {
        "titulo_planner": "...",
        "legenda": "...",
        "hashtags": ["#..."],
        "score_final": 7.8,
        "engajamento": 7.1,
        "conversao": 8.3,
        "recomendado": true/false
      }, ...
    ]
    """
    system_prompt = (
        "És o ContentForge, um assistente de marketing que cria legendas em PT-PT "
        "para Instagram e TikTok. Produz sempre texto natural, direto e adaptado ao nicho."
    )

    user_prompt = f"""
Marca: {marca}
Nicho: {nicho}
Tom de voz: {tom}
Modo de copy: {modo_copy} (ex: Venda, Storytelling, Educacional)
Plataforma: {plataforma}

O que queres comunicar hoje?
- {mensagem}

Informação extra (detalhes, promoções, urgência, etc.):
- {extra or "Sem informação extra."}

TAREFA:
Cria 3 variações de conteúdo para um post nesta plataforma.

Para cada variação, devolve JSON com:
- "titulo_planner": frase curta tipo título para aparecer no planner
- "legenda": copy completo e final (max ~5 linhas)
- "hashtags": lista com 10 a 15 hashtags em PT ou relevantes
- "score_final": número entre 0 e 10 (força geral da ideia)
- "engajamento": número 0-10 (potencial de comentários/guardados)
- "conversao": número 0-10 (probabilidade de cliques/vendas)
- "recomendado": true se for a melhor opção

Responde apenas em JSON válido com uma lista de 3 elementos.
"""

    client = get_openai_client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.8,
    )

    raw = response.choices[0].message.content.strip()

    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            data = [data]
        return data
    except Exception:
        return []


def analise_automatica_legenda(texto: str) -> Dict[str, float]:
    """
    Heurística local simples (sem nova chamada à API) para análise automática.
    """
    length = len(texto)
    clareza = 7.0
    if length < 120:
        clareza += 1
    if "?" in texto:
        clareza += 0.5

    eng = 6.0
    emojis = sum(ch in "🔥✨💥🎯💡🧠❤️😍📣📌💬" for ch in texto)
    if emojis >= 2:
        eng += 1
    if any(word in texto.lower() for word in ["comenta", "partilha", "guarda", "marca alguém"]):
        eng += 1

    conv = 6.0
    if any(x in texto.lower() for x in ["link na bio", "site", "loja", "desconto", "%"]):
        conv += 1
    if any(x in texto.lower() for x in ["até hoje", "até domingo", "limitado", "últimas unidades"]):
        conv += 1

    clareza = max(0.0, min(10.0, clareza))
    eng = max(0.0, min(10.0, eng))
    conv = max(0.0, min(10.0, conv))
    score = round((clareza + eng + conv) / 3, 1)

    return {
        "clareza": round(clareza, 1),
        "engajamento": round(eng, 1),
        "conversao": round(conv, 1),
        "score_final": score,
    }


def add_to_planner(
    dia: date,
    hora: time,
    plataforma: str,
    titulo: str,
    legenda: str,
    hashtags: List[str],
    score: float,
) -> None:
    item: Dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "date": dia,
        "time": hora,
        "plataforma": plataforma,
        "titulo": titulo,
        "legenda": legenda,
        "hashtags": hashtags,
        "score": score,
        "status": "planned",  # "planned" | "done"
    }
    st.session_state.planner_items.append(item)


def get_week_range(anchor: date) -> List[date]:
    weekday = anchor.weekday()  # 0 = Monday
    monday = anchor - timedelta(days=weekday)
    return [monday + timedelta(days=i) for i in range(7)]


def get_selected_task() -> Optional[Dict[str, Any]]:
    tid = st.session_state.selected_task_id
    if not tid:
        return None
    for item in st.session_state.planner_items:
        if item["id"] == tid:
            return item
    return None


# -----------------------------
# SIDEBAR – PLANO E PERFIL
# -----------------------------
st.sidebar.title("Plano e perfil")

plano = st.sidebar.selectbox("Plano", ["Starter", "Pro"], index=0)

st.sidebar.markdown("---")

if plano == "Starter":
    limite_hoje = 5
else:
    limite_hoje = 100

st.sidebar.caption("Gerações hoje (soft limit no front):")
st.sidebar.write(f"**{st.session_state.geracoes_hoje}/{limite_hoje}**")

st.sidebar.markdown("---")

marca = st.sidebar.text_input("Marca", value="Loukisses")
nicho = st.sidebar.text_input("Nicho/tema", value="Moda feminina")
tom = st.sidebar.selectbox("Tom de voz", ["premium", "casual", "profissional", "emocional"], index=0)
modo_copy = st.sidebar.selectbox("Modo de copy", ["Venda", "Storytelling", "Educacional"], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("**Métricas da conta (simuladas por agora)**")
seguidores = st.sidebar.number_input("Seguidores", min_value=0, value=1200, step=50)
eng_percent = st.sidebar.number_input("Engaj. %", min_value=0.0, max_value=100.0, value=3.4, step=0.1)
alcance_medio = st.sidebar.number_input("Alcance médio", min_value=0, value=1400, step=50)
st.sidebar.markdown(
    '<span class="cf-subtle">Integração real por link fica para o plano Pro+ numa futura versão.</span>',
    unsafe_allow_html=True,
)

# -----------------------------
# HEADER
# -----------------------------
st.markdown("## ContentForge v9.2 🍏")
st.markdown(
    "Gera conteúdo inteligente, organiza num planner semanal e, no plano **Pro**, "
    "acompanha a força de cada publicação."
)

tabs = st.tabs(["⚡ Gerar", "📅 Planner", "📊 Performance"])


# -----------------------------
# ABA 1 – GERAR
# -----------------------------
with tabs[0]:
    st.markdown("### ⚡ Geração inteligente de conteúdo")

    col_top1, _ = st.columns([2, 1])
    with col_top1:
        plataforma = st.selectbox("Plataforma", ["Instagram", "TikTok"], index=0)

    mensagem = st.text_input(
        "O que queres comunicar hoje?",
        value="Apresentação da nova coleção de Outono",
    )
    extra = st.text_area(
        "Informação extra (opcional)",
        value="10% de desconto no site até domingo.",
        height=80,
    )

    if plano == "Starter":
        st.markdown(
            """
            <div class="cf-subtle">
            🔒 <b>Dica Pro:</b> No plano Pro calculamos automaticamente a qualidade do copy,
            a probabilidade de engajamento e conversão para cada variação.
            </div>
            """,
            unsafe_allow_html=True,
        )

    gerar = st.button("⚡ Gerar agora", type="primary")

    if gerar:
        if st.session_state.geracoes_hoje >= limite_hoje:
            st.error(f"Limite diário de {limite_hoje} gerações atingido no plano {plano}.")
        else:
            with st.spinner("A IA está a pensar na melhor legenda para ti..."):
                variacoes = gerar_variacoes_legenda(
                    marca=marca,
                    nicho=nicho,
                    tom=tom,
                    modo_copy=modo_copy,
                    plataforma=plataforma,
                    mensagem=mensagem,
                    extra=extra,
                )

            if not variacoes:
                st.error("Não consegui interpretar a resposta da API. Tenta novamente.")
            else:
                st.session_state.geracoes_hoje += 1

                # Escolher recomendada (com base no score_final)
                best_idx = 0
                best_score = -1.0
                for i, v in enumerate(variacoes):
                    s = float(v.get("score_final", 0) or 0)
                    if v.get("recomendado") or s > best_score:
                        best_score = s
                        best_idx = i

                st.markdown("### Resultados")

                cols = st.columns(3)
                for idx, (col, var) in enumerate(zip(cols, variacoes)):
                    with col:
                        titulo = var.get("titulo_planner") or f"Variação {idx+1}"
                        legenda = var.get("legenda") or ""
                        hashtags_raw = var.get("hashtags") or []
                        hashtags = [f"#{h.strip('#')}" for h in hashtags_raw]
                        score_api = float(var.get("score_final", 0) or 0)

                        analise = analise_automatica_legenda(legenda)
                        final_score = round((score_api + analise["score_final"]) / 2, 1)

                        if idx == best_idx:
                            st.markdown(
                                '<div class="cf-badge-reco">✨ Nossa recomendação</div>',
                                unsafe_allow_html=True,
                            )

                        st.markdown(f"**{titulo}**")
                        st.write(legenda)

                        if hashtags:
                            st.markdown("**Hashtags sugeridas:**")
                            st.write(" ".join(hashtags))

                        if plano == "Pro":
                            st.markdown(
                                f"**Análise automática:** "
                                f"🧠 Score {final_score}/10 · "
                                f"💬 Engaj. {analise['engajamento']}/10 · "
                                f"💰 Conv. {analise['conversao']}/10"
                            )
                        else:
                            st.markdown(
                                f"**Análise automática (Pro):** 🔒 Pré-visualização — "
                                f"score estimado ~{final_score}/10"
                            )

                        dia = st.date_input(
                            "Dia",
                            value=date.today(),
                            key=f"dia_{idx}",
                        )
                        hora = st.time_input(
                            "Hora",
                            value=time(18, 0),
                            key=f"hora_{idx}",
                        )

                        if st.button("➕ Adicionar ao planner", key=f"add_{idx}"):
                            add_to_planner(
                                dia=dia,
                                hora=hora,
                                plataforma=plataforma.lower(),
                                titulo=titulo,
                                legenda=legenda,
                                hashtags=hashtags,
                                score=final_score,
                            )
                            st.success("Adicionado ao planner ✅")


# -----------------------------
# ABA 2 – PLANNER
# -----------------------------
with tabs[1]:
    st.markdown("### 📅 Planner de Conteúdo (v9.2)")
    st.markdown("_Vista semanal clean, com tarefas planeadas e concluídas._")

    col_nav1, col_nav2, col_anchor = st.columns([1, 1, 2])
    with col_nav1:
        if st.button("« Semana anterior"):
            st.session_state.anchor_date -= timedelta(days=7)
    with col_nav2:
        if st.button("Semana seguinte »"):
            st.session_state.anchor_date += timedelta(days=7)
    with col_anchor:
        new_anchor = st.date_input("Semana de referência", value=st.session_state.anchor_date)
        st.session_state.anchor_date = new_anchor

    semana = get_week_range(st.session_state.anchor_date)
    semana_label = f"Semana de {semana[0].strftime('%d/%m')} a {semana[-1].strftime('%d/%m')}"
    st.markdown(f"**{semana_label}**")

    cols_dias = st.columns(7)
    nomes_dias = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]

    for col_dia, nome, dia in zip(cols_dias, nomes_dias, semana):
        with col_dia:
            st.markdown(f"**{nome}**")
            st.caption(dia.strftime("%d/%m"))

            items_dia = sorted(
                [it for it in st.session_state.planner_items if it["date"] == dia],
                key=lambda x: x["time"],
            )

            if not items_dia:
                st.write('<span class="cf-subtle">Sem tarefas.</span>', unsafe_allow_html=True)
            else:
                for item in items_dia:
                    status = item["status"]
                    card_classes = "cf-card cf-card-done" if status == "done" else "cf-card"
                    html = f"""
                    <div class="{card_classes}">
                        <div style="font-size:0.8rem; opacity:0.75;">
                            {item['time'].strftime('%H:%M')} · {item['plataforma'].capitalize()}
                        </div>
                        <div style="font-weight:600; margin-top:0.15rem;">
                            {item['titulo']}
                        </div>
                        <div style="font-size:0.8rem; margin-top:0.2rem;">
                            Score: {item['score']}/10
                            {' · ✅ Concluído' if status == 'done' else ''}
                        </div>
                    </div>
                    """
                    st.markdown(html, unsafe_allow_html=True)

                    col_bt1, col_bt2 = st.columns(2)
                    with col_bt1:
                        if st.button("👁 Ver detalhes", key=f"det_{item['id']}"):
                            st.session_state.selected_task_id = item["id"]
                    with col_bt2:
                        if status == "planned":
                            if st.button("✅ Concluir", key=f"done_{item['id']}"):
                                item["status"] = "done"
                                st.success("Marcado como concluído ✅")
                        else:
                            st.write('<span class="cf-subtle">Já concluído</span>', unsafe_allow_html=True)

    st.markdown("---")
    sel = get_selected_task()
    if sel:
        st.markdown("### 🔍 Detalhes da tarefa selecionada")
        colA, colB = st.columns([2, 1])
        with colA:
            st.markdown(f"**{sel['titulo']}**")
            st.caption(
                f"{sel['date'].strftime('%d/%m/%Y')} · {sel['time'].strftime('%H:%M')} · "
                f"{sel['plataforma'].capitalize()}"
            )
            st.write(sel["legenda"])

            if sel["hashtags"]:
                st.markdown("**Hashtags:**")
                st.write(" ".join(sel["hashtags"]))

        with colB:
            st.markdown("**Estado atual:**")
            if sel["status"] == "done":
                st.success("Concluído ✅")
            else:
                st.info("Planeado")

            if sel["status"] == "planned":
                if st.button("✅ Marcar como concluído", key="det_mark_done"):
                    sel["status"] = "done"
                    st.success("Marcado como concluído ✅")
            else:
                st.write('<span class="cf-subtle">Já está concluído.</span>', unsafe_allow_html=True)

            if st.button("🗑 Remover do planner", key="det_remove"):
                st.session_state.planner_items = [
                    it for it in st.session_state.planner_items if it["id"] != sel["id"]
                ]
                st.session_state.selected_task_id = None
                st.success("Tarefa removida.")

        if st.button("Fechar detalhes"):
            st.session_state.selected_task_id = None


# -----------------------------
# ABA 3 – PERFORMANCE
# -----------------------------
with tabs[2]:
    st.markdown("### 📊 Performance (v9.2)")

    if plano != "Pro":
        st.markdown(
            """
            <div class="cf-badge-lock">
            🔒 Disponível no plano Pro. Desbloqueia métricas, previsões e recomendações de horário.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.info("Altera o plano na barra lateral para 'Pro' para ver a aba Performance completa.")
    else:
        concluidos = [it for it in st.session_state.planner_items if it["status"] == "done"]

        if not concluidos:
            st.info("Ainda não tens posts marcados como concluídos. Marca um post como concluído no planner para começar.")
        else:
            scores = [it["score"] for it in concluidos]
            media_score = round(statistics.mean(scores), 2) if scores else 0.0

            horas = [it["time"].strftime("%H:00") for it in concluidos]
            hora_recomendada = max(set(horas), key=horas.count)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Posts concluídos", len(concluidos))
            with col2:
                st.metric("Score médio da IA", media_score)
            with col3:
                st.metric("Hora recomendada", hora_recomendada)

            st.markdown(
                '<div class="cf-subtle">⚙️ Precisão da IA aumenta com o nº de postagens concluídas.</div>',
                unsafe_allow_html=True,
            )

            st.markdown("---")
            st.markdown("#### Últimos posts concluídos")

            for it in sorted(concluidos, key=lambda x: (x["date"], x["time"]), reverse=True)[:10]:
                st.markdown(
                    f"**{it['date'].strftime('%d/%m')} {it['time'].strftime('%H:%M')} · "
                    f"{it['plataforma'].capitalize()}** — {it['titulo']}  \n"
                    f"Score: **{it['score']}/10** · Estado: ✅ Concluído"
                )
