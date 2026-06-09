import streamlit as st
import anthropic
import random
import json
import time
import plotly.graph_objects as go
from deap import base, creator, tools, algorithms
import numpy as np

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prompt Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0a0f;
    color: #e8e8f0;
}

.stApp { background-color: #0a0a0f; }

h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }

.hero-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.6rem;
    font-weight: 600;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #a78bfa 0%, #60a5fa 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
}

.hero-sub {
    font-size: 0.95rem;
    color: #6b7280;
    font-weight: 300;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

.card {
    background: #111118;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}

.gen-badge {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    background: #1e1e2e;
    color: #a78bfa;
    border: 1px solid #2d2d4e;
    border-radius: 4px;
    padding: 2px 8px;
    margin-bottom: 0.6rem;
}

.score-big {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.8rem;
    font-weight: 600;
    color: #34d399;
    line-height: 1;
}

.score-label {
    font-size: 0.72rem;
    color: #4b5563;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.prompt-box {
    background: #0d0d14;
    border: 1px solid #1e1e2e;
    border-left: 3px solid #a78bfa;
    border-radius: 6px;
    padding: 0.9rem 1.1rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    color: #c4b5fd;
    line-height: 1.6;
    white-space: pre-wrap;
    margin: 0.5rem 0;
}

.best-prompt-box {
    border-left-color: #34d399;
    color: #6ee7b7;
}

.individual-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.5rem 0.7rem;
    border-radius: 6px;
    margin-bottom: 0.3rem;
    background: #0d0d14;
    border: 1px solid #1a1a28;
    font-size: 0.82rem;
}

.score-pill {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    padding: 2px 8px;
    border-radius: 999px;
    white-space: nowrap;
}

.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #2563eb);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    font-weight: 600;
    padding: 0.6rem 1.8rem;
    cursor: pointer;
    letter-spacing: 0.04em;
    transition: opacity 0.2s;
    width: 100%;
}
.stButton > button:hover { opacity: 0.85; border: none; }

.stTextArea textarea {
    background: #111118 !important;
    border: 1px solid #2d2d4e !important;
    border-radius: 8px !important;
    color: #e8e8f0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.83rem !important;
}
.stTextArea textarea:focus { border-color: #7c3aed !important; box-shadow: none !important; }

.stTextInput input {
    background: #111118 !important;
    border: 1px solid #2d2d4e !important;
    border-radius: 8px !important;
    color: #e8e8f0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.83rem !important;
}

label, .stTextArea label, .stTextInput label {
    color: #9ca3af !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    font-weight: 500;
}

.stSlider > div { color: #9ca3af; }

div[data-testid="metric-container"] {
    background: #111118;
    border: 1px solid #1e1e2e;
    border-radius: 10px;
    padding: 0.8rem 1rem;
}

.status-line {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #6b7280;
    padding: 0.3rem 0;
}

.evolving-indicator {
    display: inline-block;
    width: 8px; height: 8px;
    background: #34d399;
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 1s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

.divider { border-top: 1px solid #1e1e2e; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Anthropic client ──────────────────────────────────────────────────────────
@st.cache_resource
def get_client():
    return anthropic.Anthropic()

client = get_client()

# ── Genetic Algorithm helpers ─────────────────────────────────────────────────

MUTATION_STRATEGIES = [
    "Make it more specific and detailed",
    "Add a step-by-step instruction to think before answering",
    "Make it more concise, remove redundant words",
    "Add a role/persona (e.g. 'You are an expert...')",
    "Add output format instructions",
    "Add examples or few-shot context",
    "Make the tone more authoritative",
    "Add constraints about what NOT to do",
    "Rewrite using active voice and direct imperatives",
    "Add 'think step by step' or chain-of-thought trigger",
]

def mutate_prompt(prompt: str) -> str:
    """Use Claude to mutate a prompt using a random strategy."""
    strategy = random.choice(MUTATION_STRATEGIES)
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=400,
        messages=[{
            "role": "user",
            "content": f"""You are a prompt engineering expert. Improve this prompt using this strategy: {strategy}

Original prompt:
{prompt}

Return ONLY the improved prompt text. No explanation, no preamble, no quotes."""
        }]
    )
    return response.content[0].text.strip()

def crossover_prompts(prompt_a: str, prompt_b: str) -> tuple[str, str]:
    """Crossover two prompts to produce two children."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=600,
        messages=[{
            "role": "user",
            "content": f"""You are a prompt engineering expert. Combine the best elements of these two prompts to create two new improved variants.

Prompt A:
{prompt_a}

Prompt B:
{prompt_b}

Return ONLY a JSON object like this (no markdown, no explanation):
{{"child1": "...", "child2": "..."}}"""
        }]
    )
    raw = response.content[0].text.strip()
    try:
        data = json.loads(raw)
        return data["child1"], data["child2"]
    except Exception:
        return prompt_a, prompt_b

def score_prompt(prompt: str, task_description: str, test_input: str) -> float:
    """LLM-as-judge: run the prompt and score 1-10."""
    # Step 1: Run the prompt
    run_response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": f"{prompt}\n\nInput: {test_input}"
        }]
    )
    output = run_response.content[0].text.strip()

    # Step 2: Judge the output
    judge_response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": f"""You are an objective prompt quality judge.

Task the prompt should accomplish: {task_description}

Prompt used:
{prompt}

Output produced:
{output}

Score how well the prompt + output accomplishes the task. Consider: relevance, clarity, completeness, and whether the prompt itself is well-engineered.

Respond ONLY with a JSON object: {{"score": <number 1-10>, "reason": "<one sentence>"}}"""
        }]
    )
    raw = judge_response.content[0].text.strip()
    try:
        data = json.loads(raw)
        return float(data["score"]), data.get("reason", ""), output
    except Exception:
        return 5.0, "Parse error", output

# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []          # list of {gen, population: [{prompt, score, reason}]}
if "best_prompt" not in st.session_state:
    st.session_state.best_prompt = None
if "best_score" not in st.session_state:
    st.session_state.best_score = 0.0
if "running" not in st.session_state:
    st.session_state.running = False

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">⚡ Prompt Optimizer</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Genetic algorithm · LLM-as-judge · Real-time evolution</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1.6], gap="large")

with col_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    initial_prompt = st.text_area(
        "Initial Prompt",
        placeholder="Write a summary of the following text:",
        height=100,
        key="initial_prompt"
    )

    task_description = st.text_area(
        "What should this prompt accomplish?",
        placeholder="Summarize text clearly and concisely, capturing all key points",
        height=80,
        key="task_desc"
    )

    test_input = st.text_area(
        "Test Input (used for evaluation)",
        placeholder="Paste sample input your prompt will receive...",
        height=100,
        key="test_input"
    )

    c1, c2 = st.columns(2)
    with c1:
        pop_size = st.slider("Population size", 3, 8, 4)
    with c2:
        n_generations = st.slider("Generations", 2, 6, 3)

    st.markdown('</div>', unsafe_allow_html=True)

    run_btn = st.button("🧬 Evolve Prompts", use_container_width=True)

with col_right:
    # Score over generations chart
    chart_placeholder = st.empty()
    status_placeholder = st.empty()
    results_placeholder = st.empty()

# ── Draw chart ────────────────────────────────────────────────────────────────
def draw_chart(history):
    if not history:
        return
    
    gens = [h["gen"] for h in history]
    best_scores = [max(ind["score"] for ind in h["population"]) for h in history]
    avg_scores  = [np.mean([ind["score"] for ind in h["population"]]) for h in history]
    worst_scores = [min(ind["score"] for ind in h["population"]) for h in history]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=gens, y=worst_scores,
        mode="lines", name="Worst",
        line=dict(color="#374151", width=1.5, dash="dot"),
        fill=None
    ))
    fig.add_trace(go.Scatter(
        x=gens, y=avg_scores,
        mode="lines+markers", name="Average",
        line=dict(color="#60a5fa", width=2),
        marker=dict(size=5),
        fill="tonexty", fillcolor="rgba(96,165,250,0.05)"
    ))
    fig.add_trace(go.Scatter(
        x=gens, y=best_scores,
        mode="lines+markers", name="Best",
        line=dict(color="#34d399", width=2.5),
        marker=dict(size=7, symbol="circle"),
        fill="tonexty", fillcolor="rgba(52,211,153,0.07)"
    ))

    fig.update_layout(
        paper_bgcolor="#0a0a0f",
        plot_bgcolor="#0d0d14",
        font=dict(family="IBM Plex Mono", color="#9ca3af", size=11),
        margin=dict(l=10, r=10, t=30, b=10),
        title=dict(text="Fitness Over Generations", font=dict(color="#e8e8f0", size=13)),
        xaxis=dict(
            title="Generation", gridcolor="#1e1e2e", showline=False,
            tickfont=dict(size=10), dtick=1
        ),
        yaxis=dict(
            title="Score /10", gridcolor="#1e1e2e", range=[0, 10.5],
            showline=False, tickfont=dict(size=10)
        ),
        legend=dict(
            bgcolor="#111118", bordercolor="#1e1e2e", borderwidth=1,
            font=dict(size=10)
        ),
        height=280
    )
    chart_placeholder.plotly_chart(fig, use_container_width=True)

def draw_results(history, best_prompt, best_score):
    with results_placeholder.container():
        if best_prompt:
            st.markdown(f"""
            <div class="card">
                <div class="score-label">Best prompt found</div>
                <div class="score-big">{best_score:.1f}<span style="font-size:1rem;color:#6b7280">/10</span></div>
                <div class="prompt-box best-prompt-box">{best_prompt}</div>
            </div>
            """, unsafe_allow_html=True)

        if history:
            latest = history[-1]
            st.markdown(f'<div style="font-family:IBM Plex Mono;font-size:0.78rem;color:#6b7280;margin-bottom:0.5rem;">GENERATION {latest["gen"]} POPULATION</div>', unsafe_allow_html=True)
            sorted_pop = sorted(latest["population"], key=lambda x: x["score"], reverse=True)
            for i, ind in enumerate(sorted_pop):
                color = "#34d399" if i == 0 else "#60a5fa" if i == 1 else "#6b7280"
                score_display = f'<span class="score-pill" style="background:{color}20;color:{color};border:1px solid {color}40">{ind["score"]:.1f}</span>'
                st.markdown(f"""
                <div class="individual-row">
                    {score_display}
                    <span style="color:#9ca3af;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{ind["prompt"][:120]}{'...' if len(ind["prompt"])>120 else ''}</span>
                </div>
                """, unsafe_allow_html=True)
                if ind.get("reason"):
                    st.markdown(f'<div style="font-size:0.72rem;color:#4b5563;padding:0 0.7rem 0.3rem 3rem;font-style:italic">{ind["reason"]}</div>', unsafe_allow_html=True)

# ── Run evolution ─────────────────────────────────────────────────────────────
if run_btn:
    if not initial_prompt or not task_description or not test_input:
        st.error("Please fill in all three fields.")
    else:
        st.session_state.history = []
        st.session_state.best_prompt = None
        st.session_state.best_score = 0.0

        # ── Initialise population ─────────────────────────────────────────
        status_placeholder.markdown(
            '<div class="status-line"><span class="evolving-indicator"></span>Generating initial population...</div>',
            unsafe_allow_html=True
        )

        population = [initial_prompt]
        for i in range(pop_size - 1):
            status_placeholder.markdown(
                f'<div class="status-line"><span class="evolving-indicator"></span>Mutating initial prompt ({i+1}/{pop_size-1})...</div>',
                unsafe_allow_html=True
            )
            population.append(mutate_prompt(initial_prompt))

        # ── Generational loop ─────────────────────────────────────────────
        for gen in range(1, n_generations + 1):
            scored_population = []

            for idx, prompt in enumerate(population):
                status_placeholder.markdown(
                    f'<div class="status-line"><span class="evolving-indicator"></span>Gen {gen}/{n_generations} · Scoring individual {idx+1}/{len(population)}...</div>',
                    unsafe_allow_html=True
                )
                score, reason, output = score_prompt(prompt, task_description, test_input)
                scored_population.append({"prompt": prompt, "score": score, "reason": reason, "output": output})

                # Update best
                if score > st.session_state.best_score:
                    st.session_state.best_score = score
                    st.session_state.best_prompt = prompt

            # Sort by score
            scored_population.sort(key=lambda x: x["score"], reverse=True)
            st.session_state.history.append({"gen": gen, "population": scored_population})

            draw_chart(st.session_state.history)
            draw_results(st.session_state.history, st.session_state.best_prompt, st.session_state.best_score)

            if gen == n_generations:
                break

            # ── Selection + crossover + mutation ─────────────────────────
            status_placeholder.markdown(
                f'<div class="status-line"><span class="evolving-indicator"></span>Gen {gen} done · Breeding next generation...</div>',
                unsafe_allow_html=True
            )

            # Elite: keep top 2
            next_population = [scored_population[0]["prompt"], scored_population[1]["prompt"]]

            # Crossover top individuals to fill rest
            while len(next_population) < pop_size:
                # Tournament selection
                a = random.choice(scored_population[:max(2, len(scored_population)//2)])["prompt"]
                b = random.choice(scored_population[:max(2, len(scored_population)//2)])["prompt"]
                c1_prompt, c2_prompt = crossover_prompts(a, b)
                # Mutate children
                if random.random() < 0.6:
                    c1_prompt = mutate_prompt(c1_prompt)
                if random.random() < 0.6:
                    c2_prompt = mutate_prompt(c2_prompt)
                next_population.extend([c1_prompt, c2_prompt])

            population = next_population[:pop_size]

        status_placeholder.markdown(
            f'<div class="status-line" style="color:#34d399">✓ Evolution complete · Best score: {st.session_state.best_score:.1f}/10</div>',
            unsafe_allow_html=True
        )

elif st.session_state.history:
    draw_chart(st.session_state.history)
    draw_results(st.session_state.history, st.session_state.best_prompt, st.session_state.best_score)
