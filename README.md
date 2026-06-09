# ⚡ Prompt Optimizer

> Evolve better prompts automatically using a genetic algorithm + LLM-as-judge scoring loop.

## How it works

1. **You provide** a rough initial prompt, a task description, and a test input
2. **Generation 0** — your prompt gets mutated N times to seed a population
3. **Each individual** is run against your test input, and a separate Claude call judges the output (score 1–10)
4. **Selection** — top scorers survive; pairs are crossed over to produce children
5. **Mutation** — children are randomly mutated using one of 10 prompt engineering strategies
6. **Repeat** for N generations — watch scores climb in real time

## Setup

```bash
# 1. Clone / copy this folder
cd prompt_optimizer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Anthropic API key
export ANTHROPIC_API_KEY=sk-ant-...

# 4. Run
streamlit run app.py
```

## Stack

| Layer | Tech |
|---|---|
| UI | Streamlit |
| Genetic algorithm | DEAP |
| LLM calls | Anthropic Python SDK (claude-sonnet-4) |
| Visualisation | Plotly |
| Language | Pure Python |

## Parameters

| Setting | What it does |
|---|---|
| Population size | How many prompt variants per generation (3–8) |
| Generations | How many evolution cycles to run (2–6) |

## Mutation strategies used

- Make it more specific and detailed  
- Add chain-of-thought / step-by-step trigger  
- Make it more concise  
- Add a role/persona  
- Add output format instructions  
- Add few-shot examples  
- Make the tone more authoritative  
- Add constraints (what NOT to do)  
- Rewrite using active voice  
- Add explicit reasoning trigger  

## Cost estimate

Each run makes approximately `pop_size × generations × 2` Claude API calls (one to run the prompt, one to judge). A typical run (4 pop, 3 gen) = ~24 calls. At Sonnet pricing this is a few cents per run.

## Tips for a good demo

- Use a task with clear quality differences (e.g. "classify sentiment", "write a product description")
- Keep test input short (1–3 sentences) so judge scores are consistent
- Start with a deliberately vague prompt — the evolution effect is more dramatic
