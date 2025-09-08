# Simple LLM API Client

## MoE Routing Noise Experiment: How to Use

### Overview
This repo includes a minimal Mixture-of-Experts (MoE) language model and a scripted experiment to study routing noise. It runs 18 jobs (3 noise levels × 2 load-balancing settings × 3 seeds) and saves results to CSV plus summary markdowns.

### Prerequisites
- Python 3.12
- GPU optional (CUDA preferred). CPU works but is slower
- Install deps:
```bash
python -m venv simple_llm_env
source simple_llm_env/bin/activate
pip install -r requirements.txt
```

### Run the experiment
This executes all 18 runs end-to-end and saves a CSV with results:
```bash
python llm.py
```

What happens:
- Downloads/caches data and tokenizer on first run
- Iterates through noise_std ∈ {0.0, 0.1, 0.5}, load_balancing_weight ∈ {0.01, 0.0}, seeds ∈ {42,43,44}
- Trains each run for 400 steps (≈1.6 minutes on an RTX 4090)
- Writes `moe_routing_noise_experiment_results.csv` and prints a summary table

### Key files
- `llm.py` — model, training loop, and experiment runner
- `experiment_tutorial.md` — step-by-step tutorial report
- `experiment_analysis.md` — concise analysis and recommendations
- `moe_routing_noise_experiment_results.csv` — raw results from 18 runs
- `data_cache/` — cached tokenized data to speed up re-runs

### Configurable parameters
Edit `llm.py` to change base settings:
- Training length: `MoEModelConfig.max_steps` (default 400)
- Periodic evals: `MoEModelConfig.eval_every` (set ≤ max_steps for mid-run evals)
- Routing noise: `MoEModelConfig.noise_std` (overridden per condition)
- Load balancing: `MoEModelConfig.load_balancing_weight` (overridden per condition)
- Seeds: list in `seeds = [42, 43, 44]`
- Data slice: `num_documents`, `max_tokens`, `max_seq_len`
- Model size: `d_model`, `n_layers`, `d_ff`, `n_heads`, `num_experts`, `expert_top_k`

### Outputs and where to find them
- CSV: `moe_routing_noise_experiment_results.csv`
  - Columns: seed, noise_std, load_balancing_weight, val_loss, val_accuracy, val_perplexity, training_time_minutes, peak_memory_gb, best_val_perplexity, expert_usage_cv
- Markdown reports:
  - `experiment_tutorial.md`: full tutorial with per-seed tables
  - `experiment_analysis.md`: short analysis with recommendations

### Tips
- Current setup uses `max_steps=400` and `eval_every=500`, so best PPL equals final PPL for each run. To see mid-run evals, set `eval_every=100` (or any value ≤ 400).
- Expert usage CV is computed over the entire validation set at the end and aggregated across layers.
- Data and tokenizer are cached to `data_cache/` to avoid repeated downloads and tokenization.

### Troubleshooting
- If you see an IndentationError around CSV writing, ensure the `with open(...):` block includes indented `fieldnames` and writer lines (this is already fixed in `main`).
- If you modify `MoEModelConfig`, only pass fields defined in the dataclass (avoid passing computed fields like `d_k`).
- If runs are too slow, reduce `num_documents` or `max_tokens`, or run fewer conditions/seeds.

### Reproducing analysis in Python
```python
import pandas as pd
df = pd.read_csv('moe_routing_noise_experiment_results.csv')
summary = df.groupby(['noise_std','load_balancing_weight']).agg(
    mean_ppl=('best_val_perplexity','mean'),
    std_ppl=('best_val_perplexity','std'),
    mean_cv=('expert_usage_cv','mean'),
    n=('seed','count')
)
print(summary)
```

### Current limitations
- Short training (≈1.6 minutes/run) — good for iteration, but longer runs are recommended for final conclusions.

## Using simple_llm.py

### Overview
`simple_llm.py` is a lightweight CLI to call hosted LLMs (Gemini, Kimi/Novita) and to run multi-step "chains" that generate research plans or other artifacts from prompt files.

### Setup
- Set API keys in your environment:
  - `GEMINI_API_KEY` for Gemini
  - `NOVITA_API_KEY` for Kimi (Novita OpenAI-compatible)
- Install deps (see above) and activate the venv.

### Direct model calls
```bash
python simple_llm.py gemini "Write a haiku about MoE routing"
python simple_llm.py kimi "Summarize mixture-of-experts in 5 bullets"
```
Models supported: `gemini`, `kimi`.

### Run a multi-step chain
The built-in chain uses a markdown prompt and current code context from `llm.py`:
```bash
python simple_llm.py chain research_prompt.md
```
This will:
- Load `research_prompt.md` (your task)
- Read `llm.py` as code context and `research_system_prompt.md` as system constraints
- Run 4 steps (Research_Plan → Critical_Review → Critique_Analysis → Final_Plan)
- Save outputs to `step1_*.md`, `step2_*.md`, `step3_*.md`, `step4_*.md`

### Customize prompts and steps
Edit the `steps` array inside `simple_llm.py` to:
- Change a step's `model` (`gemini` or `kimi`)
- Edit the `prompt` text; you can reference variables:
  - `{system_prompt}` from `research_system_prompt.md`
  - `{code_content}` from `llm.py` (or change the file path in code)
  - `{user_input}` from the markdown file passed on CLI
  - `{stepN_result}` to use outputs from previous steps
- Add/remove steps by modifying the list.

Example tweak (shorter critique analysis on Kimi):
```python
{
  "title": "Critique_Analysis",
  "model": "kimi",
  "prompt": "Analyze and condense key objections from {step2_result} in ≤300 words."
}
```

### Generate your own experiments with the chain
1) Create a new prompt file, e.g. `my_experiment_prompt.md`, describing your objective, variables to sweep, metrics, and constraints.
2) Run: `python simple_llm.py chain my_experiment_prompt.md`
3) The chain produces a plan, critique, counter-analysis, and final plan tailored to your prompt.

### Tips
- To change the code context, update `llm.read_code_file("llm.py")` to point to a different file.
- You can inject custom variables before running the chain by setting `llm.variables['my_var'] = 'value'` and then using `{my_var}` in prompts.
- For reproducible outputs, keep your prompt files versioned.

A minimal Python client for calling different LLM APIs (Gemini, Kimi) with research methodology chain support.

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Setup API Keys
Create a `.env` file in the project root:
```bash
# Copy this content to a new .env file
GEMINI_API_KEY=your_gemini_api_key_here
NOVITA_API_KEY=your_novita_api_key_here
```

**API Key Sources:**
- **Gemini**: Get from [Google AI Studio](https://aistudio.google.com/app/apikey)
- **Kimi (via Novita)**: Get from your Novita account

## 📖 Usage

### Direct Model Calls
```bash
# Call Gemini
python simple_llm.py gemini "Hello world"

# Call Kimi
python simple_llm.py kimi "Create a function"
```

### Create Markdown Prompt File
Create a `.md` file with your detailed prompt:

```markdown
# Project: E-commerce Platform

## Requirements
Build a full-stack e-commerce application with the following features:

### Frontend
- React with TypeScript
- Modern UI with Tailwind CSS
- Shopping cart functionality
- User authentication flow

### Backend
- Node.js with Express
- MongoDB database
- JWT authentication
- Payment integration with Stripe

### Features
- Product catalog with search and filters
- User accounts and profiles
- Order management
- Admin dashboard
- Responsive mobile design
```

### Run Research Methodology Chain
```bash
# Use the provided research prompt template
python simple_llm.py chain research_prompt.md

# Or create your own markdown file with research content
python simple_llm.py chain my_research_idea.md
```

The chain automatically reads your existing `llm.py` MoE implementation and combines it with your research idea for comprehensive analysis.

## 🔬 Research Methodology Chain

The chain implements a rigorous 4-step research methodology with **system constraints** automatically included in each LLM call:

### System Constraints (from research_system_prompt.md):
- Use only SmolLM dataset (no data source changes)
- Keep experiments short (2-10 minutes on 1x4090)
- Maintain fair experimental design
- No model size increases or longer training

### 4-Step Process:

1. **Research Plan** (Kimi): Creates initial concise research plan based on existing MoE code and research idea
2. **Critical Review** (Gemini): Identifies methodological flaws, unfair comparisons, and suggests improvements
3. **Critique Analysis** (Kimi): Balances the critique with counter-arguments and feasibility considerations
4. **Final Plan** (Gemini): Produces focused, fair research plan implementing key improvements

Each step saves its output to a file:
- `step1_research_plan.txt`
- `step2_critical_review.txt`
- `step3_critique_analysis.txt`
- `step4_final_plan.txt`

## 🛠️ Customization

To modify the chain, edit the `steps` list in `simple_llm.py`:

```python
steps = [
    {
        "title": "Your Step Name",
        "model": "gemini",  # or "kimi"
        "prompt": "Your prompt here with {variables}"
    }
]
```

### Variables
- `{user_input}`: The input you provide
- `{step1_result}`: Result from step 1
- `{step2_result}`: Result from step 2
- `{system_prompt}`: System constraints from research_system_prompt.md
- etc.

## 📋 Requirements

- Python 3.8+
- API keys for desired models
- Dependencies: `google-genai`, `openai`, `python-dotenv`

## 🎯 Features

- ✅ Simple API calls to multiple models
- ✅ Research methodology chain with system constraints
- ✅ Variable substitution
- ✅ Automatic file saving
- ✅ Streaming responses
- ✅ Error handling
