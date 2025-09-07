# Simple LLM API Client

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
