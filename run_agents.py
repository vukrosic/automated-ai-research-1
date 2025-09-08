#!/usr/bin/env python3
"""
Simple LLM API Client with Chain Support
Call different AI models and create simple chains
"""

import os
import sys
from typing import Dict, List, Any
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from google import genai
from google.genai import types as gemini_types

# Load environment variables
load_dotenv()

class SimpleLLM:
    def __init__(self):
        self.models = {
            "gemini": {
                "client": None,
                "api_key": os.getenv("GEMINI_API_KEY"),
                "type": "gemini"
            },
            "kimi": {
                "client": None,
                "api_key": os.getenv("NOVITA_API_KEY"),
                "type": "novita",
                "model_name": "moonshotai/kimi-k2-0905"
            }
        }
        self._init_clients()
        self.variables = {}

    def _init_clients(self):
        """Initialize API clients."""
        # Gemini
        if self.models["gemini"]["api_key"]:
            try:
                self.models["gemini"]["client"] = genai.Client(
                    api_key=self.models["gemini"]["api_key"]
                )
                print("✅ Gemini ready")
            except Exception as e:
                print(f"❌ Gemini failed: {e}")

        # Kimi (Novita)
        if self.models["kimi"]["api_key"]:
            try:
                self.models["kimi"]["client"] = OpenAI(
                    base_url="https://api.novita.ai/openai",
                    api_key=self.models["kimi"]["api_key"]
                )
                print("✅ Kimi ready")
            except Exception as e:
                print(f"❌ Kimi failed: {e}")

    def call_model(self, model_name: str, prompt: str, **kwargs) -> str:
        """Call a specific model with a prompt."""
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")

        model_config = self.models[model_name]
        client = model_config["client"]

        if not client:
            raise ValueError(f"Model {model_name} not initialized")

        if model_config["type"] == "gemini":
            return self._call_gemini(client, prompt, **kwargs)
        elif model_config["type"] == "novita":
            return self._call_novita(client, model_config["model_name"], prompt, **kwargs)

    def _call_gemini(self, client, prompt: str, **kwargs) -> str:
        """Call Gemini API."""
        contents = [gemini_types.Content(
            role="user",
            parts=[gemini_types.Part.from_text(text=prompt)]
        )]

        config = gemini_types.GenerateContentConfig(
            thinking_config=gemini_types.ThinkingConfig(thinking_budget=-1)
        )

        response_text = ""
        for chunk in client.models.generate_content_stream(
            model="gemini-2.5-pro",
            contents=contents,
            config=config
        ):
            if chunk.text:
                print(chunk.text, end="", flush=True)
                response_text += chunk.text
        print()
        return response_text

    def _call_novita(self, client, model_name: str, prompt: str, **kwargs) -> str:
        """Call Novita API."""
        messages = [{"role": "user", "content": prompt}]

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True,
            max_tokens=kwargs.get("max_tokens", 1000)
        )

        response_text = ""
        for chunk in response:
            content = chunk.choices[0].delta.content or ""
            print(content, end="", flush=True)
            response_text += content
        print()
        return response_text

    def substitute_vars(self, text: str) -> str:
        """Substitute variables in text."""
        for var, value in self.variables.items():
            text = text.replace(f"{{{var}}}", value)
        return text

    def read_markdown_file(self, file_path: str) -> str:
        """Read content from a markdown file."""
        if not file_path.endswith('.md'):
            raise ValueError("File must have .md extension")

        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        return Path(file_path).read_text().strip()

    def read_code_file(self, file_path: str) -> str:
        """Read content from a Python file."""
        if not file_path.endswith('.py'):
            raise ValueError("File must have .py extension")

        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content = Path(file_path).read_text()
        return f"```python\n{content}\n```"

    def run_chain(self, steps: List[Dict[str, str]], user_input: str = "") -> None:
        """Run a chain of LLM calls."""
        if user_input:
            self.variables['user_input'] = user_input

        print("🚀 Starting Chain")
        print("=" * 40)

        for i, step in enumerate(steps, 1):
            print(f"\n🔄 Step {i}: {step['title']}")
            print("-" * 30)

            # Substitute variables in prompt
            prompt = self.substitute_vars(step['prompt'])

            try:
                # Call the model
                result = self.call_model(step['model'], prompt)

                # Store result for next steps
                var_name = f"step{i}_result"
                self.variables[var_name] = result

                # Save to file
                filename = f"step{i}_{step['title'].lower().replace(' ', '_')}.md"
                Path(filename).write_text(result)
                print(f"💾 Saved to: {filename}")

            except Exception as e:
                print(f"❌ Step failed: {e}")
                break

        print("\n✅ Chain completed!")


def main():
    """Command line interface."""
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python simple_llm.py <model> <prompt>")
        print("  python simple_llm.py chain <markdown_file.md>")
        print("\nModels: gemini, kimi")
        print("\nExamples:")
        print("  python simple_llm.py gemini 'Hello world'")
        print("  python simple_llm.py kimi 'Create a function'")
        print("  python simple_llm.py chain prompt.md")
        sys.exit(1)

    command = sys.argv[1]

    if command in ["gemini", "kimi"]:
        # Direct model call
        model_name = command
        prompt = " ".join(sys.argv[2:])

        try:
            llm = SimpleLLM()
            print(f"🤖 Calling {model_name}...")
            print("-" * 40)
            llm.call_model(model_name, prompt)
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)

    elif command == "chain":
        # Run research methodology chain with markdown file
        if len(sys.argv) < 3:
            print("❌ Please provide a markdown file path")
            print("Example: python simple_llm.py chain research_prompt.md")
            sys.exit(1)

        file_path = sys.argv[2]

        try:
            llm = SimpleLLM()

            # Read markdown prompt file
            user_input = llm.read_markdown_file(file_path)
            print(f"📄 Read prompt from: {file_path}")
            print(f"📝 Prompt length: {len(user_input)} characters")

            # Read existing code from llm.py
            code_content = llm.read_code_file("llm.py")
            print(f"📄 Read code from: llm.py")
            print(f"📝 Code length: {len(code_content)} characters")

            # Read system prompt
            system_prompt = llm.read_markdown_file("research_system_prompt.md")
            print(f"📄 Read system prompt from: research_system_prompt.md")
            print(f"📝 System prompt length: {len(system_prompt)} characters")
            print("-" * 50)

            # Set up variables
            llm.variables['user_input'] = user_input
            llm.variables['code_content'] = code_content
            llm.variables['research_paper'] = "{RESEARCH_PAPER_CONTENT_PLACEHOLDER}"
            llm.variables['system_prompt'] = system_prompt

            # Define research methodology chain
            steps = [
                {
                    "title": "Research_Plan",
                    "model": "gemini",
                    "prompt": """SYSTEM CONSTRAINTS: {system_prompt}

Based on the existing MoE implementation in {code_content} and the research idea in {research_paper}, create a CONCISE research plan (max 800 words).

Focus on:
- Clear research question
- Novel contribution
- Experimental design (fair comparisons)
- Success metrics
- Timeline (keep it realistic and short - experiments should take 2-10 minutes on 1x4090)

IMPORTANT CONSTRAINTS:
- Do not change data source - only use SmolLM dataset
- Do not make model bigger or train longer
- Keep experiments fair and methodologically sound
- Point out potential methodological issues in the existing code that could affect fairness"""
                },
                {
                    "title": "Critical_Review",
                    "model": "kimi",
                    "prompt": """SYSTEM CONSTRAINTS: {system_prompt}

Review the research plan in {step1_result} for critical methodological flaws:

FOCUS ON:
- Unfair experimental comparisons
- Statistical validity issues
- Potential biases in evaluation
- Missing controls
- Overstated claims
- Adherence to system constraints (SmolLM dataset, model size limits, training time limits)

SUGGEST SPECIFIC IMPROVEMENTS to make the research more rigorous and fair while respecting the system constraints. Be direct but constructive. Keep critique concise (max 600 words)."""
                },
                {
                    "title": "Critique_Analysis",
                    "model": "gemini",
                    "prompt": """SYSTEM CONSTRAINTS: {system_prompt}

Analyze the critique in {step2_result} and provide counter-arguments where appropriate:

ADDRESS:
- Are the criticisms valid given the system constraints?
- Are there over-cautions that ignore practical limitations?
- What aspects of the critique are most important?
- What can be realistically implemented within the constraints?

Be balanced - acknowledge valid points while defending reasonable methodological choices that work within the system constraints. Keep analysis focused (max 500 words)."""
                },
                {
                    "title": "Final_Plan",
                    "model": "gemini",
                    "prompt": """SYSTEM CONSTRAINTS: {system_prompt}

Based on the original plan {step1_result}, the critique {step2_result}, and the counter-analysis {step3_result}, create a FINAL CONCISE research plan (max 1000 words).

DECISIONS TO MAKE:
- What criticisms to implement within system constraints?
- What to reject and why?
- Final experimental design respecting SmolLM dataset and time limits
- Clear success criteria
- Minimal viable implementation

PRIORITIZE: Fairness, feasibility within constraints, and scientific rigor. Avoid feature bloat. Focus on what will actually advance the field while staying within the practical limitations."""
                }
            ]

            llm.run_chain(steps)
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)

    else:
        print(f"❌ Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
