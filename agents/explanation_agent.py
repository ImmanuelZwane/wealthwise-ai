import os
import google.generativeai as genai

class ExplanationAgent:
    """
    Generates plain‑language explanations of portfolio actions using Gemini (or Ollama fallback).
    Automatically picks an available Gemini model that supports generateContent.
    """

    def __init__(self, preferred_model=None, api_key=None):
        """
        Args:
            preferred_model (str, optional): Preferred Gemini model name (e.g., "gemini-1.5-flash").
            api_key (str, optional): Gemini API key. If None, tries environment variable GEMINI_API_KEY.
        """
        # Use provided key or fall back to environment variable
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")

        if api_key:
            # Configure Gemini with the key
            genai.configure(api_key=api_key)
            # Get list of available models that support generateContent
            models = [m.name for m in genai.list_models()
                      if 'generateContent' in m.supported_generation_methods]
            if not models:
                raise Exception("No suitable Gemini models available for your API key.")
            # Use preferred model if it's available, otherwise first available
            if preferred_model and f"models/{preferred_model}" in models:
                self.model_name = preferred_model
            else:
                # Extract short name (e.g., "gemini-1.5-flash" from "models/gemini-1.5-flash")
                self.model_name = models[0].split('/')[-1]
            self.model = genai.GenerativeModel(self.model_name)
            self.use_gemini = True
        else:
            # No API key – fallback to local Ollama (if installed)
            try:
                import ollama
                self.ollama = ollama
                self.ollama_model = "llama3.2"  # You can change this to any model you have pulled
                self.use_gemini = False
            except ImportError:
                print("Ollama not installed. Please install google-generativeai or provide a Gemini API key.")
                self.use_gemini = None

    def explain_rebalance(self, portfolio_summary, suggestions, market_conditions=""):
        """
        Generate an explanation for the suggested trades.

        Args:
            portfolio_summary (str): Text summary of current portfolio (total value, allocations).
            suggestions (str): Table or text describing suggested trades.
            market_conditions (str): Brief note about current market environment.

        Returns:
            str: Plain‑language explanation.
        """
        prompt = f"""
You are a friendly financial advisor for a retired person. Explain the following portfolio rebalancing suggestions in simple, clear language. Avoid jargon. Be reassuring and highlight the benefits.

Portfolio Summary:
{portfolio_summary}

Suggested Trades:
{suggestions}

Market Context:
{market_conditions}

Explain why these trades are being recommended and how they help keep the portfolio safe and aligned with the target allocation.
"""
        if self.use_gemini:
            try:
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as e:
                return f"Gemini error: {e}"
        elif self.use_gemini is False:
            try:
                response = self.ollama.chat(model=self.ollama_model,
                                            messages=[{'role': 'user', 'content': prompt}])
                return response['message']['content']
            except Exception as e:
                return f"Ollama error: {e}"
        else:
            return "No explanation agent available. Please set a Gemini API key or install Ollama."