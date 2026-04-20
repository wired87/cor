"""Minimal Gem placeholder when no LLM backend is configured (local pipeline)."""


class Gem:
    def ask(self, prompt: str) -> str:
        """
        Ask workflow state for the Gem workflow.

        Workflow:
        1. Reads and normalizes the incoming inputs, including `prompt`.
        2. Returns the assembled result to the caller.

        Inputs:
        - `prompt`: Caller-supplied value used during processing. Expected type: `str`.

        Returns:
        - Returns `str` to the caller.
        """
        return "[Gem stub] No GEMINI_API_KEY / gem backend configured."
