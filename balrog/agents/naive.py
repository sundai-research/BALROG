import copy
import re

from balrog.agents.base import BaseAgent


class NaiveAgent(BaseAgent):
    """An agent that generates actions based on observations without complex reasoning."""

    def __init__(self, client_factory, prompt_builder):
        """Initialize the NaiveAgent with a client and prompt builder."""
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()

    def act(self, obs, prev_action=None):
        """Generate the next action based on the observation and previous action.

        Args:
            obs (dict): The current observation in the environment.
            prev_action (str, optional): The previous action taken.

        Returns:
            str: The selected action from the LLM response.
        """
        if prev_action:
            self.prompt_builder.update_action(prev_action)

        self.prompt_builder.update_observation(obs)

        messages = self.prompt_builder.get_prompt()

        naive_instruction = """
You always have to output one of the above actions at a time and no other text. You always have to output an action until the episode terminates.
        """.strip()

        if messages and messages[-1].role == "user":
            messages[-1].content += "\n\n" + naive_instruction + " /no_think"

        # from ipdb import set_trace; set_trace()
        # ANSI escape codes for bright cyan and reset
        import os
        from pathlib import Path

        # Define the log file path
        log_file_path = Path.home() / "Downloads" / "debug.log"
        # Ensure the Downloads directory exists, create if not
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Open the file in append mode
        with open(log_file_path, 'a') as f:
            f.write("--- Messages Being Sent to LLM ---\n")
            for i, msg in enumerate(messages):
                f.write(f"Message {i+1}:\n")
                f.write(f"  Role: {msg.role}\n")
                # Write content without ANSI codes, indenting multiline content
                content_lines = str(msg.content).split('\n')
                formatted_content = '\n'.join([f"    {line}" for line in content_lines])
                f.write(f"  Content:\n{formatted_content}\n")
                if hasattr(msg, 'attachment') and msg.attachment is not None:
                    # Indicate presence of attachment without printing raw data
                    f.write("  Attachment: [Image Data Present]\n")
                f.write("----------------------------------\n")
        response = self.client.generate(messages)
        # --- Log the LLM Response ---
        with open(log_file_path, 'a') as f:
            f.write("--- LLM Response Received ---\n")
            f.write(f"  Model ID: {response.model_id}\n")
            # Format completion similar to message content
            completion_lines = str(response.completion).split('\n')
            formatted_completion = '\n'.join([f"    {line}" for line in completion_lines])
            f.write(f"  Completion:\n{formatted_completion}\n")
            f.write(f"  Stop Reason: {response.stop_reason}\n")
            f.write(f"  Input Tokens: {response.input_tokens}\n")
            f.write(f"  Output Tokens: {response.output_tokens}\n")
            # Format reasoning if present and not None
            if response.reasoning is not None:
                reasoning_lines = str(response.reasoning).split('\n')
                formatted_reasoning = '\n'.join([f"    {line}" for line in reasoning_lines])
                f.write(f"  Reasoning:\n{formatted_reasoning}\n")
            else:
                f.write("  Reasoning: None\n")
            f.write("----------------------------------\n")
        # from ipdb import set_trace; set_trace()
        final_answer = self._extract_final_answer(response)

        return final_answer

    def _extract_final_answer(self, answer):
        """Sanitize the final answer, keeping only alphabetic characters.

        Args:
            answer (LLMResponse): The response from the LLM.

        Returns:
            LLMResponse: The sanitized response.
        """

        def filter_letters(input_string):
            return re.sub(r"[^a-zA-Z\s:]", "", input_string)

        final_answer = copy.deepcopy(answer)
        final_answer = final_answer._replace(completion=filter_letters(final_answer.completion))

        return final_answer
