import copy
import re

from balrog.agents.base import BaseAgent
from balrog.client import LLMClientWrapper


class RobustCoTAgent(BaseAgent):
    """An agent that performs actions using a chain-of-thought reasoning process."""

    def __init__(self, client_factory: LLMClientWrapper, prompt_builder, config):
        """Initialize the ChainOfThoughtAgent with a client, prompt builder, and configuration.

        Args:
            client_factory (LLMClientWrapper): A factory for creating the LLM client instance.
            prompt_builder (PromptBuilder): Object to build prompts for the agent.
            config: Configuration object containing settings for the agent.
        """
        super().__init__(client_factory, prompt_builder)
        self.remember_cot = config.agent.remember_cot

    def act(self, obs, prev_action=None):
        """Generate the next action using chain-of-thought reasoning based on the current observation.

        Args:
            obs (dict): The current observation in the environment.
            prev_action (str, optional): The previous action taken.

        Returns:
            LLMResponse: The response containing the final selected action.
        """
        if prev_action:
            self.prompt_builder.update_action(prev_action)

        self.prompt_builder.update_observation(obs)

        messages = self.prompt_builder.get_prompt()

        # Updated instructions: chain of thought + strict output format
        cot_instructions = """
First, think about the best course of action. Think step by step, but be concise in your reasoning. Focus on the most important factors and avoid unnecessary details.

Then, you must choose exactly one of the listed actions and output it strictly in the following format:

<|ACTION|>YOUR_CHOSEN_ACTION<|END|>

Replace YOUR_CHOSEN_ACTION with the chosen action.
        """.strip()

        # Add the updated instructions to the last message
        messages[-1].content += "\n\n" + cot_instructions + " /no_think"
        messages_return = {"messages": [{"role": m.role, "content": m.content} for m in messages]}

        # # --- Debug Logging Start ---
        # import os
        # from pathlib import Path

        # # Define the log file path
        # log_file_path = Path.home() / "Downloads" / "debug.log"
        # # Ensure the Downloads directory exists, create if not
        # log_file_path.parent.mkdir(parents=True, exist_ok=True)

        # # Open the file in append mode to log messages sent
        # with open(log_file_path, 'a') as f:
        #     f.write("--- Messages Being Sent to LLM (RobustCoT) ---\n")
        #     for i, msg in enumerate(messages):
        #         f.write(f"Message {i+1}:\n")
        #         f.write(f"  Role: {msg.role}\n")
        #         # Write content without ANSI codes, indenting multiline content
        #         content_lines = str(msg.content).split('\n')
        #         formatted_content = '\n'.join([f"    {line}" for line in content_lines])
        #         f.write(f"  Content:\n{formatted_content}\n")
        #         if hasattr(msg, 'attachment') and msg.attachment is not None:
        #             # Indicate presence of attachment without printing raw data
        #             f.write("  Attachment: [Image Data Present]\n")
        #         f.write("----------------------------------\n")
        # # --- Debug Logging End ---

        # Generate the CoT reasoning
        cot_reasoning = self.client.generate(messages)

        # --- Debug Logging Start ---
        # Open the file in append mode to log the LLM response
        # with open(log_file_path, 'a') as f:
        #     f.write("--- LLM Response Received (RobustCoT) ---\n")
        #     f.write(f"  Model ID: {cot_reasoning.model_id}\n")
        #     # Format completion similar to message content
        #     completion_lines = str(cot_reasoning.completion).split('\n')
        #     formatted_completion = '\n'.join([f"    {line}" for line in completion_lines])
        #     f.write(f"  Completion:\n{formatted_completion}\n")
        #     f.write(f"  Stop Reason: {cot_reasoning.stop_reason}\n")
        #     f.write(f"  Input Tokens: {cot_reasoning.input_tokens}\n")
        #     f.write(f"  Output Tokens: {cot_reasoning.output_tokens}\n")
        #     # Format reasoning if present and not None
        #     if cot_reasoning.reasoning is not None:
        #         reasoning_lines = str(cot_reasoning.reasoning).split('\n')
        #         formatted_reasoning = '\n'.join([f"    {line}" for line in reasoning_lines])
        #         f.write(f"  Reasoning:\n{formatted_reasoning}\n")
        #     else:
        #         f.write("  Reasoning: None\n")
        #     f.write("----------------------------------\n")
        # # --- Debug Logging End ---

        # Extract the final answer from the CoT reasoning
        final_answer = self._extract_final_answer(cot_reasoning)

        return final_answer, cot_reasoning.completion, messages_return

    def _extract_final_answer(self, reasoning):
        """Extract the final action from the chain-of-thought reasoning response.

        Args:
            reasoning (LLMResponse): The response containing CoT reasoning and action.

        Returns:
            LLMResponse: The response with the extracted final action in `completion`
                         and the entire chain-of-thought in `reasoning`.
        """
        # Make a copy so we don't mutate the original
        final_answer = copy.deepcopy(reasoning)

        # Store the entire chain-of-thought (raw completion) in `reasoning`
        final_answer = final_answer._replace(reasoning=reasoning.completion)

        # Now parse the strict action format: <|ACTION|> ... <|END|>
        completion_text = reasoning.completion
        match = re.search(r"<\|ACTION\|>(.*?)<\|END\|>", completion_text, re.DOTALL)
        if match:
            extracted_action = match.group(1).strip()
        else:
            # Fallback to the entire completion if not matched
            extracted_action = "Failed to obtain a valid action from the reasoning."

        # Replace the final `completion` with only the extracted action
        final_answer = final_answer._replace(completion=extracted_action)

        return final_answer
