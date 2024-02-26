from typing import Dict, Union


def calculate_usage_cost(llm_model: str, usage: Union[Dict, None]) -> float:
    """
    Return the cost of the offload
    https://openai.com/pricing
    """
    if llm_model == "gpt-4-1106-preview" or "gpt-4-vision-preview":
        input_cost_per_token = 0.01 / 1e3  # $/token
        output_cost_per_token = 0.03 / 1e3  # $/token
    elif llm_model == "gpt-3.5-turbo-1106":
        input_cost_per_token = 0.0010 / 1e3  # $/token
        output_cost_per_token = 0.0020 / 1e3  # $/tokens
    elif llm_model == "gemini-pro" or "gemini-pro-vision":
        return 0

    return round(
        usage["prompt_tokens"] * input_cost_per_token
        + usage["completion_tokens"] * output_cost_per_token,
        2,
    )
