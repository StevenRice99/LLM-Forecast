import re

import ollama


def initialize() -> None:
    """
    Ensure the model can be set up for Ollama.
    :return: Nothing.
    """
    ollama.pull("llama3.1")


def clean(message: str) -> str:
    """
    Clean a message.
    :param message: The message to clean.
    :return: The message with all newlines being at most single, and all other whitespace being replaced by a space.
    """
    # Remove markdown symbols.
    for symbol in ["*", "#", "_", ">"]:
        message = message.replace(symbol, " ")
    # Replace all whitespace with spaces, except newlines.
    message = re.sub(r"[^\S\n]+", " ", message)
    # Ensure no duplicate newlines.
    while message.__contains__("\n\n"):
        message = message.replace("\n\n", "\n")
    # Strip and return the message.
    return message.strip()


def generate(prompt: str, clean_prompt: bool = False, clean_response: bool = True) -> str:
    """
    Generate a response from a large language model from Ollama.
    :param prompt: The prompt you wish to pass.
    :param clean_prompt: If the prompt should be cleaned before being passed to the large language model.
    :param clean_response: If the response from the large language model should be cleaned.
    :return: The generated response.
    """
    if clean_prompt:
        prompt = clean(prompt)
    response = ollama.generate("llama3.1", prompt)["response"]
    return clean(response) if clean_response else response
