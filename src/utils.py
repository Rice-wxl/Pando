"""Utility functions for the benchmark."""

import logging
import random
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


def set_seed(seed: int) -> int:
    """Set random seed for reproducibility.

    Args:
        seed: Random seed. If -1, generates a random seed.

    Returns:
        The actual seed used (useful when seed=-1).
    """
    if seed < 0:
        seed = random.randint(0, 2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def get_timestamp() -> str:
    """Get current timestamp in YYYYMMDD_HHMMSS format."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_logging(output_dir: Path | str | None = None, level: int = logging.INFO) -> logging.Logger:
    """Setup logging configuration.

    Args:
        output_dir: If provided, also log to a file in this directory.
        level: Logging level.

    Returns:
        Configured logger.
    """
    logger = logging.getLogger("interp_bench")
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(output_dir / "run.log")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def get_chat_template_parts(tokenizer: "PreTrainedTokenizer") -> dict:
    """Discover chat template structure by fuzzing with placeholders.

    This dynamically discovers the chat template format for any model by
    using placeholder strings and finding their positions in the output.

    Args:
        tokenizer: HuggingFace tokenizer with chat template support.

    Returns:
        Dict with:
            - prefix: Text before user message
            - user_suffix: Text between user message and assistant message
            - asst_suffix: Text after assistant message (end token)
            - generation_prompt: Text after user message for generation

    Raises:
        ValueError: If tokenizer doesn't support chat templates or placeholders not preserved.
    """
    USER_PLACEHOLDER = "[USER_MSG_PLACEHOLDER_12345]"
    ASST_PLACEHOLDER = "[ASST_MSG_PLACEHOLDER_67890]"

    # Get full template with both messages
    messages = [
        {"role": "user", "content": USER_PLACEHOLDER},
        {"role": "assistant", "content": ASST_PLACEHOLDER}
    ]
    try:
        full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except (AttributeError, TypeError) as e:
        raise ValueError(f"Tokenizer does not support chat templates: {e}")

    # Get template with just user message + generation prompt
    user_only = tokenizer.apply_chat_template(
        [{"role": "user", "content": USER_PLACEHOLDER}],
        tokenize=False, add_generation_prompt=True
    )

    # Find positions and validate
    user_start = full.find(USER_PLACEHOLDER)
    asst_start = full.find(ASST_PLACEHOLDER)

    if user_start == -1:
        raise ValueError(f"Chat template did not preserve user placeholder. Got: {full[:200]}...")
    if asst_start == -1:
        raise ValueError(f"Chat template did not preserve assistant placeholder. Got: {full[:200]}...")
    if user_start >= asst_start:
        raise ValueError(f"Placeholders in wrong order: user at {user_start}, assistant at {asst_start}")

    user_end = user_start + len(USER_PLACEHOLDER)
    asst_end = asst_start + len(ASST_PLACEHOLDER)

    user_only_pos = user_only.find(USER_PLACEHOLDER)
    if user_only_pos == -1:
        raise ValueError(f"User-only template did not preserve placeholder. Got: {user_only[:200]}...")

    return {
        "prefix": full[:user_start],           # Before user message
        "user_suffix": full[user_end:asst_start],  # Between user msg and assistant msg
        "asst_suffix": full[asst_end:],        # After assistant message (end token)
        "generation_prompt": user_only[user_only_pos + len(USER_PLACEHOLDER):],
    }
