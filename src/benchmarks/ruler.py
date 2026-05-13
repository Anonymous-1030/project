"""
RULER Benchmark Integration for ProSE-X 2.0.

RULER (Real-world Understanding of Long-context Evaluation and Reasoning)
tests models at controlled sequence lengths with 4 task categories:

1. Needle-in-a-Haystack (NIAH): Single/multi-key retrieval
2. Variable Tracking (VT): Track variable assignments across context
3. Common/Frequent Words (CW/FW): Identify most common/frequent words
4. Question Answering (QA): Answer questions from long documents

Key for HPCA: RULER controls sequence length precisely, enabling
systematic evaluation of KV cache management at 4K, 8K, 16K, 32K,
64K, and 128K tokens.

Reference: Hsieh et al., "RULER: What's the Real Context Size of
Your Long-Context Language Models?", COLM 2024.
"""

import torch
import random
import logging
import string
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RULERExample:
    """Single RULER benchmark example."""
    task: str
    context: str
    query: str
    answer: str
    all_answers: List[str] = field(default_factory=list)
    context_length: int = 0
    target_length: int = 0
    needle_depth: float = 0.0  # Position of needle in context [0, 1]


class RULERBenchmark:
    """
    RULER benchmark for systematic long-context evaluation.

    Generates synthetic tasks at controlled sequence lengths to
    precisely measure KV cache retention quality.
    """

    # Filler sentences for building long contexts
    FILLER_SENTENCES = [
        "The city was bustling with activity as people went about their daily routines.",
        "Research in artificial intelligence continues to advance at a rapid pace.",
        "The mountain trail wound through dense forests and across rushing streams.",
        "Economic indicators suggest a period of moderate growth ahead.",
        "Ancient civilizations developed sophisticated systems of governance.",
        "The library contained thousands of rare manuscripts from centuries past.",
        "Climate patterns have been shifting noticeably over the past decade.",
        "Musicians from around the world gathered for the annual festival.",
        "The laboratory experiment yielded unexpected but promising results.",
        "Historical records indicate that trade routes were established early.",
    ]

    def __init__(
        self,
        tokenizer,
        context_lengths: Optional[List[int]] = None,
        num_samples_per_config: int = 10,
    ):
        """
        Args:
            tokenizer: Model tokenizer
            context_lengths: Sequence lengths to test
            num_samples_per_config: Samples per (task, length) pair
        """
        self.tokenizer = tokenizer
        self.context_lengths = context_lengths or [4096, 8192, 16384, 32768, 65536]
        self.num_samples = num_samples_per_config

    # ── Task Generators ──────────────────────────────────────────────

    def _build_filler(self, target_tokens: int) -> str:
        """Build filler text of approximately target_tokens length."""
        filler_block = " ".join(self.FILLER_SENTENCES)
        filler_tokens = len(self.tokenizer.encode(filler_block))
        repeats = max(1, (target_tokens // filler_tokens) + 1)
        full_filler = (filler_block + " ") * repeats
        # Trim to approximate target
        tokens = self.tokenizer.encode(full_filler)
        tokens = tokens[:target_tokens]
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def generate_niah_single(
        self, target_length: int, needle_depth: float
    ) -> RULERExample:
        """
        Single Needle-in-a-Haystack.

        Hide a single key-value pair in filler text, ask for value.
        """
        key = "".join(random.choices(string.ascii_lowercase, k=6))
        value = "".join(random.choices(string.digits, k=7))
        needle = f"The special key '{key}' has the value '{value}'."

        # Reserve tokens for needle + query
        needle_tokens = len(self.tokenizer.encode(needle))
        filler_tokens = target_length - needle_tokens - 30

        filler = self._build_filler(filler_tokens)
        insert_char = int(len(filler) * needle_depth)
        context = filler[:insert_char] + " " + needle + " " + filler[insert_char:]

        query = f"What is the value of the key '{key}'? The value is"

        return RULERExample(
            task="niah_single",
            context=context,
            query=query,
            answer=value,
            all_answers=[value],
            context_length=len(self.tokenizer.encode(context)),
            target_length=target_length,
            needle_depth=needle_depth,
        )

    def generate_niah_multi(
        self, target_length: int, num_needles: int = 3
    ) -> RULERExample:
        """
        Multi-Needle-in-a-Haystack.

        Hide multiple key-value pairs and ask for a specific one.
        """
        keys = [
            "".join(random.choices(string.ascii_lowercase, k=5))
            for _ in range(num_needles)
        ]
        values = [
            "".join(random.choices(string.digits, k=6))
            for _ in range(num_needles)
        ]
        needles = [
            f"Record: key='{k}' maps to value='{v}'."
            for k, v in zip(keys, values)
        ]

        # Distribute needles evenly
        filler = self._build_filler(target_length - 100)
        context_parts = []
        chunk_size = len(filler) // (num_needles + 1)
        for i, needle in enumerate(needles):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            context_parts.append(filler[start:end])
            context_parts.append(" " + needle + " ")
        context_parts.append(filler[(num_needles) * chunk_size:])
        context = "".join(context_parts)

        # Ask about a random key
        target_idx = random.randint(0, num_needles - 1)
        query = f"What value does the key '{keys[target_idx]}' map to? The value is"

        return RULERExample(
            task="niah_multi",
            context=context,
            query=query,
            answer=values[target_idx],
            all_answers=[values[target_idx]],
            context_length=len(self.tokenizer.encode(context)),
            target_length=target_length,
        )

    def generate_variable_tracking(
        self, target_length: int, num_vars: int = 5
    ) -> RULERExample:
        """
        Variable Tracking task.

        Define variables, reassign some, ask for final value.
        """
        var_names = [f"var_{chr(65 + i)}" for i in range(num_vars)]
        initial_values = [random.randint(100, 999) for _ in range(num_vars)]

        assignments = []
        for name, val in zip(var_names, initial_values):
            assignments.append(f"Set {name} = {val}.")

        # Perform random reassignments
        current_values = dict(zip(var_names, initial_values))
        num_reassigns = random.randint(2, min(num_vars, 4))
        for _ in range(num_reassigns):
            var = random.choice(var_names)
            new_val = random.randint(100, 999)
            assignments.append(f"Update {var} = {new_val}.")
            current_values[var] = new_val

        # Interleave with filler
        filler = self._build_filler(target_length - 200)
        context_parts = []
        filler_chunk = len(filler) // (len(assignments) + 1)
        for i, assignment in enumerate(assignments):
            context_parts.append(filler[i * filler_chunk:(i + 1) * filler_chunk])
            context_parts.append(" " + assignment + " ")
        context_parts.append(filler[len(assignments) * filler_chunk:])
        context = "".join(context_parts)

        target_var = random.choice(var_names)
        query = f"After all updates, what is the final value of {target_var}? The value is"

        return RULERExample(
            task="variable_tracking",
            context=context,
            query=query,
            answer=str(current_values[target_var]),
            all_answers=[str(current_values[target_var])],
            context_length=len(self.tokenizer.encode(context)),
            target_length=target_length,
        )

    def generate_frequent_words(
        self, target_length: int
    ) -> RULERExample:
        """
        Common/Frequent Words task.

        Ask which special word appears most frequently in the context.
        """
        special_words = ["alpha", "beta", "gamma", "delta", "epsilon"]
        # Assign frequencies
        target_word = random.choice(special_words)
        word_counts = {}
        for w in special_words:
            word_counts[w] = random.randint(2, 8) if w != target_word else random.randint(12, 20)

        # Build context with embedded special words
        filler = self._build_filler(target_length - 200)
        sentences = filler.split(". ")
        for word, count in word_counts.items():
            positions = random.sample(
                range(len(sentences)), min(count, len(sentences))
            )
            for pos in positions:
                sentences[pos] = sentences[pos] + f" (marker: {word})"

        context = ". ".join(sentences)
        query = (
            f"Among the markers ({', '.join(special_words)}), "
            f"which one appears most frequently? The most frequent marker is"
        )

        return RULERExample(
            task="frequent_words",
            context=context,
            query=query,
            answer=target_word,
            all_answers=[target_word],
            context_length=len(self.tokenizer.encode(context)),
            target_length=target_length,
        )

    # ── Main Interface ───────────────────────────────────────────────

    def generate_dataset(
        self,
        tasks: Optional[List[str]] = None,
    ) -> List[RULERExample]:
        """Generate full RULER dataset across all tasks and lengths."""
        tasks = tasks or ["niah_single", "niah_multi", "variable_tracking", "frequent_words"]
        examples = []

        depths = [0.0, 0.25, 0.5, 0.75, 1.0]

        for target_length in self.context_lengths:
            for _ in range(self.num_samples):
                if "niah_single" in tasks:
                    depth = random.choice(depths)
                    examples.append(
                        self.generate_niah_single(target_length, depth)
                    )
                if "niah_multi" in tasks:
                    examples.append(
                        self.generate_niah_multi(target_length)
                    )
                if "variable_tracking" in tasks:
                    examples.append(
                        self.generate_variable_tracking(target_length)
                    )
                if "frequent_words" in tasks:
                    examples.append(
                        self.generate_frequent_words(target_length)
                    )

        logger.info(
            f"Generated {len(examples)} RULER examples "
            f"across {len(self.context_lengths)} lengths"
        )
        return examples

    def evaluate(
        self,
        runner,
        examples: Optional[List[RULERExample]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a runner on RULER benchmark.

        Args:
            runner: Runner with .run(context_ids, query_ids, max_new_tokens)
            examples: Pre-generated examples (generates if None)

        Returns:
            Results dict with per-task, per-length, and aggregate scores
        """
        if examples is None:
            examples = self.generate_dataset()

        results_by_task = {}
        results_by_length = {}

        for example in examples:
            try:
                context_ids = self.tokenizer.encode(
                    example.context, return_tensors="pt",
                    truncation=True, max_length=example.target_length,
                )
                query_ids = self.tokenizer.encode(
                    example.query, return_tensors="pt"
                )
                generated_ids, _ = runner.run(
                    context_ids, query_ids, max_new_tokens=20
                )
                prediction = self.tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                ).strip()

                is_correct = any(
                    ans in prediction for ans in example.all_answers
                )
                score = 1.0 if is_correct else 0.0

            except Exception as e:
                logger.warning(f"RULER eval failed: {e}")
                score = 0.0

            # By task
            task = example.task
            if task not in results_by_task:
                results_by_task[task] = {"correct": 0, "total": 0}
            results_by_task[task]["total"] += 1
            results_by_task[task]["correct"] += int(score)

            # By length
            length = example.target_length
            if length not in results_by_length:
                results_by_length[length] = {"correct": 0, "total": 0}
            results_by_length[length]["total"] += 1
            results_by_length[length]["correct"] += int(score)

        # Compute averages
        task_accs = {}
        for task, counts in results_by_task.items():
            task_accs[task] = counts["correct"] / counts["total"] if counts["total"] > 0 else 0

        length_accs = {}
        for length, counts in results_by_length.items():
            length_accs[length] = counts["correct"] / counts["total"] if counts["total"] > 0 else 0

        all_scores = list(task_accs.values())
        overall = sum(all_scores) / len(all_scores) if all_scores else 0.0

        return {
            "accuracy": overall,
            "by_task": task_accs,
            "by_length": length_accs,
            "n_examples": len(examples),
        }
