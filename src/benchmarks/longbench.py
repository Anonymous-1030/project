"""
LongBench Benchmark Integration for ProSE-X 2.0.

LongBench is a comprehensive benchmark for long-context understanding,
covering 6 task categories across 21 datasets:
1. Single-document QA (NarrativeQA, Qasper, MultiFieldQA)
2. Multi-document QA (HotpotQA, 2WikiMQA, MuSiQue)
3. Summarization (GovReport, QMSum, MultiNews)
4. Few-shot Learning (TREC, TriviaQA, SAMSum)
5. Synthetic Tasks (PassageCount, PassageRetrieval)
6. Code Completion (LCC, RepoBench-P)

Reference: Bai et al., "LongBench: A Bilingual, Multitask Benchmark
for Long Context Understanding", ACL 2024.
"""

import torch
import json
import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Task Definitions ─────────────────────────────────────────────────

LONGBENCH_TASKS = {
    # Single-doc QA
    "narrativeqa": {
        "category": "single_doc_qa",
        "metric": "f1",
        "max_length": 32768,
        "dataset": "THUDM/LongBench",
        "subset": "narrativeqa",
    },
    "qasper": {
        "category": "single_doc_qa",
        "metric": "f1",
        "max_length": 32768,
        "dataset": "THUDM/LongBench",
        "subset": "qasper",
    },
    "multifieldqa_en": {
        "category": "single_doc_qa",
        "metric": "f1",
        "max_length": 32768,
        "dataset": "THUDM/LongBench",
        "subset": "multifieldqa_en",
    },
    # Multi-doc QA
    "hotpotqa": {
        "category": "multi_doc_qa",
        "metric": "f1",
        "max_length": 32768,
        "dataset": "THUDM/LongBench",
        "subset": "hotpotqa",
    },
    "2wikimqa": {
        "category": "multi_doc_qa",
        "metric": "f1",
        "max_length": 32768,
        "dataset": "THUDM/LongBench",
        "subset": "2wikimqa",
    },
    "musique": {
        "category": "multi_doc_qa",
        "metric": "f1",
        "max_length": 32768,
        "dataset": "THUDM/LongBench",
        "subset": "musique",
    },
    # Summarization
    "gov_report": {
        "category": "summarization",
        "metric": "rouge_l",
        "max_length": 32768,
        "dataset": "THUDM/LongBench",
        "subset": "gov_report",
    },
    "qmsum": {
        "category": "summarization",
        "metric": "rouge_l",
        "max_length": 32768,
        "dataset": "THUDM/LongBench",
        "subset": "qmsum",
    },
    "multi_news": {
        "category": "summarization",
        "metric": "rouge_l",
        "max_length": 32768,
        "dataset": "THUDM/LongBench",
        "subset": "multi_news",
    },
    # Few-shot
    "trec": {
        "category": "few_shot",
        "metric": "accuracy",
        "max_length": 32768,
        "dataset": "THUDM/LongBench",
        "subset": "trec",
    },
    "triviaqa": {
        "category": "few_shot",
        "metric": "f1",
        "max_length": 32768,
        "dataset": "THUDM/LongBench",
        "subset": "triviaqa",
    },
    "samsum": {
        "category": "few_shot",
        "metric": "rouge_l",
        "max_length": 32768,
        "dataset": "THUDM/LongBench",
        "subset": "samsum",
    },
    # Synthetic
    "passage_count": {
        "category": "synthetic",
        "metric": "accuracy",
        "max_length": 32768,
        "dataset": "THUDM/LongBench",
        "subset": "passage_count",
    },
    "passage_retrieval_en": {
        "category": "synthetic",
        "metric": "accuracy",
        "max_length": 32768,
        "dataset": "THUDM/LongBench",
        "subset": "passage_retrieval_en",
    },
    # Code
    "lcc": {
        "category": "code",
        "metric": "edit_similarity",
        "max_length": 32768,
        "dataset": "THUDM/LongBench",
        "subset": "lcc",
    },
    "repobench-p": {
        "category": "code",
        "metric": "edit_similarity",
        "max_length": 32768,
        "dataset": "THUDM/LongBench",
        "subset": "repobench-p",
    },
}


@dataclass
class LongBenchExample:
    """Single LongBench example."""
    task: str
    context: str
    query: str
    answer: str  # Gold answer (may be a list joined by newline)
    all_answers: List[str] = field(default_factory=list)
    context_length: int = 0
    category: str = ""


class LongBenchBenchmark:
    """
    LongBench evaluation benchmark.

    Loads data from HuggingFace or local JSONL files, runs generation
    with the given runner, and computes task-appropriate metrics.
    """

    def __init__(
        self,
        tokenizer,
        tasks: Optional[List[str]] = None,
        data_dir: Optional[str] = None,
        max_samples_per_task: int = 50,
        max_gen_tokens: int = 128,
    ):
        """
        Args:
            tokenizer: Model tokenizer
            tasks: Task names to evaluate (None = all)
            data_dir: Local data directory (None = download from HF)
            max_samples_per_task: Cap samples per task
            max_gen_tokens: Max tokens to generate per sample
        """
        self.tokenizer = tokenizer
        self.tasks = tasks or list(LONGBENCH_TASKS.keys())
        self.data_dir = data_dir
        self.max_samples = max_samples_per_task
        self.max_gen_tokens = max_gen_tokens

    def load_dataset(self, task: str) -> List[LongBenchExample]:
        """Load examples for a single task."""
        task_info = LONGBENCH_TASKS[task]
        examples = []

        # Try local JSONL first
        if self.data_dir:
            local_path = Path(self.data_dir) / f"{task}.jsonl"
            if local_path.exists():
                return self._load_from_jsonl(local_path, task, task_info)

        # Fall back to HuggingFace datasets
        try:
            from datasets import load_dataset
            ds = load_dataset(
                task_info["dataset"],
                task_info["subset"],
                split="test",
            )
            for item in ds:
                context = item.get("context", item.get("input", ""))
                query = item.get("input", item.get("question", ""))
                answers = item.get("answers", [item.get("answer", "")])
                if isinstance(answers, str):
                    answers = [answers]

                ctx_tokens = len(self.tokenizer.encode(context))
                examples.append(LongBenchExample(
                    task=task,
                    context=context,
                    query=query,
                    answer=answers[0] if answers else "",
                    all_answers=answers,
                    context_length=ctx_tokens,
                    category=task_info["category"],
                ))
        except Exception as e:
            logger.warning(f"Failed to load {task} from HuggingFace: {e}")
            return []

        if len(examples) > self.max_samples:
            examples = examples[:self.max_samples]

        logger.info(f"Loaded {len(examples)} examples for {task}")
        return examples

    def _load_from_jsonl(
        self, path: Path, task: str, task_info: Dict
    ) -> List[LongBenchExample]:
        """Load examples from local JSONL file."""
        examples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                context = item.get("context", item.get("input", ""))
                query = item.get("input", item.get("question", ""))
                answers = item.get("answers", [item.get("answer", "")])
                if isinstance(answers, str):
                    answers = [answers]

                ctx_tokens = len(self.tokenizer.encode(context))
                examples.append(LongBenchExample(
                    task=task,
                    context=context,
                    query=query,
                    answer=answers[0] if answers else "",
                    all_answers=answers,
                    context_length=ctx_tokens,
                    category=task_info["category"],
                ))
                if len(examples) >= self.max_samples:
                    break
        return examples

    def evaluate(
        self,
        runner,
        tasks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run full LongBench evaluation.

        Args:
            runner: Runner with .run(context_ids, query_ids, max_new_tokens) method
            tasks: Subset of tasks to run (None = use self.tasks)

        Returns:
            Results dict with per-task and aggregate scores
        """
        tasks = tasks or self.tasks
        results_by_task = {}
        results_by_category = {}

        for task in tasks:
            if task not in LONGBENCH_TASKS:
                logger.warning(f"Unknown task: {task}, skipping")
                continue

            task_info = LONGBENCH_TASKS[task]
            examples = self.load_dataset(task)
            if not examples:
                logger.warning(f"No examples for {task}, skipping")
                continue

            task_scores = []
            for example in examples:
                try:
                    prediction = self._generate(runner, example)
                    score = self._score(prediction, example, task_info["metric"])
                    task_scores.append(score)
                except Exception as e:
                    logger.warning(f"Failed on {task} example: {e}")
                    task_scores.append(0.0)

            avg_score = sum(task_scores) / len(task_scores) if task_scores else 0.0
            results_by_task[task] = {
                "score": avg_score,
                "n_samples": len(task_scores),
                "metric": task_info["metric"],
                "category": task_info["category"],
            }

            # Aggregate by category
            cat = task_info["category"]
            if cat not in results_by_category:
                results_by_category[cat] = []
            results_by_category[cat].append(avg_score)

        # Compute category averages
        category_averages = {
            cat: sum(scores) / len(scores)
            for cat, scores in results_by_category.items()
            if scores
        }

        # Overall average
        all_scores = [r["score"] for r in results_by_task.values()]
        overall = sum(all_scores) / len(all_scores) if all_scores else 0.0

        return {
            "overall": overall,
            "by_task": results_by_task,
            "by_category": category_averages,
        }

    def _generate(self, runner, example: LongBenchExample) -> str:
        """Generate prediction for a single example."""
        context_ids = self.tokenizer.encode(
            example.context, return_tensors="pt", truncation=True,
            max_length=LONGBENCH_TASKS[example.task]["max_length"],
        )
        query_ids = self.tokenizer.encode(
            example.query, return_tensors="pt",
        )

        generated_ids, _ = runner.run(
            context_ids, query_ids, max_new_tokens=self.max_gen_tokens
        )
        prediction = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        )
        return prediction.strip()

    def _score(
        self, prediction: str, example: LongBenchExample, metric: str
    ) -> float:
        """Score a prediction against gold answers."""
        if metric == "f1":
            return self._token_f1(prediction, example.all_answers)
        elif metric == "rouge_l":
            return self._rouge_l(prediction, example.answer)
        elif metric == "accuracy":
            return self._exact_match(prediction, example.all_answers)
        elif metric == "edit_similarity":
            return self._edit_similarity(prediction, example.answer)
        else:
            logger.warning(f"Unknown metric: {metric}")
            return 0.0

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for scoring."""
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def _token_f1(self, prediction: str, answers: List[str]) -> float:
        """Compute token-level F1 against multiple gold answers."""
        pred_tokens = self._normalize(prediction).split()
        if not pred_tokens:
            return 0.0

        best_f1 = 0.0
        for answer in answers:
            ans_tokens = self._normalize(answer).split()
            if not ans_tokens:
                continue
            common = set(pred_tokens) & set(ans_tokens)
            if not common:
                continue
            precision = len(common) / len(pred_tokens)
            recall = len(common) / len(ans_tokens)
            f1 = 2 * precision * recall / (precision + recall)
            best_f1 = max(best_f1, f1)

        return best_f1

    def _rouge_l(self, prediction: str, answer: str) -> float:
        """Compute ROUGE-L F1 score via longest common subsequence."""
        pred_tokens = self._normalize(prediction).split()
        ans_tokens = self._normalize(answer).split()
        if not pred_tokens or not ans_tokens:
            return 0.0

        lcs_len = self._lcs_length(pred_tokens, ans_tokens)
        precision = lcs_len / len(pred_tokens)
        recall = lcs_len / len(ans_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def compute_rouge_l_batch(self, predictions: List[str], answers: List[str]) -> List[float]:
        """Compute ROUGE-L for a batch of predictions and answers."""
        return [self._rouge_l(p, a) for p, a in zip(predictions, answers)]

    @staticmethod
    def _lcs_length(a: List[str], b: List[str]) -> int:
        """Length of longest common subsequence."""
        m, n = len(a), len(b)
        # Space-optimized DP
        prev = [0] * (n + 1)
        for i in range(1, m + 1):
            curr = [0] * (n + 1)
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(curr[j - 1], prev[j])
            prev = curr
        return prev[n]

    def _exact_match(self, prediction: str, answers: List[str]) -> float:
        """Check if prediction matches any answer."""
        pred_norm = self._normalize(prediction)
        for answer in answers:
            if self._normalize(answer) in pred_norm:
                return 1.0
        return 0.0

    @staticmethod
    def _edit_similarity(prediction: str, answer: str) -> float:
        """Compute normalized edit similarity (1 - edit_distance/max_len)."""
        m, n = len(prediction), len(answer)
        if m == 0 and n == 0:
            return 1.0
        if m == 0 or n == 0:
            return 0.0

        # Levenshtein distance (space-optimized)
        prev = list(range(n + 1))
        for i in range(1, m + 1):
            curr = [i] + [0] * n
            for j in range(1, n + 1):
                cost = 0 if prediction[i - 1] == answer[j - 1] else 1
                curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
            prev = curr

        edit_dist = prev[n]
        return 1.0 - edit_dist / max(m, n)
