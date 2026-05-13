"""
Unified Causal Verification Runner.

Orchestrates the Seven-Layer Causal Verification Framework.
Generates realistic KV cache trace data (modeling production transformer
attention patterns) and feeds it to all 7 layers.

Usage:
    # Run all layers
    python -m prosex.src.runners.run_causal_verification --all

    # Run a single layer
    python -m prosex.src.runners.run_causal_verification --layer 1
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.config import ProSEXv2Config, CausalVerificationConfig
from src.core_types import EvidenceVector, CausalVerificationReport
from src.eval.causal import (
    EvidenceDecomposer,
    CEILayerRunner,
    QUDMLayerRunner,
    EBQPTLayerRunner,
    ITLBPLayerRunner,
    ACSLayerRunner,
    CACTLayerRunner,
    OCALayerRunner,
    CausalVerificationEvaluator,
)
from src.eval.causal.trace_data import (
    TraceDataGenerator,
    generate_full_trace_dataset,
    trace_to_evidence_vectors,
    trace_to_attention_utility,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default trace generation parameters
DEFAULT_NUM_CHUNKS = 64
DEFAULT_NUM_STEPS = 50


class CausalVerificationRunner:
    """
    Unified runner for the Seven-Layer Causal Verification Framework.

    Generates a shared set of realistic KV cache traces and feeds them
    to all 7 layers. The traces encode genuine causal structure mirroring
    production transformer attention patterns.
    """

    def __init__(self, config: ProSEXv2Config, trace_seed: int = 42):
        self.config = config
        self.causal_config = config.causal
        self.evaluator = CausalVerificationEvaluator()
        self.trace_seed = trace_seed

        # Initialize layer runners
        self.cei_runner = CEILayerRunner(self.causal_config)
        self.qudm_runner = QUDMLayerRunner(self.causal_config)
        self.ebqpt_runner = EBQPTLayerRunner(self.causal_config)
        self.itlbp_runner = ITLBPLayerRunner(self.causal_config)
        self.acs_runner = ACSLayerRunner(self.causal_config)
        self.cact_runner = CACTLayerRunner(self.causal_config)
        self.oca_runner = OCALayerRunner(self.causal_config)

        # Cached trace data (generated on first use)
        self._traces = None
        self._evidence_per_step = None
        self._utility_per_step = None
        self._phase_per_step = None
        self._all_evidence = None
        self._all_utilities = None
        self._all_phases = None

    def _ensure_traces(self):
        """Generate trace data if not already cached."""
        if self._traces is None:
            logger.info("Generating realistic KV cache trace data ...")
            (
                self._traces,
                self._evidence_per_step,
                self._utility_per_step,
                self._phase_per_step,
            ) = generate_full_trace_dataset(
                num_chunks=DEFAULT_NUM_CHUNKS,
                num_steps=DEFAULT_NUM_STEPS,
                seed=self.trace_seed,
            )

            # Flatten all evidence vectors across steps
            self._all_evidence = []
            for evs in self._evidence_per_step:
                self._all_evidence.extend(evs)

            # Flatten utilities and phases
            self._all_utilities = np.concatenate(self._utility_per_step)
            self._all_phases = np.concatenate(self._phase_per_step)

            logger.info(
                f"  Generated {len(self._traces)} steps x "
                f"{len(self._traces[0].chunks)} chunks = "
                f"{len(self._all_evidence)} total evidence vectors"
            )

    def run_all(self) -> CausalVerificationReport:
        """Run all enabled layers."""
        self._ensure_traces()

        if self.causal_config.run_layer_1_cei:
            self.run_layer_1()
        if self.causal_config.run_layer_2_qudm:
            self.run_layer_2()
        if self.causal_config.run_layer_3_ebqpt:
            self.run_layer_3()
        if self.causal_config.run_layer_4_itlbp:
            self.run_layer_4()
        if self.causal_config.run_layer_5_acs:
            self.run_layer_5()
        if self.causal_config.run_layer_6_cact:
            self.run_layer_6()
        if self.causal_config.run_layer_7_oca:
            self.run_layer_7()

        return self.evaluator.finalize("causal_verification")

    def run_layer_1(self):
        """Run Layer 1: Counterfactual Evidence Intervention."""
        logger.info("=== Layer 1: Counterfactual Evidence Intervention (CEI) ===")

        # Use ALL evidence vectors across all steps to measure phase consistency
        # across all 4 decode phases (early, mid, late, terminal).
        evs = list(self._all_evidence)
        utils = self._all_utilities
        phases = self._all_phases

        results = self.cei_runner.run(
            evs,
            admission_threshold=self.causal_config.cei_pass_threshold * 3,
            utility_labels=utils,
            phase_labels=phases,
        )
        self.evaluator.add_layer_1_results(results)
        logger.info(f"  {len(results)} intervention results computed")

    def run_layer_2(self):
        """Run Layer 2: Query-Utility Disentanglement Matrix."""
        logger.info("=== Layer 2: Query-Utility Disentanglement Matrix (QUDM) ===")

        # Use trace-based evidence as input
        metrics, qcdr, pass_fail = self.qudm_runner.run_analytical(
            evidence_vectors=self._all_evidence,
            utility_labels=self._all_utilities,
        )
        self.evaluator.add_layer_2_results(metrics, qcdr, pass_fail)
        logger.info(f"  QCDR={qcdr:.3f}, pass={pass_fail}")

    def run_layer_3(self):
        """Run Layer 3: Evidence Budget-Query Projection Tradeoff."""
        logger.info("=== Layer 3: Evidence Budget-Query Projection Tradeoff (EB-QPT) ===")

        results = self.ebqpt_runner.run_analytical(
            evidence_vectors=self._all_evidence,
            utility_labels=self._all_utilities,
        )
        self.evaluator.add_layer_3_results(results)
        logger.info(f"  {len(results)} budget configurations swept")

    def run_layer_4(self):
        """Run Layer 4: Information-Theoretic Lower Bound Probing."""
        logger.info("=== Layer 4: Information-Theoretic Lower Bound Probing (ITLBP) ===")

        result = self.itlbp_runner.run_analytical(
            evidence_vectors=self._all_evidence,
            utility_labels=self._all_utilities,
        )
        self.evaluator.add_layer_4_results(result)
        logger.info(
            f"  Saturation={result.saturation_entropy:.0f} bits, "
            f"64B sufficient={result.is_64b_sufficient}"
        )

    def run_layer_5(self):
        """Run Layer 5: Adversarial Causal Spoofing."""
        logger.info("=== Layer 5: Adversarial Causal Spoofing (ACS) ===")

        results = self.acs_runner.run_analytical(
            evidence_vectors=self._all_evidence,
            utility_labels=self._all_utilities,
        )
        self.evaluator.add_layer_5_results(results)
        for r in results:
            logger.info(f"  {r.spoof_type}: CVI={r.cvi:.3f}, pass={r.pass_fail}")

    def run_layer_6(self):
        """Run Layer 6: Cross-Architectural Causal Transfer."""
        logger.info("=== Layer 6: Cross-Architectural Causal Transfer (CACT) ===")

        results = self.cact_runner.run_analytical(
            evidence_vectors=self._all_evidence,
            utility_labels=self._all_utilities,
        )
        self.evaluator.add_layer_6_results(results)
        for r in results:
            logger.info(f"  {r.architecture}: CEC={r.cec_vs_mha:.3f}")

    def run_layer_7(self):
        """Run Layer 7: Online Causal Adaptation."""
        logger.info("=== Layer 7: Online Causal Adaptation (OCA) ===")

        results = self.oca_runner.run_analytical(
            evidence_per_step=self._evidence_per_step,
            utility_per_step=self._utility_per_step,
        )
        self.evaluator.add_layer_7_results(results)
        for r in results:
            logger.info(
                f"  {r.algorithm.value}: regret={r.cumulative_regret:.3f}, "
                f"SBFI violations={r.sbfi_boundary_violations}, pass={r.pass_fail}"
            )

    def save_report(self, report: CausalVerificationReport, output_dir: str):
        """Save the verification report to JSON."""
        output_path = Path(output_dir) / "causal_verification_report.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2, default=str)

        logger.info(f"Report saved to {output_path}")


def load_config(config_path: str) -> ProSEXv2Config:
    """Load ProSEXv2Config from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return ProSEXv2Config.from_dict(config_dict)


def main():
    parser = argparse.ArgumentParser(
        description="Seven-Layer Causal Verification Framework for ODUS-X"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (uses defaults if not specified)"
    )
    parser.add_argument(
        "--output", type=str, default="outputs/causal",
        help="Output directory for reports"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all 7 layers"
    )
    parser.add_argument(
        "--layer", type=int, choices=range(1, 8),
        help="Run a single layer (1-7)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for trace generation"
    )

    args = parser.parse_args()

    if not args.all and args.layer is None:
        parser.print_help()
        print("\nPlease specify --all or --layer N")
        return

    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config = ProSEXv2Config()

    # Run
    runner = CausalVerificationRunner(config, trace_seed=args.seed)

    if args.all:
        report = runner.run_all()
    else:
        layer_method = getattr(runner, f"run_layer_{args.layer}")
        runner._ensure_traces()
        layer_method()
        report = runner.evaluator.finalize(f"layer_{args.layer}")

    # Print and save
    runner.evaluator.print_report(report)
    runner.save_report(report, args.output)


if __name__ == "__main__":
    main()
