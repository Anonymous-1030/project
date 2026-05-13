"""
Contradiction Report Generator for ProSE-X 2.0

Generates detailed reports on metric inconsistencies and contradictions.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Contradiction:
    """A single detected contradiction."""
    contradiction_id: str
    name: str
    description: str
    severity: str  # "CRITICAL", "HIGH", "MEDIUM", "LOW"
    observed_values: Dict[str, Any] = field(default_factory=dict)
    expected_relationship: str = ""
    possible_causes: List[str] = field(default_factory=list)
    suggested_fixes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "contradiction_id": self.contradiction_id,
            "name": self.name,
            "description": self.description,
            "severity": self.severity,
            "observed_values": self.observed_values,
            "expected_relationship": self.expected_relationship,
            "possible_causes": self.possible_causes,
            "suggested_fixes": self.suggested_fixes,
        }


@dataclass
class ContradictionReport:
    """Complete contradiction report."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    contradictions: List[Contradiction] = field(default_factory=list)
    consistency_checks_passed: int = 0
    consistency_checks_failed: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "summary": {
                "total_contradictions": len(self.contradictions),
                "critical": sum(1 for c in self.contradictions if c.severity == "CRITICAL"),
                "high": sum(1 for c in self.contradictions if c.severity == "HIGH"),
                "medium": sum(1 for c in self.contradictions if c.severity == "MEDIUM"),
                "low": sum(1 for c in self.contradictions if c.severity == "LOW"),
                "consistency_checks_passed": self.consistency_checks_passed,
                "consistency_checks_failed": self.consistency_checks_failed,
            },
            "contradictions": [c.to_dict() for c in self.contradictions],
        }
    
    def save(self, output_path: str) -> None:
        """Save report to JSON file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Contradiction report saved to {output_path}")


class ContradictionDetector:
    """Detects metric contradictions and inconsistencies."""
    
    def __init__(self):
        self.report = ContradictionReport()
        logger.info("ContradictionDetector initialized")
    
    def check_recovery_upr_consistency(
        self,
        steps_with_recovery: int,
        upr_recovery_based: float,
        total_promoted_bytes: int,
    ) -> bool:
        """
        Check C1: Recovery events exist but UPR (recovery-based) is 0.
        
        If steps_with_recovery > 0, then upr_recovery_based should be > 0
        (unless recovery happened without any promoted bytes, which is unlikely).
        """
        check_passed = True
        
        if steps_with_recovery > 0 and upr_recovery_based == 0.0 and total_promoted_bytes > 0:
            check_passed = False
            self.report.consistency_checks_failed += 1
            
            contradiction = Contradiction(
                contradiction_id="C1",
                name="Zero Recovery-based UPR despite recovery events",
                description=(
                    f"UPR (Recovery-based) is 0.00% but there are {steps_with_recovery} "
                    f"recovery events with {total_promoted_bytes} promoted bytes. "
                    "This implies recovery happened but no promoted chunk is credited."
                ),
                severity="CRITICAL",
                observed_values={
                    "steps_with_recovery": steps_with_recovery,
                    "upr_recovery_based": upr_recovery_based,
                    "total_promoted_bytes": total_promoted_bytes,
                },
                expected_relationship=(
                    "If steps_with_recovery > 0 and total_promoted_bytes > 0, "
                    "then upr_recovery_based should be > 0 (some promoted bytes "
                    "should be attributed to recovery)."
                ),
                possible_causes=[
                    "Promoted chunks not linked to recovery events (most likely)",
                    "Recovery event detection broken",
                    "Chunk ID mismatch between promotion and recovery tracking",
                    "Only center chunks counted, burst-expanded chunks ignored",
                    "Recovery attribution logic missing in evaluator",
                ],
                suggested_fixes=[
                    "Implement _link_promotions_to_recovery_events() in evaluator",
                    "Verify chunk ID consistency across pipeline stages",
                    "Check that recovery events are properly recorded",
                    "Ensure burst-expanded chunks are included in attribution",
                ],
            )
            self.report.contradictions.append(contradiction)
            logger.warning(f"[C1] {contradiction.description}")
        else:
            self.report.consistency_checks_passed += 1
        
        return check_passed
    
    def check_miss_consistency(
        self,
        steps_with_gold: int,
        steps_with_recovery: int,
        total_misses: int,
    ) -> bool:
        """
        Check C2: Total misses is 0 despite unrecovered gold steps.
        
        If steps_with_gold > steps_with_recovery, then there are gold-step misses.
        These should be reflected in failure attribution unless gold never reached candidates.
        """
        check_passed = True
        gold_step_misses = steps_with_gold - steps_with_recovery
        
        if gold_step_misses > 0 and total_misses == 0:
            check_passed = False
            self.report.consistency_checks_failed += 1
            
            contradiction = Contradiction(
                contradiction_id="C2",
                name="Zero Total Misses despite unrecovered gold steps",
                description=(
                    f"Total Misses = 0 but there are {gold_step_misses} gold-step misses "
                    f"({steps_with_gold} steps with gold, {steps_with_recovery} steps with recovery). "
                    "This may indicate that failure attribution is not capturing all miss types."
                ),
                severity="HIGH",
                observed_values={
                    "steps_with_gold": steps_with_gold,
                    "steps_with_recovery": steps_with_recovery,
                    "gold_step_misses": gold_step_misses,
                    "total_misses": total_misses,
                },
                expected_relationship=(
                    "If gold_step_misses > 0, then total_misses should be >= 0, "
                    "and failure_attribution.unrecovered_gold_step should be set. "
                    "Note: This is not strictly a contradiction if gold never made it to candidates."
                ),
                possible_causes=[
                    "Miss counting only tracks system-attributed misses, not gold-step misses",
                    "Failure attribution not capturing all miss types",
                    "Gold-step misses categorized differently than pipeline misses (valid)",
                    "Candidate recall failure not tracked as a 'miss' in current logic",
                ],
                suggested_fixes=[
                    "Add unrecovered_gold_step category to failure attribution",
                    "Distinguish between gold-step misses and system-attributed misses",
                    "Document that total_misses != gold_step_misses is valid if candidates missed",
                ],
            )
            self.report.contradictions.append(contradiction)
            logger.warning(f"[C2] {contradiction.description}")
        else:
            self.report.consistency_checks_passed += 1
        
        return check_passed
    
    def check_gold_upr_consistency(
        self,
        conditional_recovery: float,
        upr_gold_based: float,
        steps_with_gold: int,
    ) -> bool:
        """
        Check C3: Gold-based UPR is 0 despite positive Conditional Recovery.
        
        If Conditional Recovery > 0, then some gold chunks were recovered.
        These should be marked as useful in gold-based UPR.
        """
        check_passed = True
        
        if conditional_recovery > 0.0 and upr_gold_based == 0.0 and steps_with_gold > 0:
            check_passed = False
            self.report.consistency_checks_failed += 1
            
            contradiction = Contradiction(
                contradiction_id="C3",
                name="Zero Gold-based UPR despite positive Conditional Recovery",
                description=(
                    f"UPR (Gold-based) = 0.00% but Conditional Recovery = {conditional_recovery:.2%}. "
                    "Some gold chunks were recovered but marked as 0% useful."
                ),
                severity="HIGH",
                observed_values={
                    "conditional_recovery": conditional_recovery,
                    "upr_gold_based": upr_gold_based,
                    "steps_with_gold": steps_with_gold,
                },
                expected_relationship=(
                    "If conditional_recovery > 0, then upr_gold_based should be > 0 "
                    "(recovered gold chunks should count as useful)."
                ),
                possible_causes=[
                    "Gold overlap detection not working",
                    "Token range mismatch between gold and promoted chunks",
                    "Chunk ID format mismatch",
                    "Gold chunks not properly tracked in promotion records",
                ],
                suggested_fixes=[
                    "Verify gold overlap computation logic",
                    "Check that gold chunk IDs match promoted chunk IDs",
                    "Add debug logging to trace gold chunk flow",
                ],
            )
            self.report.contradictions.append(contradiction)
            logger.warning(f"[C3] {contradiction.description}")
        else:
            self.report.consistency_checks_passed += 1
        
        return check_passed
    
    def check_upr_bounds(
        self,
        upr_attention: float,
        upr_gold: float,
        upr_recovery: float,
    ) -> bool:
        """
        Check C4: UPR values are within valid bounds [0, 1].
        """
        check_passed = True
        
        for name, value in [
            ("attention", upr_attention),
            ("gold", upr_gold),
            ("recovery", upr_recovery),
        ]:
            if value < 0.0 or value > 1.0:
                check_passed = False
                self.report.consistency_checks_failed += 1
                
                contradiction = Contradiction(
                    contradiction_id=f"C4-{name}",
                    name=f"UPR ({name}) out of valid range",
                    description=f"UPR ({name}) = {value}, which is outside valid range [0, 1].",
                    severity="CRITICAL",
                    observed_values={f"upr_{name}_based": value},
                    expected_relationship="0.0 <= UPR <= 1.0",
                    possible_causes=[
                        "Bytes calculation error",
                        "Integer overflow/underflow",
                        "Division by zero not handled correctly",
                    ],
                    suggested_fixes=[
                        "Clamp UPR values to [0, 1] range",
                        "Check bytes calculation logic",
                    ],
                )
                self.report.contradictions.append(contradiction)
                logger.error(f"[C4-{name}] {contradiction.description}")
        
        if check_passed:
            self.report.consistency_checks_passed += 1
        
        return check_passed
    
    def check_bytes_consistency(
        self,
        total_promoted_bytes: int,
        useful_attention: int,
        useful_gold: int,
        useful_recovery: int,
    ) -> bool:
        """
        Check C5: Useful bytes do not exceed total promoted bytes.
        """
        check_passed = True
        
        for name, useful_bytes in [
            ("attention", useful_attention),
            ("gold", useful_gold),
            ("recovery", useful_recovery),
        ]:
            if useful_bytes > total_promoted_bytes:
                check_passed = False
                self.report.consistency_checks_failed += 1
                
                contradiction = Contradiction(
                    contradiction_id=f"C5-{name}",
                    name=f"Useful bytes ({name}) exceed total promoted bytes",
                    description=(
                        f"Useful bytes ({name}) = {useful_bytes} > "
                        f"total_promoted_bytes = {total_promoted_bytes}"
                    ),
                    severity="CRITICAL",
                    observed_values={
                        "total_promoted_bytes": total_promoted_bytes,
                        f"useful_{name}_bytes": useful_bytes,
                    },
                    expected_relationship="useful_bytes <= total_promoted_bytes",
                    possible_causes=[
                        "Double counting of useful bytes",
                        "Bytes calculation error",
                        "Chunk bytes not properly tracked",
                    ],
                    suggested_fixes=[
                        "Ensure each chunk's bytes are only counted once",
                        "Verify bytes tracking in promotion pipeline",
                    ],
                )
                self.report.contradictions.append(contradiction)
                logger.error(f"[C5-{name}] {contradiction.description}")
        
        if check_passed:
            self.report.consistency_checks_passed += 1
        
        return check_passed
    
    def generate_report(
        self,
        evaluation_result: Any,  # EvaluationResult
        output_path: Optional[str] = None,
    ) -> ContradictionReport:
        """
        Generate full contradiction report from evaluation result.
        """
        # Run all checks
        self.check_recovery_upr_consistency(
            steps_with_recovery=evaluation_result.steps_with_recovery,
            upr_recovery_based=evaluation_result.upr_recovery_based,
            total_promoted_bytes=evaluation_result.total_promoted_bytes,
        )
        
        self.check_miss_consistency(
            steps_with_gold=evaluation_result.steps_with_gold,
            steps_with_recovery=evaluation_result.steps_with_recovery,
            total_misses=evaluation_result.failure_attribution.total_misses,
        )
        
        self.check_gold_upr_consistency(
            conditional_recovery=evaluation_result.conditional_recovery,
            upr_gold_based=evaluation_result.upr_gold_based,
            steps_with_gold=evaluation_result.steps_with_gold,
        )
        
        self.check_upr_bounds(
            upr_attention=evaluation_result.upr_attention_based,
            upr_gold=evaluation_result.upr_gold_based,
            upr_recovery=evaluation_result.upr_recovery_based,
        )
        
        # Note: We don't have access to useful_bytes_* directly in EvaluationResult
        # Those would need to be added or we check via the upr values
        
        if output_path:
            self.report.save(output_path)
        
        return self.report


def generate_contradiction_report(
    result: Any,
    output_path: str = "outputs/reports/contradiction_report.json",
) -> Dict[str, Any]:
    """
    Convenience function to generate contradiction report.
    """
    detector = ContradictionDetector()
    report = detector.generate_report(result, output_path)
    return report.to_dict()
