"""
Drug Interaction Checking Module
Detects potential drug-drug interactions for patient safety
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class InteractionSeverity(Enum):
    """Severity levels for drug interactions"""
    CONTRAINDICATED = "contraindicated"  # Never use together
    MAJOR = "major"  # May cause serious harm
    MODERATE = "moderate"  # May require monitoring
    MINOR = "minor"  # Usually clinically insignificant


@dataclass
class DrugInteraction:
    """Drug interaction information"""
    drug1: str
    drug2: str
    severity: InteractionSeverity
    description: str
    mechanism: str
    management: str
    evidence_level: str  # "Excellent", "Good", "Fair"


class DrugInteractionChecker:
    """
    Check for drug-drug interactions

    Data sources:
    - DrugBank
    - FDA drug interaction tables
    - Clinical guidelines
    """

    def __init__(self):
        """Initialize interaction database"""
        self.interactions_db = self._load_interactions_database()

    def _load_interactions_database(self) -> Dict[Tuple[str, str], DrugInteraction]:
        """
        Load drug interaction database

        In production, load from:
        - DrugBank API
        - RxNorm
        - FDA tables
        """
        # Simplified antiepileptic drug interactions
        interactions = {}

        # Levetiracetam interactions
        interactions[("levetiracetam", "valproic_acid")] = DrugInteraction(
            drug1="Levetiracetam",
            drug2="Valproic Acid",
            severity=InteractionSeverity.MODERATE,
            description="Valproic acid may increase levetiracetam levels",
            mechanism="Possible competition for renal tubular secretion",
            management="Monitor for levetiracetam toxicity; may need dose adjustment",
            evidence_level="Fair"
        )

        # Phenytoin interactions
        interactions[("phenytoin", "warfarin")] = DrugInteraction(
            drug1="Phenytoin",
            drug2="Warfarin",
            severity=InteractionSeverity.MAJOR,
            description="Phenytoin increases warfarin metabolism, decreasing anticoagulant effect",
            mechanism="CYP450 enzyme induction",
            management="Monitor INR closely; warfarin dose may need to be increased",
            evidence_level="Excellent"
        )

        interactions[("phenytoin", "oral_contraceptives")] = DrugInteraction(
            drug1="Phenytoin",
            drug2="Oral Contraceptives",
            severity=InteractionSeverity.MAJOR,
            description="Phenytoin reduces effectiveness of oral contraceptives",
            mechanism="CYP450 enzyme induction increases hormone metabolism",
            management="Use alternative or additional contraceptive methods",
            evidence_level="Excellent"
        )

        # Carbamazepine interactions
        interactions[("carbamazepine", "simvastatin")] = DrugInteraction(
            drug1="Carbamazepine",
            drug2="Simvastatin",
            severity=InteractionSeverity.MAJOR,
            description="Carbamazepine significantly reduces simvastatin levels",
            mechanism="CYP3A4 induction",
            management="Consider alternative statin (e.g., pravastatin) or increase dose",
            evidence_level="Good"
        )

        interactions[("carbamazepine", "grapefruit_juice")] = DrugInteraction(
            drug1="Carbamazepine",
            drug2="Grapefruit Juice",
            severity=InteractionSeverity.MODERATE,
            description="Grapefruit juice may increase carbamazepine levels",
            mechanism="CYP3A4 inhibition",
            management="Avoid grapefruit juice; monitor for toxicity",
            evidence_level="Good"
        )

        # Valproic acid interactions
        interactions[("valproic_acid", "lamotrigine")] = DrugInteraction(
            drug1="Valproic Acid",
            drug2="Lamotrigine",
            severity=InteractionSeverity.MAJOR,
            description="Valproic acid doubles lamotrigine levels, increasing rash risk",
            mechanism="Inhibition of lamotrigine glucuronidation",
            management="Reduce lamotrigine dose by 50%; slow titration",
            evidence_level="Excellent"
        )

        interactions[("valproic_acid", "aspirin")] = DrugInteraction(
            drug1="Valproic Acid",
            drug2="Aspirin",
            severity=InteractionSeverity.MODERATE,
            description="Aspirin may increase free valproic acid levels and bleeding risk",
            mechanism="Protein binding displacement and platelet inhibition",
            management="Monitor valproic acid levels; watch for bleeding",
            evidence_level="Good"
        )

        # Lamotrigine interactions
        interactions[("lamotrigine", "oral_contraceptives")] = DrugInteraction(
            drug1="Lamotrigine",
            drug2="Oral Contraceptives",
            severity=InteractionSeverity.MODERATE,
            description="Oral contraceptives reduce lamotrigine levels by 50%",
            mechanism="Induction of lamotrigine glucuronidation",
            management="May need to increase lamotrigine dose; monitor seizure control",
            evidence_level="Excellent"
        )

        # Benzodiazepine interactions
        interactions[("diazepam", "alcohol")] = DrugInteraction(
            drug1="Diazepam",
            drug2="Alcohol",
            severity=InteractionSeverity.CONTRAINDICATED,
            description="Combined CNS depression can be fatal",
            mechanism="Additive CNS depressant effects",
            management="Avoid alcohol completely; warn patient of risks",
            evidence_level="Excellent"
        )

        interactions[("clonazepam", "opioids")] = DrugInteraction(
            drug1="Clonazepam",
            drug2="Opioids",
            severity=InteractionSeverity.MAJOR,
            description="Increased risk of respiratory depression and death",
            mechanism="Additive respiratory depressant effects",
            management="Avoid combination; if necessary, use lowest doses and monitor closely",
            evidence_level="Excellent"
        )

        return interactions

    def check_interactions(
        self,
        medications: List[str],
        include_minor: bool = False
    ) -> List[DrugInteraction]:
        """
        Check for interactions in a medication list

        Args:
            medications: List of medication names (generic names)
            include_minor: Include minor interactions

        Returns:
            List of detected interactions
        """
        detected_interactions = []

        # Normalize medication names
        meds_normalized = [self._normalize_drug_name(med) for med in medications]

        # Check all pairs
        for i, med1 in enumerate(meds_normalized):
            for med2 in meds_normalized[i+1:]:
                # Check both directions
                interaction = self._get_interaction(med1, med2)

                if interaction:
                    # Filter by severity if needed
                    if not include_minor and interaction.severity == InteractionSeverity.MINOR:
                        continue

                    detected_interactions.append(interaction)
                    logger.warning(
                        f"Interaction detected: {interaction.drug1} + {interaction.drug2} "
                        f"({interaction.severity.value})"
                    )

        return detected_interactions

    def _get_interaction(self, drug1: str, drug2: str) -> Optional[DrugInteraction]:
        """Get interaction between two drugs"""
        # Check both orderings
        interaction = self.interactions_db.get((drug1, drug2))
        if not interaction:
            interaction = self.interactions_db.get((drug2, drug1))
        return interaction

    def _normalize_drug_name(self, drug_name: str) -> str:
        """Normalize drug name to generic name (lowercase, no spaces)"""
        # Simple normalization (in production, use RxNorm API)
        return drug_name.lower().replace(" ", "_").replace("-", "_")

    def get_interaction_summary(
        self,
        interactions: List[DrugInteraction]
    ) -> Dict[str, any]:
        """
        Generate summary of interactions

        Args:
            interactions: List of interactions

        Returns:
            Summary dictionary
        """
        summary = {
            "total_interactions": len(interactions),
            "contraindicated": 0,
            "major": 0,
            "moderate": 0,
            "minor": 0,
            "highest_severity": None,
            "requires_action": False
        }

        for interaction in interactions:
            if interaction.severity == InteractionSeverity.CONTRAINDICATED:
                summary["contraindicated"] += 1
                summary["requires_action"] = True
            elif interaction.severity == InteractionSeverity.MAJOR:
                summary["major"] += 1
                summary["requires_action"] = True
            elif interaction.severity == InteractionSeverity.MODERATE:
                summary["moderate"] += 1
            elif interaction.severity == InteractionSeverity.MINOR:
                summary["minor"] += 1

        # Determine highest severity
        if summary["contraindicated"] > 0:
            summary["highest_severity"] = "CONTRAINDICATED"
        elif summary["major"] > 0:
            summary["highest_severity"] = "MAJOR"
        elif summary["moderate"] > 0:
            summary["highest_severity"] = "MODERATE"
        elif summary["minor"] > 0:
            summary["highest_severity"] = "MINOR"

        return summary


def check_drug_interactions(medications: List[str]) -> Tuple[List[DrugInteraction], Dict]:
    """
    Convenience function to check drug interactions

    Args:
        medications: List of medication names

    Returns:
        Tuple of (interactions, summary)
    """
    checker = DrugInteractionChecker()
    interactions = checker.check_interactions(medications, include_minor=True)
    summary = checker.get_interaction_summary(interactions)
    return interactions, summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test drug interactions
    medications = [
        "Levetiracetam",
        "Valproic Acid",
        "Lamotrigine"
    ]

    interactions, summary = check_drug_interactions(medications)

    print("="*80)
    print("DRUG INTERACTION CHECK")
    print("="*80)
    print(f"Medications: {', '.join(medications)}")
    print(f"\nInteractions found: {summary['total_interactions']}")
    print(f"Highest severity: {summary['highest_severity']}")
    print(f"Requires action: {summary['requires_action']}")

    for interaction in interactions:
        print(f"\n{interaction.drug1} + {interaction.drug2}")
        print(f"  Severity: {interaction.severity.value.upper()}")
        print(f"  {interaction.description}")
        print(f"  Management: {interaction.management}")
