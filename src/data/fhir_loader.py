"""
FHIR (Fast Healthcare Interoperability Resources) Loader

Loads clinical records in FHIR format for integration with EEG data
Supports: Patient demographics, observations, medications, conditions
"""

from fhir.resources.patient import Patient
from fhir.resources.observation import Observation
from fhir.resources.condition import Condition
from fhir.resources.medicationstatement import MedicationStatement
from fhir.resources.bundle import Bundle
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class FHIRDataLoader:
    """
    Load and parse FHIR clinical records

    Extracts:
    - Patient demographics
    - Medical conditions (diagnoses)
    - Medications
    - Observations (vitals, labs)
    - Procedures
    """

    def __init__(self):
        """Initialize FHIR loader"""
        pass

    def load_patient_bundle(self, file_path: str) -> Dict:
        """
        Load FHIR bundle from JSON file

        Args:
            file_path: Path to FHIR JSON bundle

        Returns:
            Parsed patient data dictionary
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"FHIR file not found: {file_path}")

        with open(file_path, 'r') as f:
            bundle_data = json.load(f)

        # Parse bundle
        try:
            bundle = Bundle.parse_obj(bundle_data)
        except Exception as e:
            logger.error(f"Error parsing FHIR bundle: {e}")
            raise

        # Extract resources
        patient_data = {
            'demographics': None,
            'conditions': [],
            'medications': [],
            'observations': [],
        }

        for entry in bundle.entry or []:
            resource = entry.resource

            if resource.resource_type == 'Patient':
                patient_data['demographics'] = self._parse_patient(resource)

            elif resource.resource_type == 'Condition':
                patient_data['conditions'].append(self._parse_condition(resource))

            elif resource.resource_type == 'MedicationStatement':
                patient_data['medications'].append(self._parse_medication(resource))

            elif resource.resource_type == 'Observation':
                patient_data['observations'].append(self._parse_observation(resource))

        return patient_data

    def _parse_patient(self, patient: Patient) -> Dict:
        """Parse patient demographics"""

        demographics = {
            'id': patient.id,
            'birth_date': str(patient.birthDate) if patient.birthDate else None,
            'gender': patient.gender,
            'age': self._calculate_age(patient.birthDate) if patient.birthDate else None,
        }

        # Name
        if patient.name and len(patient.name) > 0:
            name = patient.name[0]
            demographics['family_name'] = name.family
            demographics['given_name'] = ' '.join(name.given) if name.given else None

        # Address
        if patient.address and len(patient.address) > 0:
            address = patient.address[0]
            demographics['city'] = address.city
            demographics['state'] = address.state
            demographics['country'] = address.country

        return demographics

    def _parse_condition(self, condition: Condition) -> Dict:
        """Parse clinical condition (diagnosis)"""

        cond_data = {
            'id': condition.id,
            'clinical_status': None,
            'verification_status': None,
            'code': None,
            'display': None,
            'onset_date': None,
            'recorded_date': None,
        }

        # Clinical status
        if condition.clinicalStatus:
            cond_data['clinical_status'] = condition.clinicalStatus.coding[0].code if condition.clinicalStatus.coding else None

        # Verification status
        if condition.verificationStatus:
            cond_data['verification_status'] = condition.verificationStatus.coding[0].code if condition.verificationStatus.coding else None

        # Condition code (ICD-10, SNOMED-CT, etc.)
        if condition.code and condition.code.coding:
            coding = condition.code.coding[0]
            cond_data['code'] = coding.code
            cond_data['display'] = coding.display
            cond_data['system'] = coding.system

        # Onset date
        if condition.onsetDateTime:
            cond_data['onset_date'] = str(condition.onsetDateTime)

        # Recorded date
        if condition.recordedDate:
            cond_data['recorded_date'] = str(condition.recordedDate)

        return cond_data

    def _parse_medication(self, med_statement: MedicationStatement) -> Dict:
        """Parse medication statement"""

        med_data = {
            'id': med_statement.id,
            'status': med_statement.status,
            'medication_code': None,
            'medication_display': None,
            'dosage': None,
            'effective_date': None,
        }

        # Medication code
        if med_statement.medicationCodeableConcept:
            if med_statement.medicationCodeableConcept.coding:
                coding = med_statement.medicationCodeableConcept.coding[0]
                med_data['medication_code'] = coding.code
                med_data['medication_display'] = coding.display

        # Dosage
        if med_statement.dosage and len(med_statement.dosage) > 0:
            dosage = med_statement.dosage[0]
            if dosage.text:
                med_data['dosage'] = dosage.text
            elif dosage.doseAndRate:
                dose = dosage.doseAndRate[0]
                if dose.doseQuantity:
                    med_data['dosage'] = f"{dose.doseQuantity.value} {dose.doseQuantity.unit}"

        # Effective date
        if med_statement.effectiveDateTime:
            med_data['effective_date'] = str(med_statement.effectiveDateTime)

        return med_data

    def _parse_observation(self, observation: Observation) -> Dict:
        """Parse clinical observation"""

        obs_data = {
            'id': observation.id,
            'status': observation.status,
            'code': None,
            'display': None,
            'value': None,
            'unit': None,
            'effective_date': None,
            'category': None,
        }

        # Observation code
        if observation.code and observation.code.coding:
            coding = observation.code.coding[0]
            obs_data['code'] = coding.code
            obs_data['display'] = coding.display

        # Category
        if observation.category and len(observation.category) > 0:
            if observation.category[0].coding:
                obs_data['category'] = observation.category[0].coding[0].code

        # Value
        if observation.valueQuantity:
            obs_data['value'] = observation.valueQuantity.value
            obs_data['unit'] = observation.valueQuantity.unit
        elif observation.valueString:
            obs_data['value'] = observation.valueString
        elif observation.valueCodeableConcept:
            if observation.valueCodeableConcept.coding:
                obs_data['value'] = observation.valueCodeableConcept.coding[0].display

        # Effective date
        if observation.effectiveDateTime:
            obs_data['effective_date'] = str(observation.effectiveDateTime)

        return obs_data

    def _calculate_age(self, birth_date) -> int:
        """Calculate age from birth date"""
        today = datetime.now()
        birth = datetime.fromisoformat(str(birth_date))
        age = today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
        return age

    def create_clinical_summary(self, patient_data: Dict) -> str:
        """
        Create a clinical summary text for use with Llama

        Args:
            patient_data: Parsed FHIR patient data

        Returns:
            Clinical summary string
        """
        summary = []

        # Demographics
        if patient_data['demographics']:
            demo = patient_data['demographics']
            summary.append("**Patient Demographics:**")
            summary.append(f"Age: {demo.get('age', 'Unknown')}")
            summary.append(f"Gender: {demo.get('gender', 'Unknown')}")

        # Active conditions
        active_conditions = [
            c for c in patient_data['conditions']
            if c.get('clinical_status') == 'active'
        ]
        if active_conditions:
            summary.append("\n**Active Diagnoses:**")
            for cond in active_conditions:
                summary.append(f"- {cond.get('display', 'Unknown')} ({cond.get('code', 'N/A')})")

        # Current medications
        active_meds = [
            m for m in patient_data['medications']
            if m.get('status') == 'active'
        ]
        if active_meds:
            summary.append("\n**Current Medications:**")
            for med in active_meds:
                dosage = f" - {med.get('dosage')}" if med.get('dosage') else ""
                summary.append(f"- {med.get('medication_display', 'Unknown')}{dosage}")

        # Recent observations
        if patient_data['observations']:
            summary.append("\n**Recent Observations:**")
            for obs in patient_data['observations'][:5]:  # Last 5
                value = obs.get('value', 'N/A')
                unit = obs.get('unit', '')
                summary.append(f"- {obs.get('display', 'Unknown')}: {value} {unit}")

        return '\n'.join(summary)

    def extract_medical_history(self, patient_data: Dict) -> Dict:
        """
        Extract structured medical history for model input

        Args:
            patient_data: Parsed FHIR patient data

        Returns:
            Structured medical history dictionary
        """
        history = {
            'age': patient_data['demographics'].get('age') if patient_data['demographics'] else None,
            'gender': patient_data['demographics'].get('gender') if patient_data['demographics'] else None,
            'diagnoses': [],
            'medications': [],
            'comorbidities': [],
        }

        # Extract diagnoses
        for cond in patient_data['conditions']:
            history['diagnoses'].append({
                'name': cond.get('display'),
                'code': cond.get('code'),
                'status': cond.get('clinical_status'),
            })

        # Extract medications
        for med in patient_data['medications']:
            history['medications'].append({
                'name': med.get('medication_display'),
                'dosage': med.get('dosage'),
                'status': med.get('status'),
            })

        # Identify comorbidities (conditions that are not the primary)
        history['comorbidities'] = [
            cond.get('display') for cond in patient_data['conditions']
            if cond.get('clinical_status') == 'active'
        ]

        return history


def create_sample_fhir_bundle() -> Dict:
    """Create a sample FHIR bundle for testing"""

    bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "id": "patient-001",
                    "name": [{"family": "Doe", "given": ["John"]}],
                    "gender": "male",
                    "birthDate": "1985-03-15"
                }
            },
            {
                "resource": {
                    "resourceType": "Condition",
                    "id": "condition-001",
                    "clinicalStatus": {
                        "coding": [{"code": "active"}]
                    },
                    "code": {
                        "coding": [{
                            "system": "http://snomed.info/sct",
                            "code": "84757009",
                            "display": "Epilepsy"
                        }]
                    },
                    "onsetDateTime": "2020-01-01"
                }
            },
            {
                "resource": {
                    "resourceType": "MedicationStatement",
                    "id": "med-001",
                    "status": "active",
                    "medicationCodeableConcept": {
                        "coding": [{
                            "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                            "code": "114477",
                            "display": "Levetiracetam"
                        }]
                    },
                    "dosage": [{
                        "text": "500mg twice daily"
                    }]
                }
            }
        ]
    }

    return bundle


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test with sample data
    loader = FHIRDataLoader()

    # Create sample bundle
    sample_bundle = create_sample_fhir_bundle()

    # Save to file
    sample_file = Path("sample_fhir_bundle.json")
    with open(sample_file, 'w') as f:
        json.dump(sample_bundle, f, indent=2)

    # Load and parse
    patient_data = loader.load_patient_bundle(str(sample_file))

    # Create summary
    summary = loader.create_clinical_summary(patient_data)
    print("="*80)
    print("CLINICAL SUMMARY:")
    print(summary)
    print("="*80)

    # Clean up
    sample_file.unlink()

    print("\nFHIR loader ready")
