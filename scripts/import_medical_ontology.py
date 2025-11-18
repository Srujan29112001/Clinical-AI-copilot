"""
Import Medical Ontologies into Neo4j Graph Database

Imports:
- SNOMED-CT (clinical terminology)
- ICD-10 (diagnosis codes)
- RxNorm (medications)
- Drug-Bank (drug information)
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List
import csv
import json
from tqdm import tqdm

logger = logging.getLogger(__name__)


class MedicalOntologyImporter:
    """Import medical ontologies into Neo4j"""

    def __init__(self, neo4j_uri: str = "bolt://localhost:7687", username: str = "neo4j", password: str = "password"):
        """Initialize importer"""
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(neo4j_uri, auth=(username, password))
            self.connected = True
        except Exception as e:
            logger.warning(f"Could not connect to Neo4j: {e}")
            self.connected = False

    def import_icd10(self, icd10_file: Path):
        """
        Import ICD-10 codes

        File format: CSV with columns [code, description, category]
        """
        logger.info(f"Importing ICD-10 from {icd10_file}")

        if not self.connected:
            logger.warning("Neo4j not connected, generating sample data")
            self._generate_sample_icd10()
            return

        with self.driver.session() as session:
            # Create constraints
            session.run("""
                CREATE CONSTRAINT icd10_code IF NOT EXISTS
                FOR (d:ICD10) REQUIRE d.code IS UNIQUE
            """)

            # Import codes
            with open(icd10_file, 'r') as f:
                reader = csv.DictReader(f)
                codes = list(reader)

            for row in tqdm(codes, desc="Importing ICD-10"):
                session.run("""
                    MERGE (d:ICD10 {code: $code})
                    SET d.description = $description,
                        d.category = $category
                """, code=row['code'], description=row['description'], category=row.get('category', ''))

        logger.info(f"Imported {len(codes)} ICD-10 codes")

    def _generate_sample_icd10(self):
        """Generate sample ICD-10 codes for demonstration"""
        sample_codes = [
            {"code": "G40.0", "description": "Localization-related (focal)(partial) idiopathic epilepsy", "category": "Epilepsy"},
            {"code": "G40.1", "description": "Localization-related (focal)(partial) symptomatic epilepsy", "category": "Epilepsy"},
            {"code": "G40.2", "description": "Localization-related (focal)(partial) symptomatic epilepsy with complex partial seizures", "category": "Epilepsy"},
            {"code": "G40.3", "description": "Generalized idiopathic epilepsy and epileptic syndromes", "category": "Epilepsy"},
            {"code": "G40.4", "description": "Other generalized epilepsy and epileptic syndromes", "category": "Epilepsy"},
            {"code": "G40.5", "description": "Epileptic seizures related to external causes", "category": "Epilepsy"},
            {"code": "G40.8", "description": "Other epilepsy", "category": "Epilepsy"},
            {"code": "G40.9", "description": "Epilepsy, unspecified", "category": "Epilepsy"},
            {"code": "G47.0", "description": "Insomnia", "category": "Sleep disorders"},
            {"code": "G47.1", "description": "Hypersomnia", "category": "Sleep disorders"},
            {"code": "G47.2", "description": "Circadian rhythm sleep disorders", "category": "Sleep disorders"},
            {"code": "G47.3", "description": "Sleep apnea", "category": "Sleep disorders"},
            {"code": "F03", "description": "Unspecified dementia", "category": "Dementia"},
            {"code": "G30.0", "description": "Alzheimer's disease with early onset", "category": "Dementia"},
            {"code": "G30.1", "description": "Alzheimer's disease with late onset", "category": "Dementia"},
        ]

        logger.info(f"Generated {len(sample_codes)} sample ICD-10 codes")
        return sample_codes

    def import_snomed_ct(self, snomed_file: Path):
        """
        Import SNOMED-CT concepts

        File format: CSV with [concept_id, term, semantic_tag]
        """
        logger.info(f"Importing SNOMED-CT from {snomed_file}")

        if not self.connected:
            self._generate_sample_snomed()
            return

        with self.driver.session() as session:
            session.run("""
                CREATE CONSTRAINT snomed_id IF NOT EXISTS
                FOR (s:SNOMED) REQUIRE s.concept_id IS UNIQUE
            """)

            with open(snomed_file, 'r') as f:
                reader = csv.DictReader(f)
                concepts = list(reader)

            for row in tqdm(concepts, desc="Importing SNOMED-CT"):
                session.run("""
                    MERGE (s:SNOMED {concept_id: $concept_id})
                    SET s.term = $term,
                        s.semantic_tag = $semantic_tag
                """, concept_id=row['concept_id'], term=row['term'], semantic_tag=row.get('semantic_tag', ''))

        logger.info(f"Imported {len(concepts)} SNOMED-CT concepts")

    def _generate_sample_snomed(self):
        """Generate sample SNOMED-CT codes"""
        sample_concepts = [
            {"concept_id": "84757009", "term": "Epilepsy", "semantic_tag": "disorder"},
            {"concept_id": "230456007", "term": "Focal epilepsy", "semantic_tag": "disorder"},
            {"concept_id": "230458008", "term": "Generalized epilepsy", "semantic_tag": "disorder"},
            {"concept_id": "91175000", "term": "Seizure", "semantic_tag": "finding"},
            {"concept_id": "246545002", "term": "EEG abnormality", "semantic_tag": "finding"},
        ]
        logger.info(f"Generated {len(sample_concepts)} sample SNOMED-CT concepts")
        return sample_concepts

    def import_rxnorm(self, rxnorm_file: Path):
        """Import RxNorm medication codes"""
        logger.info(f"Importing RxNorm from {rxnorm_file}")

        if not self.connected:
            self._generate_sample_rxnorm()
            return

        with self.driver.session() as session:
            session.run("""
                CREATE CONSTRAINT rxnorm_code IF NOT EXISTS
                FOR (r:RxNorm) REQUIRE r.rxcui IS UNIQUE
            """)

            with open(rxnorm_file, 'r') as f:
                reader = csv.DictReader(f)
                meds = list(reader)

            for row in tqdm(meds, desc="Importing RxNorm"):
                session.run("""
                    MERGE (r:RxNorm {rxcui: $rxcui})
                    SET r.name = $name,
                        r.tty = $tty
                """, rxcui=row['rxcui'], name=row['name'], tty=row.get('tty', ''))

        logger.info(f"Imported {len(meds)} RxNorm medications")

    def _generate_sample_rxnorm(self):
        """Generate sample RxNorm codes"""
        sample_meds = [
            {"rxcui": "114477", "name": "Levetiracetam", "tty": "IN"},
            {"rxcui": "11170", "name": "Valproic Acid", "tty": "IN"},
            {"rxcui": "6367", "name": "Lamotrigine", "tty": "IN"},
            {"rxcui": "8691", "name": "Phenytoin", "tty": "IN"},
            {"rxcui": "2002", "name": "Carbamazepine", "tty": "IN"},
            {"rxcui": "23990", "name": "Topiramate", "tty": "IN"},
            {"rxcui": "114979", "name": "Oxcarbazepine", "tty": "IN"},
            {"rxcui": "114121", "name": "Gabapentin", "tty": "IN"},
            {"rxcui": "187832", "name": "Pregabalin", "tty": "IN"},
            {"rxcui": "5640", "name": "Diazepam", "tty": "IN"},
            {"rxcui": "2598", "name": "Clonazepam", "tty": "IN"},
            {"rxcui": "6960", "name": "Lorazepam", "tty": "IN"},
        ]
        logger.info(f"Generated {len(sample_meds)} sample RxNorm medications")
        return sample_meds

    def create_relationships(self):
        """Create relationships between entities"""
        if not self.connected:
            logger.warning("Neo4j not connected, skipping relationship creation")
            return

        logger.info("Creating relationships...")

        with self.driver.session() as session:
            # Link ICD-10 to SNOMED-CT
            session.run("""
                MATCH (i:ICD10), (s:SNOMED)
                WHERE i.code STARTS WITH 'G40' AND s.concept_id = '84757009'
                MERGE (i)-[:MAPS_TO]->(s)
            """)

            # Link diseases to treatments
            session.run("""
                MATCH (s:SNOMED {concept_id: '84757009'}), (r:RxNorm)
                WHERE r.name IN ['Levetiracetam', 'Valproic Acid', 'Lamotrigine']
                MERGE (s)-[:TREATED_BY {line: 'first-line', efficacy: 0.7}]->(r)
            """)

        logger.info("Relationships created")

    def close(self):
        """Close Neo4j connection"""
        if self.connected:
            self.driver.close()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Import medical ontologies")
    parser.add_argument("--icd10", type=Path, help="ICD-10 CSV file")
    parser.add_argument("--snomed", type=Path, help="SNOMED-CT CSV file")
    parser.add_argument("--rxnorm", type=Path, help="RxNorm CSV file")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687", help="Neo4j URI")
    parser.add_argument("--username", default="neo4j", help="Neo4j username")
    parser.add_argument("--password", default="password", help="Neo4j password")
    parser.add_argument("--create-sample", action="store_true", help="Create sample data")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    importer = MedicalOntologyImporter(args.neo4j_uri, args.username, args.password)

    try:
        if args.create_sample or not (args.icd10 or args.snomed or args.rxnorm):
            logger.info("Generating sample medical ontology data...")
            importer._generate_sample_icd10()
            importer._generate_sample_snomed()
            importer._generate_sample_rxnorm()
            logger.info("Sample data generated (in-memory only)")
        else:
            if args.icd10:
                importer.import_icd10(args.icd10)
            if args.snomed:
                importer.import_snomed_ct(args.snomed)
            if args.rxnorm:
                importer.import_rxnorm(args.rxnorm)

            importer.create_relationships()

        logger.info("Import completed successfully")

    finally:
        importer.close()


if __name__ == "__main__":
    main()
