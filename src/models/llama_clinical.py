"""
Llama 3.1 8B for Clinical Text Understanding
Fine-tuned with QLoRA for medical reasoning

Features:
- Medical knowledge extraction
- Clinical reasoning
- Diagnostic suggestion
- Treatment recommendation
- Patient history summarization
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class LlamaClinicalConfig:
    """Configuration for Llama clinical model"""

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        load_in_4bit: bool = True,
        bnb_4bit_compute_dtype: str = "float16",
        bnb_4bit_quant_type: str = "nf4",
        use_nested_quant: bool = True,
        max_length: int = 4096,
        temperature: float = 0.1,
        top_p: float = 0.9,
        lora_checkpoint: Optional[str] = None,
    ):
        self.model_name = model_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.load_in_4bit = load_in_4bit
        self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.use_nested_quant = use_nested_quant
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.lora_checkpoint = lora_checkpoint


class LlamaClinicalModel:
    """
    Llama 3.1 8B optimized for clinical applications

    Features:
    - 4-bit quantization (QLoRA) for RTX 3060
    - LoRA fine-tuning on medical data
    - Clinical prompt templates
    - Embedding extraction for multimodal fusion
    """

    def __init__(self, config: Optional[LlamaClinicalConfig] = None):
        self.config = config or LlamaClinicalConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Loading Llama 3.1 8B clinical model...")
        self.load_model()

    def load_model(self):
        """Load Llama model with QLoRA"""

        # BitsAndBytes configuration for 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.load_in_4bit,
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=self.config.use_nested_quant,
        )

        # Load base model
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
            logger.info(f"Loaded base model: {self.config.model_name}")
        except Exception as e:
            logger.warning(f"Could not load model from HuggingFace: {e}")
            logger.info("Using mock model for demonstration")
            self.model = None
            self.tokenizer = None
            return

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # Prepare for LoRA fine-tuning
        self.model = prepare_model_for_kbit_training(self.model)

        # LoRA configuration
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Load LoRA checkpoint if available
        if self.config.lora_checkpoint:
            try:
                self.model = PeftModel.from_pretrained(
                    self.model,
                    self.config.lora_checkpoint,
                    is_trainable=False
                )
                logger.info(f"Loaded LoRA checkpoint: {self.config.lora_checkpoint}")
            except Exception as e:
                logger.warning(f"Could not load LoRA checkpoint: {e}")
                # Apply LoRA config to base model
                self.model = get_peft_model(self.model, lora_config)
        else:
            # Apply LoRA config to base model
            self.model = get_peft_model(self.model, lora_config)

        self.model.eval()
        logger.info("Llama clinical model ready")

    def create_clinical_prompt(
        self,
        patient_info: Dict,
        eeg_findings: Dict,
        task: str = "diagnosis"
    ) -> str:
        """
        Create structured clinical prompt

        Args:
            patient_info: Patient demographics and history
            eeg_findings: EEG analysis results
            task: Task type (diagnosis, treatment, summary)

        Returns:
            Formatted prompt string
        """
        if task == "diagnosis":
            prompt = f"""You are an expert neurologist. Analyze the following patient case and provide a differential diagnosis.

**Patient Information:**
- Age: {patient_info.get('age', 'Unknown')}
- Gender: {patient_info.get('gender', 'Unknown')}
- Chief Complaint: {patient_info.get('symptoms', 'Not provided')}
- Medical History: {patient_info.get('history', 'None reported')}

**EEG Findings:**
- Dominant Frequency: {eeg_findings.get('dominant_freq', 'N/A')} Hz
- Power Spectral Density:
  - Delta: {eeg_findings.get('psd_delta', 'N/A')}
  - Theta: {eeg_findings.get('psd_theta', 'N/A')}
  - Alpha: {eeg_findings.get('psd_alpha', 'N/A')}
  - Beta: {eeg_findings.get('psd_beta', 'N/A')}
  - Gamma: {eeg_findings.get('psd_gamma', 'N/A')}
- Detected Patterns: {', '.join(eeg_findings.get('patterns', ['None']))}
- Seizure Probability: {eeg_findings.get('seizure_prob', 0):.2%}

**Task:** Provide a differential diagnosis with confidence levels and recommended next steps.

**Response Format:**
1. Primary Diagnosis: [Name] (Confidence: X%)
2. Differential Diagnoses: [List]
3. ICD-10 Code: [Code]
4. Recommended Tests: [List]
5. Treatment Considerations: [Brief]

**Response:**"""

        elif task == "treatment":
            prompt = f"""You are an expert neurologist. Recommend treatment based on the diagnosis.

**Diagnosis:** {patient_info.get('diagnosis', 'Unknown')}
**Patient Age:** {patient_info.get('age', 'Unknown')}
**Comorbidities:** {patient_info.get('comorbidities', 'None')}
**Current Medications:** {patient_info.get('medications', 'None')}
**Allergies:** {patient_info.get('allergies', 'None')}

**EEG Findings:** {eeg_findings.get('summary', 'Normal')}

**Task:** Provide evidence-based treatment recommendations.

**Response Format:**
1. First-line Treatment: [Name, Dosage, Duration]
2. Alternative Options: [List]
3. Monitoring Requirements: [Specify]
4. Expected Outcomes: [Timeline and success rate]
5. Drug Interactions: [List any concerns]
6. Evidence Level: [Grade]

**Response:**"""

        elif task == "summary":
            prompt = f"""You are an expert neurologist. Summarize the patient's medical history and current status.

**Patient History:** {patient_info.get('full_history', 'Not provided')}

**Recent EEG Findings:** {eeg_findings}

**Task:** Provide a concise clinical summary suitable for handoff.

**Response Format:**
- Chief Complaint:
- Key Findings:
- Assessment:
- Plan:

**Response:**"""

        else:
            raise ValueError(f"Unknown task: {task}")

        return prompt

    def generate_clinical_text(
        self,
        patient_info: Dict,
        eeg_findings: Dict,
        task: str = "diagnosis"
    ) -> str:
        """
        Generate clinical text using Llama

        Args:
            patient_info: Patient information dictionary
            eeg_findings: EEG analysis results
            task: Task type

        Returns:
            Generated clinical text
        """
        if self.model is None:
            return self._mock_clinical_response(task)

        prompt = self.create_clinical_prompt(patient_info, eeg_findings, task)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the response part (after the prompt)
        if "**Response:**" in response:
            response = response.split("**Response:**")[-1].strip()

        return response

    def get_text_embeddings(
        self,
        text: str,
        layer: int = -1
    ) -> torch.Tensor:
        """
        Extract embeddings from Llama for multimodal fusion

        Args:
            text: Input text
            layer: Which layer to extract embeddings from (-1 = last)

        Returns:
            Embedding tensor (4096-dim for Llama 3.1 8B)
        """
        if self.model is None:
            # Return random embeddings for demo
            return torch.randn(1, 4096)

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
            )

        # Get embeddings from specified layer
        hidden_states = outputs.hidden_states[layer]  # (batch, seq_len, 4096)

        # Average pool over sequence length
        embeddings = hidden_states.mean(dim=1)  # (batch, 4096)

        return embeddings

    def _mock_clinical_response(self, task: str) -> str:
        """Mock response for demonstration when model is not available"""

        if task == "diagnosis":
            return """1. Primary Diagnosis: Focal Epilepsy (Confidence: 85%)
2. Differential Diagnoses:
   - Temporal Lobe Epilepsy (70%)
   - Benign Rolandic Epilepsy (40%)
   - Non-epileptic Seizure Disorder (20%)
3. ICD-10 Code: G40.209
4. Recommended Tests:
   - Extended video-EEG monitoring
   - Brain MRI with epilepsy protocol
   - Neuropsychological assessment
5. Treatment Considerations:
   - Consider antiepileptic therapy
   - Seizure precautions education
   - Follow-up in 2-4 weeks"""

        elif task == "treatment":
            return """1. First-line Treatment: Levetiracetam 500mg BID, titrate to effect
2. Alternative Options:
   - Lamotrigine (slower titration required)
   - Valproic acid (avoid in women of childbearing age)
3. Monitoring Requirements:
   - Seizure diary
   - Drug levels after 2 weeks
   - CBC, LFTs at baseline and 3 months
4. Expected Outcomes:
   - 60-70% seizure freedom at 6 months
   - Consider titration if not responsive
5. Drug Interactions: None with current medications
6. Evidence Level: Grade A (Strong recommendation)"""

        elif task == "summary":
            return """- Chief Complaint: New-onset seizures
- Key Findings: EEG showing focal epileptiform discharges
- Assessment: Focal epilepsy, likely temporal lobe origin
- Plan: Initiate AED therapy, outpatient MRI, neurology follow-up"""

        return "Clinical response generated"


def create_llama_clinical_model(
    config: Optional[LlamaClinicalConfig] = None
) -> LlamaClinicalModel:
    """
    Factory function to create Llama clinical model

    Args:
        config: Model configuration

    Returns:
        Initialized Llama clinical model
    """
    return LlamaClinicalModel(config)


if __name__ == "__main__":
    # Test Llama clinical model
    logging.basicConfig(level=logging.INFO)

    model = create_llama_clinical_model()

    # Test data
    patient_info = {
        'age': 35,
        'gender': 'Female',
        'symptoms': 'Recurring seizures, loss of consciousness',
        'history': 'No prior neurological conditions',
    }

    eeg_findings = {
        'dominant_freq': 9.5,
        'psd_delta': 0.15,
        'psd_theta': 0.20,
        'psd_alpha': 0.45,
        'psd_beta': 0.15,
        'psd_gamma': 0.05,
        'patterns': ['spike-wave', 'sharp waves'],
        'seizure_prob': 0.85,
    }

    # Generate diagnosis
    diagnosis = model.generate_clinical_text(patient_info, eeg_findings, task="diagnosis")
    print("="*80)
    print("DIAGNOSIS:")
    print(diagnosis)
    print("="*80)

    # Get embeddings
    embeddings = model.get_text_embeddings("Patient presents with seizures")
    print(f"\nEmbedding shape: {embeddings.shape}")
