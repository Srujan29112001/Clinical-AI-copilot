import type { PatientContext } from "./types";

export interface SampleDataset {
  id: string;
  name: string;
  description: string;
  patient: PatientContext;
  profile: "ictal" | "normal" | "sleep";
}

export const SAMPLES: SampleDataset[] = [
  {
    id: "ictal",
    name: "Ictal seizure episode",
    description: "16-channel scalp EEG with rhythmic spike-wave discharge and high-frequency recruitment.",
    profile: "ictal",
    patient: {
      patient_id: "DEMO-ICTAL",
      age: 34,
      sex: "female",
      symptoms: "recurrent generalized seizures with loss of consciousness",
      medications: ["levetiracetam", "lamotrigine"],
    },
  },
  {
    id: "normal",
    name: "Normal awake EEG",
    description: "Healthy adult, eyes-closed posterior dominant alpha rhythm, broadband background.",
    profile: "normal",
    patient: {
      patient_id: "DEMO-NORMAL",
      age: 28,
      sex: "male",
      symptoms: "routine screening, no acute complaints",
      medications: [],
    },
  },
  {
    id: "sleep",
    name: "Sleep / drowsy EEG",
    description: "Theta-dominant drowsy recording with reduced inter-channel synchrony.",
    profile: "sleep",
    patient: {
      patient_id: "DEMO-SLEEP",
      age: 51,
      sex: "female",
      symptoms: "excessive daytime sleepiness, suspected sleep disorder",
      medications: ["melatonin", "phenytoin"],
    },
  },
];
