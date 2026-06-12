"use client";
import { useEffect, useState } from "react";
import type { AnalysisResult, LLMConfig, PatientContext } from "./types";

const KEYS = { llm: "cac.llm", result: "cac.result", patient: "cac.patient" };

function read<T>(key: string, fallback: T): T {
  if (typeof window === "undefined") return fallback;
  try {
    const v = localStorage.getItem(key);
    return v ? (JSON.parse(v) as T) : fallback;
  } catch {
    return fallback;
  }
}

function write<T>(key: string, value: T) {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(key, JSON.stringify(value));
    window.dispatchEvent(new CustomEvent("cac-store", { detail: key }));
  } catch {
    /* ignore quota */
  }
}

/** Persisted state hook synced across tabs/pages via localStorage + events. */
export function usePersisted<T>(key: string, fallback: T): [T, (v: T) => void] {
  const [state, setState] = useState<T>(fallback);
  useEffect(() => {
    setState(read(key, fallback));
    const onChange = () => setState(read(key, fallback));
    window.addEventListener("cac-store", onChange);
    window.addEventListener("storage", onChange);
    return () => {
      window.removeEventListener("cac-store", onChange);
      window.removeEventListener("storage", onChange);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  const set = (v: T) => { setState(v); write(key, v); };
  return [state, set];
}

export const useLLMConfig = () => usePersisted<LLMConfig>(KEYS.llm, {});
export const useLastResult = () => usePersisted<AnalysisResult | null>(KEYS.result, null);
export const useLastPatient = () => usePersisted<PatientContext | null>(KEYS.patient, null);
export const STORE_KEYS = KEYS;
