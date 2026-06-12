"use client";
import { useEffect, useRef, useState } from "react";
import { Upload, FileCheck2, X, Play, Loader2, Download } from "lucide-react";
import { SAMPLES, type SampleDataset } from "@/lib/samples";
import type { PatientContext } from "@/lib/types";
import { fetchDatasets, datasetDownloadUrl, type DatasetInfo } from "@/lib/api";
import { cn } from "@/lib/utils";

export interface RunInput {
  file: File | null;
  sample: SampleDataset | null;
  patient: PatientContext;
}

export function InputPanel({
  running, onRun,
}: {
  running: boolean;
  onRun: (input: RunInput) => void;
}) {
  const [sampleId, setSampleId] = useState<string>("ictal");
  const [file, setFile] = useState<File | null>(null);
  const [drag, setDrag] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const sample = SAMPLES.find((s) => s.id === sampleId) || SAMPLES[0];
  const [patient, setPatient] = useState<PatientContext>(sample.patient);
  const [downloads, setDownloads] = useState<DatasetInfo[]>([]);
  useEffect(() => { fetchDatasets().then(setDownloads).catch(() => {}); }, []);

  const pickSample = (id: string) => {
    setSampleId(id);
    setFile(null);
    const s = SAMPLES.find((x) => x.id === id)!;
    setPatient(s.patient);
  };

  const onFile = (f: File | null) => {
    setFile(f);
    if (f) setPatient((p) => ({ ...p, patient_id: "UPLOAD-" + f.name.split(".")[0].slice(0, 10) }));
  };

  return (
    <div className="space-y-5">
      {/* dataset source */}
      <div>
        <h3 className="mb-2 text-sm font-semibold">1 · Choose a dataset</h3>
        <div className="grid gap-2 sm:grid-cols-3">
          {SAMPLES.map((s) => (
            <button
              key={s.id}
              onClick={() => pickSample(s.id)}
              className={cn(
                "rounded-xl border p-3 text-left transition-all",
                !file && sampleId === s.id
                  ? "border-[var(--color-cyan)] bg-[var(--color-cyan)]/[0.06]"
                  : "border-[var(--color-border)] hover:border-[var(--color-cyan)]/40",
              )}
            >
              <span className="text-sm font-medium">{s.name}</span>
              <span className="mt-1 block text-xs text-[var(--color-faint)] line-clamp-2">{s.description}</span>
            </button>
          ))}
        </div>
      </div>

      {/* upload */}
      <div>
        <h3 className="mb-2 text-sm font-semibold">…or upload your own</h3>
        <div
          onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => { e.preventDefault(); setDrag(false); onFile(e.dataTransfer.files?.[0] || null); }}
          onClick={() => fileRef.current?.click()}
          className={cn(
            "flex cursor-pointer flex-col items-center justify-center rounded-xl border border-dashed p-6 text-center transition-colors",
            drag ? "border-[var(--color-cyan)] bg-[var(--color-cyan)]/[0.06]" : "border-[var(--color-border)] hover:border-[var(--color-cyan)]/50",
          )}
        >
          <input
            ref={fileRef}
            type="file"
            accept=".csv,.tsv,.txt,.npy,.json,.edf,.bdf"
            className="hidden"
            onChange={(e) => onFile(e.target.files?.[0] || null)}
          />
          {file ? (
            <div className="flex items-center gap-2 text-sm">
              <FileCheck2 className="h-5 w-5 text-emerald-400" />
              <span className="font-medium">{file.name}</span>
              <span className="text-[var(--color-faint)]">({(file.size / 1024).toFixed(0)} KB)</span>
              <button onClick={(e) => { e.stopPropagation(); onFile(null); }} className="ml-1 rounded p-0.5 hover:bg-white/10">
                <X className="h-4 w-4" />
              </button>
            </div>
          ) : (
            <>
              <Upload className="h-7 w-7 text-[var(--color-cyan)]" />
              <p className="mt-2 text-sm">Drag & drop an EEG file, or click to browse</p>
              <p className="mt-1 text-xs text-[var(--color-faint)]">EDF · CSV · TSV · NPY · JSON — channels × samples</p>
            </>
          )}
        </div>
      </div>

      {/* downloadable test datasets */}
      {downloads.length > 0 && (
        <div>
          <h3 className="mb-2 flex items-center gap-2 text-sm font-semibold">
            <Download className="h-3.5 w-3.5 text-[var(--color-cyan)]" /> Download test datasets
          </h3>
          <div className="flex flex-wrap gap-1.5">
            {downloads.map((d) => (
              <a key={d.file} href={datasetDownloadUrl(d.file)} download
                title={d.description}
                className="chip !py-1 hover:border-[var(--color-cyan)] hover:text-[var(--color-cyan)]">
                {d.file.replace("eeg_", "").replace(".csv", "").replace(/_/g, " ")}
              </a>
            ))}
          </div>
          <p className="mt-1.5 text-xs text-[var(--color-faint)]">
            Download a real 16-channel CSV, then drag it into the upload box above to test the full pipeline.
          </p>
        </div>
      )}

      {/* patient context */}
      <div>
        <h3 className="mb-2 text-sm font-semibold">2 · Patient context</h3>
        <div className="grid gap-3 sm:grid-cols-2">
          <Input label="Patient ID" value={patient.patient_id} onChange={(v) => setPatient({ ...patient, patient_id: v })} />
          <div className="grid grid-cols-2 gap-3">
            <Input label="Age" type="number" value={String(patient.age ?? "")} onChange={(v) => setPatient({ ...patient, age: v ? Number(v) : undefined })} />
            <Input label="Sex" value={patient.sex || ""} onChange={(v) => setPatient({ ...patient, sex: v })} />
          </div>
          <div className="sm:col-span-2">
            <Input label="Symptoms" value={patient.symptoms} onChange={(v) => setPatient({ ...patient, symptoms: v })} />
          </div>
          <div className="sm:col-span-2">
            <Input
              label="Medications (comma-separated)"
              value={patient.medications.join(", ")}
              onChange={(v) => setPatient({ ...patient, medications: v.split(",").map((x) => x.trim()).filter(Boolean) })}
            />
          </div>
        </div>
      </div>

      <button
        onClick={() => onRun({ file, sample: file ? null : sample, patient })}
        disabled={running}
        className="btn btn-primary w-full text-base !py-3 disabled:opacity-60"
      >
        {running ? <><Loader2 className="h-4 w-4 animate-spin" /> Running pipeline…</> : <><Play className="h-4 w-4" /> Run multi-agent analysis</>}
      </button>
    </div>
  );
}

function Input({ label, value, onChange, type = "text" }: { label: string; value: string; onChange: (v: string) => void; type?: string }) {
  return (
    <label className="block">
      <span className="mb-1.5 block text-xs font-medium text-[var(--color-muted)]">{label}</span>
      <input
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-soft)] px-3 py-2 text-sm outline-none focus:border-[var(--color-cyan)]"
      />
    </label>
  );
}
