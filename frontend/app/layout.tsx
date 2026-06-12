import type { Metadata } from "next";
import "./globals.css";
import { Navbar } from "@/components/site/navbar";
import { Footer } from "@/components/site/footer";

export const metadata: Metadata = {
  title: "Clinical AI Copilot — Multi-Agent EEG Analysis",
  description:
    "A multi-agent clinical decision-support system: upload EEG/clinical data, watch seven specialized AI agents triage, diagnose and report — running on your local GPU or any API provider.",
  keywords: ["EEG", "clinical AI", "multi-agent", "seizure detection", "GraphRAG", "neurology"],
  authors: [{ name: "Clinical AI Copilot" }],
  openGraph: {
    title: "Clinical AI Copilot",
    description: "Multi-agent EEG analysis with hybrid local/API inference.",
    type: "website",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>
        <Navbar />
        <main className="min-h-screen pt-16">{children}</main>
        <Footer />
      </body>
    </html>
  );
}
