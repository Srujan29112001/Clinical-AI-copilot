import Link from "next/link";
import { Logo } from "./logo";

export function Footer() {
  return (
    <footer className="border-t border-[var(--color-border)] mt-24">
      <div className="container-page py-12">
        <div className="flex flex-col gap-8 md:flex-row md:items-start md:justify-between">
          <div className="max-w-sm">
            <Link href="/" className="flex items-center gap-2.5 font-bold">
              <Logo size={24} />
              <span>Clinical AI Copilot</span>
            </Link>
            <p className="mt-3 text-sm text-[var(--color-muted)]">
              Multi-agent EEG analysis with clinical GraphRAG and hybrid local/API
              inference. Research & educational use only — not FDA-approved.
            </p>
          </div>
          <div className="grid grid-cols-2 gap-10 sm:grid-cols-3">
            <FooterCol title="Product" links={[["Studio", "/studio"], ["AI Chat", "/chat"], ["Knowledge Graph", "/graph"]]} />
            <FooterCol title="System" links={[["Architecture", "/architecture"], ["API Docs", "/architecture#api"]]} />
            <FooterCol
              title="Source"
              links={[["GitHub", "https://github.com/Srujan29112001/Clinical-AI-copilot"]]}
            />
          </div>
        </div>
        <div className="mt-10 flex flex-col gap-2 border-t border-[var(--color-border)] pt-6 text-xs text-[var(--color-faint)] sm:flex-row sm:items-center sm:justify-between">
          <p>© {new Date().getFullYear()} Clinical AI Copilot. MIT License.</p>
          <p>Decision-support only · Requires qualified clinician oversight.</p>
        </div>
      </div>
    </footer>
  );
}

function FooterCol({ title, links }: { title: string; links: [string, string][] }) {
  return (
    <div>
      <h4 className="text-xs font-semibold uppercase tracking-wider text-[var(--color-faint)]">{title}</h4>
      <ul className="mt-3 space-y-2">
        {links.map(([label, href]) => (
          <li key={label}>
            <Link href={href} className="text-sm text-[var(--color-muted)] hover:text-[var(--color-cyan)]">
              {label}
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}
