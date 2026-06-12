"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { Logo } from "./logo";
import { cn } from "@/lib/utils";
import { isLive } from "@/lib/api";

const LINKS = [
  { href: "/studio", label: "Studio" },
  { href: "/chat", label: "AI Chat" },
  { href: "/graph", label: "Knowledge Graph" },
  { href: "/architecture", label: "Architecture" },
];

export function Navbar() {
  const pathname = usePathname();
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);
  const [live, setLive] = useState(false);

  useEffect(() => {
    setLive(isLive());
    const onScroll = () => setScrolled(window.scrollY > 12);
    onScroll();
    window.addEventListener("scroll", onScroll);
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <header
      className={cn(
        "fixed inset-x-0 top-0 z-50 transition-all duration-300",
        scrolled ? "backdrop-blur-xl bg-[var(--color-bg)]/80 border-b border-[var(--color-border)]" : "bg-transparent",
      )}
    >
      <nav className="container-page flex h-16 items-center justify-between">
        <Link href="/" className="flex items-center gap-2.5 font-bold tracking-tight">
          <Logo />
          <span className="text-[15px]">
            Clinical<span className="gradient-text">AI</span> Copilot
          </span>
        </Link>

        <div className="hidden items-center gap-1 md:flex">
          {LINKS.map((l) => (
            <Link
              key={l.href}
              href={l.href}
              className={cn(
                "rounded-lg px-3.5 py-2 text-sm font-medium transition-colors",
                pathname === l.href
                  ? "text-[var(--color-cyan)] bg-white/5"
                  : "text-[var(--color-muted)] hover:text-[var(--color-ink)] hover:bg-white/5",
              )}
            >
              {l.label}
            </Link>
          ))}
        </div>

        <div className="flex items-center gap-3">
          <span className="chip hidden sm:inline-flex">
            <span className={cn("h-1.5 w-1.5 rounded-full", live ? "bg-emerald-400" : "bg-amber-400", "animate-pulse-glow")} />
            {live ? "Backend live" : "Demo mode"}
          </span>
          <Link href="/studio" className="btn btn-primary !px-4 !py-2">
            Launch Studio
          </Link>
          <button
            className="md:hidden rounded-lg border border-[var(--color-border)] p-2"
            onClick={() => setOpen((o) => !o)}
            aria-label="Menu"
          >
            <div className="space-y-1">
              <span className="block h-0.5 w-5 bg-current" />
              <span className="block h-0.5 w-5 bg-current" />
              <span className="block h-0.5 w-5 bg-current" />
            </div>
          </button>
        </div>
      </nav>

      {open && (
        <div className="md:hidden border-t border-[var(--color-border)] bg-[var(--color-bg)]/95 backdrop-blur-xl">
          <div className="container-page flex flex-col py-3">
            {LINKS.map((l) => (
              <Link
                key={l.href}
                href={l.href}
                onClick={() => setOpen(false)}
                className="rounded-lg px-3 py-3 text-sm text-[var(--color-muted)] hover:text-[var(--color-ink)]"
              >
                {l.label}
              </Link>
            ))}
          </div>
        </div>
      )}
    </header>
  );
}
