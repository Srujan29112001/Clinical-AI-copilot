export function Logo({ size = 28 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 40 40" fill="none" aria-hidden>
      <defs>
        <linearGradient id="lg" x1="0" y1="0" x2="40" y2="40">
          <stop offset="0" stopColor="#ff3b4e" />
          <stop offset="1" stopColor="#ff5d8f" />
        </linearGradient>
      </defs>
      <rect x="2" y="2" width="36" height="36" rx="10" stroke="url(#lg)" strokeWidth="2" />
      {/* EEG waveform */}
      <path
        d="M7 22 L12 22 L14 13 L17 28 L20 9 L23 26 L26 18 L28 22 L33 22"
        stroke="url(#lg)"
        strokeWidth="2.2"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
      />
    </svg>
  );
}
