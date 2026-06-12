/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: "standalone",
  // Allow the static export / Vercel build to succeed even if lint/types lag.
  eslint: { ignoreDuringBuilds: true },
  async rewrites() {
    // Proxy /api to the backend when NEXT_PUBLIC_API_URL is set at build time.
    const api = process.env.NEXT_PUBLIC_API_URL;
    if (!api) return [];
    return [{ source: "/backend/:path*", destination: `${api}/:path*` }];
  },
};

export default nextConfig;
