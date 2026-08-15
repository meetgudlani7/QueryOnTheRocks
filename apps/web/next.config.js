/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  // Standalone output traces only the dependencies actually used at
  // runtime into .next/standalone, so the Docker image doesn't need the
  // full node_modules tree copied in.
  output: 'standalone',
}

module.exports = nextConfig
