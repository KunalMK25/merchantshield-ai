# MerchantShield AI — frontend

This is the React/Vite dashboard for MerchantShield AI. It consumes the FastAPI
backend's documented REST contract only — see the root [`README.md`](../README.md)
for the full project overview, and [`../docs/frontend.md`](../docs/frontend.md) for
this app's architecture and design decisions.

## Quick commands

```bash
npm install       # install dependencies
npm run dev       # local dev server (requires the backend running separately, see below)
npm run build     # production build, output to dist/ (served by the backend at "/")
```

By default the dev server expects the backend at the same origin. To point it at a
backend running on a different port, see `.env.example` in the project root
(`VITE_API_BASE_URL`).
