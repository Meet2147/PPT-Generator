# DeckMint

DeckMint is a subscription-first presentation generation product built as both a web app and a FastAPI backend.

## What changed

- Reframed the repo around one product instead of a mixed monolith.
- Added a web app with Gamma-style positioning, deck generation, and subscription pricing.
- Added a cleaner FastAPI API for catalog data and PPTX generation.
- Made the PowerPoint export editable and visually more professional.

## API

- `GET /api/v1/health`
- `GET /api/v1/catalog`
- `POST /api/v1/presentations/generate`
- `GET /api/v1/presentations/download/{filename}`
- `POST /api/v1/billing/checkout`
- `POST /api/v1/billing/webhooks/razorpay`
- `POST /api/v1/docs-token`

## Local run

```bash
python3 -m uvicorn app.main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000)

## Environment

Optional:

- `PPLX_API_KEY` for Perplexity-powered deck narratives
- `PUBLIC_BASE_URL` for production download links
- `RAZORPAY_KEY_ID`
- `RAZORPAY_KEY_SECRET`
- `RAZORPAY_WEBHOOK_SECRET`
- `DOCS_ADMIN_SECRET`
- `DOCS_JWT_SECRET`
- `RAZORPAY_PLAN_STUDENT_MONTHLY`
- `RAZORPAY_PLAN_STUDENT_ANNUAL`
- `RAZORPAY_PLAN_CORPORATE_MONTHLY`
- `RAZORPAY_PLAN_CORPORATE_ANNUAL`
- `RAZORPAY_PLAN_EXECUTIVE_MONTHLY`
- `RAZORPAY_PLAN_EXECUTIVE_ANNUAL`

Without `PPLX_API_KEY`, the app still works using a structured local fallback narrative.

## Razorpay setup

Use Razorpay Subscriptions for this product and create six plans:

- `Student Monthly`
- `Student Annual`
- `Corporate Monthly`
- `Corporate Annual`
- `Executive Monthly`
- `Executive Annual`

Then map those plan ids into the matching environment variables above.

The `Earlybird Lifetime` offer is intentionally handled as a one-time Razorpay Payment Link instead of a subscription.

## Render split deployment

This repo is set up for two separate Render services via [render.yaml](/Users/meetjethwa/Development/PPT_Latest/render.yaml:1):

- `deckmint-api` as the FastAPI backend at `api.dashovia.com`
- `deckmint-web` as the static frontend at `slides.dashovia.com`

The frontend is built with:

```bash
python3 scripts/build_web.py
```

That generates `web-dist/` and injects a `config.js` file with the API base URL for the deployed web app.

## Swagger protection

The public FastAPI docs are disabled by default. To get a 24-hour Swagger access token:

```bash
curl -X POST https://api.dashovia.com/api/v1/docs-token \
  -H "Content-Type: application/json" \
  -H "X-Docs-Admin-Secret: YOUR_DOCS_ADMIN_SECRET" \
  -d '{"requested_by":"admin"}'
```

Then open the returned `docs_url`.
