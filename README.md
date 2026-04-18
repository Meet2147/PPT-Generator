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
