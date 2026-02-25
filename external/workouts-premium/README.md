# Workouts Premium API (RapidAPI)

A production-ready workouts API for fitness apps, coaches, and AI assistants.

## Base URL
`https://workouts-premium.p.rapidapi.com`

## Authentication
Include these headers in every request:

- `X-RapidAPI-Key: <your_rapidapi_key>`
- `X-RapidAPI-Host: workouts-premium.p.rapidapi.com`

For direct origin usage (non-RapidAPI), send either:

- `x-api-key: <your_api_key>`
- `Authorization: Bearer <your_api_key>`

## Postman Collection
Import this file into Postman:

- `Workouts-Premium-RapidAPI.postman_collection.json`

## Endpoints

1. `GET /`
Purpose: API health and endpoint list.

2. `GET /available-muscles`
Purpose: List all supported target muscles.

3. `GET /available-body-parts`
Purpose: List all supported body parts.

4. `GET /exercises-by-muscles?muscle1={muscle1}&muscle2={muscle2}`
Purpose: Return exercises for two muscle groups.
Example:
`/exercises-by-muscles?muscle1=biceps&muscle2=triceps`

5. `GET /exercises-by-body-parts?part1={part1}&part2={part2}`
Purpose: Return exercises for two body parts.
Example:
`/exercises-by-body-parts?part1=chest&part2=back`

6. `GET /exercise/{exercise_id}`
Purpose: Return detailed exercise info, including instructions.
Example:
`/exercise/3XFdb1Z`

7. `POST /suggest-workouts?question={question}`
Purpose: AI-powered workout suggestions based on user intent.
Example:
`/suggest-workouts?question=I want bigger arms and shoulders`

8. `GET /media/gif/{gif_filename}`
Purpose: Fetch 360x360 exercise GIF from internal media route.
Example:
`/media/gif/05Cf2v8.gif`

## Example cURL

```bash
curl --request GET \
  --url 'https://workouts-premium.p.rapidapi.com/exercises-by-muscles?muscle1=biceps&muscle2=triceps' \
  --header 'X-RapidAPI-Key: YOUR_RAPIDAPI_KEY' \
  --header 'X-RapidAPI-Host: workouts-premium.p.rapidapi.com'
```

## Response Notes

- Exercise objects include:
  - `exerciseId`
  - `name`
  - `gifUrl` (internal route format: `/media/gif/{file}.gif`)
  - `equipment`
  - muscle/body-part metadata depending on endpoint
- GIF delivery defaults to `360x360` for better mobile performance.

## Errors

- `400 Bad Request`: invalid muscle/body part or invalid GIF filename.
- `401 Unauthorized`: missing or invalid API key.
- `429 Too Many Requests`: plan limit reached (usage service).
- `404 Not Found`: exercise or GIF not found.
- `502 Bad Gateway`: usage service validation/consume failed.
- `503 Service Unavailable`: usage service unavailable.
- `500 Internal Server Error`: AI suggestion provider failure.

## Why Developers Buy This API

- Real exercise metadata with clear categorization.
- Mobile-optimized GIF delivery via backend-controlled media route.
- Fast integration for workout planners, personal trainer apps, and fitness chatbots.
- AI workout suggestion endpoint included.

## Recommended RapidAPI Listing Copy

**Short description:**
Premium workout exercise API with structured exercise metadata, body-part and muscle filters, 360x360 GIF media links, and AI-powered workout suggestions.

**Use cases:**
- Personal trainer and gym companion apps
- Workout planning SaaS
- AI fitness assistants
- Health and wellness dashboards
