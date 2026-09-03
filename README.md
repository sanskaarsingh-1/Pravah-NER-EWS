# NER-EWS — Live Site

A single Flask app that serves the landing page, the dashboard, **and**
the API — one deployment, one URL.

## Merged in this build

This build combines everything from both prior drops into one app:

- **GIS System page** (`/gis` or `/gis.html`) — a dedicated MapLibre-based
  GIS view (district layer, hazard grid, field reports layer, search) that
  was previously a separate, unlinked build. It reuses the same backend
  endpoints as the main dashboard (`/api/tile`, `/api/predict/grid`,
  `/api/weather`, `/api/predict`, `/api/reports`) — no new backend code
  needed — and now lists all 33 NER districts (it previously only knew
  about 6). Linked from a new **"GIS System"** button in the dashboard
  header.
- **3D Terrain view restored** (`/dashboard-3d` or `/dashboard-3d.html`) —
  the Three.js terrain-draping page and its JS already existed in the
  file tree but had no Flask route pointing to it, so it was unreachable.
  The route is back, and it's linked from a new **"3D Terrain"** button in
  the dashboard header.
- Everything else below (admin panel, live hazard grid, per-district
  markers/search, media uploads on field reports, auto-refresh, the NASA
  proxy rate-limit fix) carries over unchanged from the more complete of
  the two prior builds.

## Recent fixes (this update)

- **The hazard grid was using random placeholder numbers.** `buildGrid()`
  in `dashboard.html` generated rainfall/soil-moisture/slope for every grid
  cell with `Math.random()` instead of real data, even though the README
  always described this as live IDW-interpolated weather. This made the
  heat surface visually plausible but disconnected from actual conditions
  — the likely reason the map looked like it only "worked" for a few
  districts. It now pulls live weather from all 33 district centroids
  (via `/api/weather`) and inverse-distance-weights it across the grid,
  plus derives slope from the real terrain DEM already loaded into the
  map, same methodology `dashboard-3d.js` already used correctly.
- **Every one of the 33 districts now has its own marker** on the map,
  colour-coded by its own live model prediction (not just the diffuse
  background heatmap), with a click-to-inspect popup.
- **Added a district/state search box** above the map — type to filter,
  arrow keys + Enter to jump, flies the camera and opens that district's
  live reading.
- **The dashboard now refreshes itself** every 3 minutes (weather, hazard
  grid, alerts, field reports, quakes) instead of only ever loading data
  once when the page first opens.
- **Map click popups** now fetch real live weather for the clicked point
  instead of a hardcoded 75mm/60% placeholder.
- **Fixed a dead-code bug** in the admin-only `/api/nasa-imagery` route: a
  429 rate-limit handler was written after a `return` inside an `except`
  block, so it could never execute; a real rate-limited response from
  NASA now gets caught and reported correctly.

## Why this version has no `.pkl` file

Earlier versions loaded a pre-trained model from `landslide_model.pkl`
via joblib. That broke across machines with confusing errors
(`InconsistentVersionWarning`, `MT19937 is not a known BitGenerator`)
whenever the numpy/scikit-learn versions on the machine that trained
the file didn't exactly match the machine loading it — which is a
common, annoying, and easy-to-hit problem with pickle files in
general.

**Fix:** `app.py` now trains the hazard model itself, in memory, the
moment it starts up (takes well under a second). It always uses
whatever numpy/scikit-learn you actually have installed, so there is
no version mismatch possible, ever. `train_landslide_model.py` is
kept in this folder for reference / experimentation (it compares more
model candidates and prints detailed metrics) but `app.py` no longer
depends on any file it produces.

## Project layout

```
app.py                     Flask app: serves pages + API, trains model at startup
templates/index.html       "Pravah" landing page
templates/dashboard.html   Live map dashboard: Leaflet + whole-NER hazard grid +
                            USGS quakes + NASA GIBS imagery + OpenTopoMap +
                            Open-Meteo rainfall/soil feeds + Chart.js gauges
static/                    (empty — add any extra images/JS/CSS here)
train_landslide_model.py   Optional: standalone script comparing more model candidates
requirements.txt
Procfile                   Tells Render/Railway how to start the app
```

Note: the dashboard pulls live third-party data client-side (USGS
earthquakes, OpenTopoMap terrain tiles, Open-Elevation for real slope
data, Open-Meteo rainfall/soil moisture). NASA GIBS/Earth Imagery,
the alert dispatcher, and the data-source-status panel now live in a
separate **admin panel** (see "Admin panel" below) rather than on the
public dashboard.

## Admin panel

The public dashboard is for citizens and field officials: the live
hazard map, field reports, rainfall/signal widgets, and the **Report a
hazard** tiles (photo/video upload of cracks, slope movement, or
blocked roads — see below).

Three tools that used to sit on the public dashboard are now behind a
simple password-gated admin panel at `/admin`, since they're
operational tools, not something the public needs:

- **NASA Earth observation** — satellite imagery lookup, proxied
  through the backend at `GET /api/nasa-imagery?lat=&lon=&date=`
  (admin-only) because the free `DEMO_KEY` is capped at 30
  requests/hour and 50/day *shared across every app on the internet
  using it*. Set the `NASA_API_KEY` environment variable to your own
  free key from api.nasa.gov (instant, no approval wait) to get 1,000
  requests/hour instead; without it, the app falls back to `DEMO_KEY`
  automatically. The proxy also tells the frontend *why* a request
  failed — rate limit vs. no imagery available for that lat/lon/date —
  instead of one generic "unavailable" message.
- **Alert dispatcher** — broadcasts warnings (app feed + optional SMS)
- **Data source status** — which live feeds are actually connected

The admin panel also shows the full field-reports table (including
any attached photos/videos) and the alert dispatch history, so an
operator can review citizen-submitted reports before/while dispatching
an alert.

Log in at `/admin/login`. Set the `ADMIN_PASSWORD` environment
variable to your own password before deploying — without it, the app
falls back to a default (`admin123`) and logs a warning on startup.
Also set `SECRET_KEY` to a random value (used to sign the login
session cookie); without it, a fixed development key is used and
logged as a warning. This is a single shared admin password, not
per-user accounts — enough to keep the public dashboard from doubling
as an unauthenticated ops console, but not a substitute for real
multi-operator auth in a production rollout.

## Report a hazard (citizen/field photo & video upload)

The public dashboard has a **Report a hazard** section with three
tap-to-upload tiles — **Cracks**, **Slope movement**, and **Blocked
roads** — so a citizen or field official can go straight from seeing a
problem to submitting a geo-tagged photo or video of it, without
filling in a hazard-type dropdown first. Tapping a tile opens the
report form with that hazard type pre-selected and immediately
requests the device's current location (editable if it's off).

The form accepts an optional photo or video (`<input type="file"
accept="image/*,video/*" capture="environment">`, so on a phone it can
open the camera directly). On submit, the report — including the file
— is uploaded to `POST /api/reports` as `multipart/form-data` and
saved under `static/uploads/reports/`; allowed types are
jpg/jpeg/png/webp/gif for photos and mp4/mov/webm/m4v for video, up to
30MB. The admin panel's field-reports table shows a thumbnail (photos)
or a "View video" link for every report with an attachment.

The existing offline-queue behavior (see "Offline field reporting"
below) still covers the text fields (district, hazard type,
coordinates, severity, remarks) when the device has no connectivity —
but a photo/video attachment needs a live connection to upload, since
browser `localStorage` isn't a practical place to queue large binary
files. The form tells the citizen this plainly if they're offline.

## 3D terrain now uses real satellite/topo imagery

`dashboard-3d.html` no longer shades the terrain with a purely
procedural hillshade. It now drapes real map tiles — satellite
photography (Esri World Imagery) or a real topographic map
(OpenTopoMap), switchable from the panel — over the same real
elevation mesh, the way MapTiler-style 3D terrain viewers do. The
tiles are fetched through a new backend route,
`GET /api/tile/<satellite|topo>/<z>/<x>/<y>`, which proxies the
upstream tile servers. That's a proxy rather than a direct
browser→ArcGIS/OpenTopoMap fetch specifically so the imagery is
same-origin: cross-origin tiles without CORS headers get "tainted"
and WebGL refuses to use them as a texture. Every one of the ~21×21
grid points is now marked on the terrain by default (colour-coded by
its own hazard level), not just the ones above a threshold — the
marker-threshold slider still lets you filter down from there.

## Whole-region hazard coverage (no fixed sensors)

Earlier versions of the dashboard showed hazard at a dozen fixed,
fabricated "IoT sensor" points (each with made-up InSAR/tilt/pore-
pressure readings). That's gone. The dashboard now scores the real
trained model across a grid covering the **entire North Eastern
Region** (~80 cells on a 1° lat/lon grid) and renders it as a
continuous heat surface:

- **Slope** — real elevation data from the free
  [Open-Elevation](https://www.open-elevation.com/) API; slope at
  each grid cell is derived from the elevation difference to two
  nearby probe points (finite-difference gradient), not invented.
- **Rainfall (72h) & soil moisture** — pulled live from Open-Meteo at
  each of the 6 district centroids, then inverse-distance-weighted
  across the grid, same live source the rest of the dashboard already
  used.
- **Hazard probability** — all ~80 grid cells are scored in a single
  call to the new `/api/predict/grid` batch endpoint, which runs the
  same trained pipeline as `/api/predict` (see "Model methodology"
  below) vectorized over the whole grid.
- **Topographic base map** — a new "Topographic Base Map" layer
  toggle switches the base tiles to [OpenTopoMap](https://opentopomap.org/)
  (real contour lines / terrain shading, free, no key), so the
  hazard surface can be read against actual terrain instead of a
  flat street map.

If the elevation service is unreachable, the dashboard falls back to
a clearly-labeled, cruder distance-based slope estimate rather than
silently pretending it has real data — check the grid-point popup or
the toast notification to see which mode is active.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000/` for the landing page, or
`http://localhost:5000/dashboard.html` for the dashboard.

You'll see this in the terminal on startup:
```
Training landslide hazard model...
Model trained. ROC-AUC=0.85..., alert threshold=0.12...
```

## Optional environment variables

None of these are required to run the app — every feature they unlock
degrades gracefully (and says so via `GET /api/health`) when unset.

| Variable | Enables |
|---|---|
| `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER` | Real SMS delivery on alert dispatch, via Twilio |
| `MSG91_AUTH_KEY`, `MSG91_SENDER_ID` | Real SMS delivery on alert dispatch, via MSG91 |
| `DATA_GOV_IN_API_KEY` | `/api/weather` prefers official IMD data (via data.gov.in) over the Open-Meteo fallback |
| `DATA_GOV_IN_IMD_RESOURCE_ID` | Overrides which data.gov.in IMD resource `/api/weather` queries |
| `NASA_API_KEY` | Higher-rate-limit NASA Earth Imagery access (admin panel). Falls back to the shared `DEMO_KEY` if unset |
| `ADMIN_PASSWORD` | Password required to sign in at `/admin/login`. **Set this before deploying** — defaults to `admin123` (with a startup warning) if unset |
| `SECRET_KEY` | Signs the admin login session cookie. **Set this before deploying** — defaults to a fixed insecure value (with a startup warning) if unset |
| `PORT` | Port Flask/gunicorn binds to (Render/Railway set this automatically) |
| `FLASK_DEBUG` | Set to `1` for Flask's debug reloader when running `python app.py` locally |

## Deploy for free — Render.com (~5 minutes)

1. Push this folder to a GitHub repo.
2. On [render.com](https://render.com) → **New +** → **Web Service** →
   connect that repo.
3. Render auto-detects Python. Build command:
   `pip install -r requirements.txt`. Start command is already set via
   the included `Procfile`.
4. Click **Create Web Service**. You'll get a live URL like
   `https://ner-ews.onrender.com` within a minute or two.

**Free-tier note:** Render's free web services sleep after ~15 minutes
of inactivity and take ~30-60s to wake on the next request.

## Model methodology

Ground-truth labeling rule is grounded in Caine (1980) rainfall
intensity-duration thresholds and standard infinite-slope stability
theory (soil saturation lowers effective shear strength; steeper
slopes fail more readily) — there's no public dataset pairing
[rainfall, soil moisture, slope] with confirmed NER landslide outcomes
at this resolution, so training data is synthetic but the labeling
rule itself is grounded in published geotechnical research. Full
methodology is documented in the `_train_model()` docstring in
`app.py` and in `train_landslide_model.py`.

Model: gradient-boosted trees on the 3 raw features (rainfall, soil
moisture, slope). A separate F2-optimal alert threshold (weights
recall 4x over precision) is computed alongside the standard 0.5
decision threshold, since for an early-warning system missing a real
event is worse than a false alarm.

## Data persistence note

SQLite (`ner_ews.db`) is created on first run. On Render's free tier
the filesystem is **ephemeral** — the DB resets on redeploy or long
idle periods. Fine for a demo; for anything longer-lived, add a
persistent disk or move to Postgres.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Liveness check — also reports whether the SMS gateway and IMD data-source are actually configured, or running in fallback mode |
| GET | `/api/weather?lat=&lon=` | Rainfall/soil-moisture for one point. Uses IMD data (via data.gov.in) if `DATA_GOV_IN_API_KEY` is set, otherwise transparently falls back to Open-Meteo and says so in the response's `source`/`note` fields |
| POST | `/api/predict` | Run the hazard model for one point. Body: `{rain_72h_mm, soil_moisture_pct, slope_deg, district}` |
| POST | `/api/predict/grid` | Batch-score many points at once (powers the whole-NER hazard surface). Body: `{points: [{lat, lon, rain_72h_mm, soil_moisture_pct, slope_deg}, ...]}` |
| GET | `/api/reports` | List field reports (most recent 200) |
| POST | `/api/reports` | Create one field report. Accepts JSON (text-only, e.g. offline queue replay) or `multipart/form-data` with fields `{client_id, lat, lon, district, hazard_type, severity, remarks}` plus an optional `media` file (photo/video) — `client_id` dedupes retries |
| POST | `/api/reports/sync` | Bulk sync an offline queue. Body: `{reports: [...]}`. Dedupes by `client_id`, so replaying an already-synced queue is a safe no-op |
| GET | `/api/alerts` | List logged alert dispatches (optionally `?district=`) — this is the app-based early-warning feed, also shown on the public landing page |
| POST | `/api/alerts/dispatch` | **Admin-only.** Log + broadcast an alert. Body: `{district, level, language, message, recipients: ["+91..."]}`. Always logs to the app feed; also attempts a real SMS send if a gateway is configured (see below) |
| POST | `/api/sensors` | Ingest one ground-sensor reading (rain gauge, piezometer, extensometer, etc). Body: `{sensor_id, sensor_type, value, unit, district, lat, lon, recorded_at}` |
| GET | `/api/sensors/latest` | Latest reading per sensor, flagged `is_live` if received in the last 3h. Reports `sensors_reporting: 0` honestly when nothing is connected |
| GET | `/api/tile/<layer>/<z>/<x>/<y>` | Proxies a real map tile (`layer` = `satellite` or `topo`) so the 3D dashboard can drape it on the terrain mesh as a same-origin texture |
| GET | `/api/nasa-imagery?lat=&lon=&date=` | **Admin-only.** Proxies NASA's Earth Imagery API |
| GET/POST | `/admin/login` | Admin login page / password check |
| GET | `/admin/logout` | Clears the admin session |
| GET | `/admin` | **Admin-only.** NASA imagery viewer, alert dispatcher, data-source status, and the full field-reports table (with any attached media) |
| GET | `/gis`, `/gis.html` | Dedicated GIS System page (district layer, hazard grid, field-reports layer, search) |
| GET | `/dashboard-3d`, `/dashboard-3d.html` | Three.js 3D terrain-draping view of the hazard grid |

## SMS / app-based early warning

`/api/alerts/dispatch` **always** logs the alert to the `alerts` table,
which powers the in-app alert feed on the dashboard and a ticker on the
public landing page — that part is real and works out of the box with
no configuration.

Actual SMS delivery is pluggable and off by default (there is no free
universal SMS API to wire up for a demo):

- Set `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER` to
  send via Twilio, **or**
- Set `MSG91_AUTH_KEY` (and optionally `MSG91_SENDER_ID`) to send via
  MSG91 (common for Indian SMS routes / DDMA-style deployments).

Without either, `/api/alerts/dispatch` still succeeds and the response's
`sms.status` is `"simulated"` — it's logged and visible in the app feed,
just not texted to a phone. This is reported honestly via
`GET /api/health` → `sms_gateway`, and in the dispatcher panel's
"DATA SOURCE STATUS" box in the dashboard.

## Weather: IMD vs Open-Meteo

IMD does not publish a simple free public REST API. The closest official
route is IMD data mirrored on **data.gov.in**, which needs a free
registered API key. `/api/weather` uses it automatically when
`DATA_GOV_IN_API_KEY` is set; otherwise it calls Open-Meteo (same
source the rest of the dashboard already used) and marks the response
`imd_available: false` rather than silently claiming official IMD data.
Satellite feeds (NASA GIBS Earth imagery, USGS earthquakes) and terrain
imagery (Esri World Imagery, OpenTopoMap) are unaffected by this and
already integrated, as described above.

## Ground sensor integration

`POST /api/sensors` is a real ingestion endpoint any rain gauge,
piezometer, extensometer, or other telemetry gateway can push readings
to — no physical sensors are wired up in this deployment, and
`GET /api/sensors/latest` reports `sensors_reporting: 0` and says so
plainly when nothing has posted recently, rather than showing invented
fixed "IoT sensor" points. When real hardware does post here, the
dashboard's "DATA SOURCE STATUS" panel picks it up automatically.

## Offline field reporting

The dashboard's Field Report form works with or without connectivity:

- **Online:** submits straight to `POST /api/reports`.
- **Offline:** the report is queued in the browser (`localStorage`) and
  the panel shows an OFFLINE badge with a queued-report count.
- Queued reports sync automatically the moment the browser's `online`
  event fires, or via the "Sync queued reports" button, by posting the
  whole queue to `POST /api/reports/sync`. Each report carries a
  client-generated `client_id`, so a retried sync (e.g. connection drops
  mid-upload) never creates duplicates.

This covers the client side of "offline sync support for remote
regions" from a single-page web app; a native/PWA wrapper would use the
same two endpoints with a longer-lived local queue (e.g. IndexedDB).

## What this backend does NOT do

- **No SMS delivery unless you configure a gateway.** See "SMS /
  app-based early warning" above — the app-based feed always works,
  SMS is opt-in via Twilio or MSG91 credentials.
- **No official IMD access unless you configure a data.gov.in key.**
  See "Weather: IMD vs Open-Meteo" above.
- **SQLite, not a production database.** Fine for a demo; a real
  multi-district rollout would want Postgres (+ PostGIS).
- **Single shared admin password, not per-user accounts.** The alert
  dispatcher and NASA imagery lookup require an admin login (see
  "Admin panel" above), but everyone with that one password shares the
  same access — there's no per-operator audit trail. The citizen
  report-submission and sensor-ingest endpoints (`/api/reports`,
  `/api/sensors`) are still open to anyone who can reach the API, by
  design (citizens shouldn't need an account to report a hazard), so
  treat their contents as unverified until reviewed in the admin
  panel. Add per-operator auth and/or rate limiting before any real
  deployment.
- **Synthetic training data**, as explained above — not a substitute
  for a model trained on verified incident records.
- **Coarse grid, not a fine-grained slope map.** The whole-NER hazard
  surface samples on a ~1° lat/lon grid (~110km spacing) for speed and
  to stay within free API limits — real deployments would want a much
  finer DEM-derived slope raster, not a live point grid.
- **No physical sensors connected.** `/api/sensors` is a real,
  functioning ingestion endpoint, but nothing is currently posting to
  it — see "Ground sensor integration" above.


## Combined feature set

This build combines the live seismic-aware early-warning stack with automated
field-report media verification.

- **Seismic-aware landslide risk:** live USGS earthquake data contributes a
  normalized seismic-influence feature to point and grid predictions.
- **Live weather + terrain GIS:** rainfall/soil-moisture feeds, terrain slope,
  field reports, road disruption signals, and live seismic context are available
  in the GIS dashboard.
- **Automated field-report verification:** uploaded images are checked for EXIF
  GPS/location mismatch, stale capture time, duplicate/near-duplicate media,
  basic image integrity, and consistency with live rainfall/soil conditions.
- **Human review workflow:** verification results are shown in the admin queue,
  with filters and approve/reject actions. Reports are not silently discarded.
- **3D terrain dashboard:** the existing 3D view remains part of the combined
  build alongside the seismic/live-GIS features.

Verification is a triage aid, not proof that a report is genuine.


## Strict visual field-report gate (v1.3)

Uploaded field photos are now processed in this order:

1. The browser shows the photo preview.
2. A multimodal vision model inspects the actual image and identifies the visible hazard.
3. The visual result is compared with the reporter's selected hazard.
4. The backend rechecks photo GPS/time/duplicate signals.
5. The backend pulls live rainfall + soil moisture and the current USGS earthquake feed for the report location.
6. Only when the required gates pass is the row inserted into `Field Reports`. Failed images are deleted and are **not** inserted into the field-report table.

### Vision API configuration

Set `GEMINI_API_KEY` in the deployment environment. Optional: set `GEMINI_MODEL` (default `gemini-2.5-flash`). The system fails closed if the vision key/service is unavailable, so an uninspected image cannot become a trusted field report.

This is an AI verification/triage gate, not proof that a report is genuine. A production deployment should retain operator oversight for high-impact decisions.


## Final combined build

This ZIP merges the two supplied versions into one application. It retains the seismic-aware prediction stack, historical landslide inventory/location intelligence, GIS/3D dashboards, road status, sensors, alerts, offline report sync, and the newer automated field-photo verification + admin review workflow.

The report pipeline uses image verification metadata (GPS/time/duplicate/live-condition checks) and optional Gemini visual analysis. The public report list hides unapproved media reports until an admin verifies them.


### Photo verification fix

The photo verifier now defaults to `gemini-2.5-flash`, avoids unsupported sampling settings when using Gemini 3.x, and treats temporary live-feed outages as warnings instead of automatically rejecting an otherwise valid photo.

Set a fresh Gemini API key in `1.env` or your hosting provider environment variables, restart the app, then open `/api/verification-status` to confirm the server can reach Gemini. Never commit or share the API key.
