import threading
"""
NER Landslide Early Warning System — Backend API
"""

import hashlib
import io
import csv
import base64
import re
import json
import logging
import math
import os
import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from functools import wraps

# Load environment variables from files beside app.py, regardless of the
# process working directory. Hosting platforms should still use environment
# variables rather than committing secrets.
from dotenv import load_dotenv
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))
load_dotenv(os.path.join(BASE_DIR, "1.env"), override=False)


import numpy as np
import requests
from flask import (
    Flask, Response, g, jsonify, redirect, render_template, request, session,
    url_for,
)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from werkzeug.utils import secure_filename
from PIL import ExifTags, Image

try:
    import imagehash
except ImportError:  # pragma: no cover
    imagehash = None

DB_PATH = os.path.join(os.path.dirname(__file__), "ner_ews.db")

app = Flask(__name__, template_folder="templates", static_folder="static")
logger = logging.getLogger("ner_ews")
logging.basicConfig(level=logging.INFO)

APP_VERSION = "1.4.0"

# Optional multimodal vision verifier. The report pipeline is intentionally
# strict: when a vision API key is configured, an uploaded image must be
# visually classified and agree with the live telemetry before it can enter
# the public field-report table.
GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()


# ---------------------------------------------------------------------------
# Gemini helper: gives actionable errors instead of silently returning a
# generic failure. It works with the Gemini REST generateContent endpoint.
# ---------------------------------------------------------------------------
def _gemini_request(model, contents, generation_config=None):
    if not GEMINI_API_KEY or GEMINI_API_KEY.startswith("PASTE_YOUR_"):
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Set a valid key in your deployment "
            "environment (or in 1.env for local development)."
        )

    import requests

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )
    payload = {"contents": contents}
    if generation_config:
        payload["generationConfig"] = generation_config

    try:
        response = requests.post(
            url,
            params={"key": GEMINI_API_KEY},
            json=payload,
            timeout=60,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not reach Gemini API: {exc}") from exc

    if not response.ok:
        try:
            details = response.json()
        except ValueError:
            details = response.text
        raise RuntimeError(
            f"Gemini API returned HTTP {response.status_code}: {details}"
        )

    data = response.json()
    candidates = data.get("candidates") or []
    if not candidates:
        feedback = data.get("promptFeedback") or data.get("error")
        raise RuntimeError(f"Gemini returned no candidates: {feedback or data}")

    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(
        p.get("text", "") for p in parts if isinstance(p, dict)
    ).strip()

    if not text:
        raise RuntimeError(f"Gemini returned an empty response: {data}")

    return text

GEMINI_MODEL = "gemini-3.6-flash"
STARTED_AT = datetime.now(timezone.utc)

# ---------------------------------------------------------------------------
# Admin access
# ---------------------------------------------------------------------------
# NASA imagery lookup, alert dispatch, and the "data source status" panel are
# gated behind a single shared admin password rather than being open to
# anyone who loads the public dashboard. This is intentionally simple
# (one shared password, server-side session) — enough to keep the public
# citizen-facing dashboard from doubling as an unauthenticated ops console.
# For a real multi-operator rollout, replace this with per-user accounts.
app.secret_key = os.environ.get("SECRET_KEY", "").strip() or "dev-only-insecure-secret-change-me"
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "").strip() or "admin123"
if app.secret_key == "dev-only-insecure-secret-change-me" or ADMIN_PASSWORD == "admin123":
    logger.warning(
        "Using a default SECRET_KEY and/or ADMIN_PASSWORD. Set the SECRET_KEY "
        "and ADMIN_PASSWORD environment variables before deploying this for real."
    )


def admin_required(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if not session.get("is_admin"):
            if request.path.startswith("/api/"):
                return jsonify({"error": "admin login required"}), 401
            return redirect(url_for("admin_login", next=request.path))
        return view_func(*args, **kwargs)
    return wrapped


# ---------------------------------------------------------------------------
# Citizen/field media uploads (photos & short videos attached to reports)
# ---------------------------------------------------------------------------
UPLOAD_DIR = os.path.join(app.static_folder, "uploads", "reports")
os.makedirs(UPLOAD_DIR, exist_ok=True)
ALLOWED_MEDIA_EXT = {
    "jpg": "image", "jpeg": "image", "png": "image", "webp": "image", "gif": "image",
    "mp4": "video", "mov": "video", "webm": "video", "m4v": "video",
}
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024  # 30 MB per upload


def save_report_media(file_storage):
    """Validate and save an uploaded photo/video. Returns (filename, media_type)
    or (None, None) if no valid file was provided."""
    if not file_storage or not file_storage.filename:
        return None, None
    ext = file_storage.filename.rsplit(".", 1)[-1].lower() if "." in file_storage.filename else ""
    media_type = ALLOWED_MEDIA_EXT.get(ext)
    if not media_type:
        raise ValueError(f"unsupported file type: .{ext}")
    filename = f"{uuid.uuid4().hex}.{ext}"
    safe_name = secure_filename(filename)
    file_storage.save(os.path.join(UPLOAD_DIR, safe_name))
    return f"uploads/reports/{safe_name}", media_type


# ---------------------------------------------------------------------------
# Media verification — catches fake/spammed field-report photos before an
# operator ever sees them.
#
# For every photo attached to a report, this pipeline checks:
#   1. Location   — does the photo's own EXIF GPS tag (if the camera/phone
#                    recorded one) agree with the lat/lon the reporter typed
#                    or the browser captured?
#   2. Recency    — does the photo's EXIF capture time look like it was
#                    taken around the time of the report, not pulled from
#                    an old camera roll?
#   3. Duplicates — does this exact/near-identical image already exist
#                    against a *different* report (perceptual hash), which
#                    is the classic signature of someone spamming the same
#                    photo across many fake reports?
#   4. Conditions — do the reporter's own claims (a rain-driven hazard type)
#                    square with the live rainfall/soil-moisture feed this
#                    app already pulls for that exact point?
#   5. Integrity  — is it even a valid, non-corrupt, non-trivial image?
#
# None of this can prove a photo is genuine — it can only catch the cheap,
# common ways fake reports get made (wrong location, recycled photo, copy-
# pasted spam image, or a claim that contradicts the weather on record).
# Reports are never silently dropped: everything still lands in the table,
# just tagged so an operator can triage flagged/needs_review reports first
# instead of wading through the full unsorted list.
# ---------------------------------------------------------------------------

# Anything closer than this is treated as a GPS match (phone GPS + rounding
# error easily accounts for a km or two).
LOCATION_MATCH_KM = 3.0
# Beyond this, the photo's own location tag flatly disagrees with the
# claimed report location — a strong signal, not just noise.
LOCATION_MISMATCH_KM = 15.0
# A photo whose EXIF capture time is older than this relative to the report
# is treated as a recycled/stale photo rather than fresh field evidence.
STALE_MEDIA_DAYS = 14
# Perceptual-hash Hamming distance at/under which two images are considered
# the same photo (small values tolerate recompression/resizing, not a
# different scene).
DUPLICATE_HASH_DISTANCE = 6
# Below this, a claimed rain-driven hazard has essentially no rainfall or
# soil-moisture signal behind it at that exact point/time.
DRY_CONDITIONS_RAIN_MM = 5.0
DRY_CONDITIONS_SOIL_PCT = 20.0
RAIN_SENSITIVE_HAZARD_TYPES = {"slope_movement", "cracks", "landslide", "mudslide"}


def _exif_gps_to_degrees(value, ref):
    """Convert a PIL EXIF GPS ((deg,min,sec) as IFDRational tuples) value +
    hemisphere ref into signed decimal degrees. Returns None on bad data."""
    try:
        deg, minutes, seconds = (float(v) for v in value)
    except (TypeError, ValueError):
        return None
    decimal = deg + minutes / 60.0 + seconds / 3600.0
    if ref in ("S", "W"):
        decimal = -decimal
    return decimal


def extract_image_metadata(filepath):
    """Best-effort extraction of GPS + capture time + basic image health
    from a JPEG/PNG/etc. Returns a dict; any field that can't be determined
    is left as None rather than raising. Never trusts video files — PIL
    can't read those, so callers should skip this for media_type=='video'."""
    result = {
        "valid_image": False,
        "width": None,
        "height": None,
        "gps_lat": None,
        "gps_lon": None,
        "captured_at": None,
        "phash": None,
    }
    try:
        with Image.open(filepath) as img:
            img.verify()  # cheap corruption check; re-open below to actually read data
        with Image.open(filepath) as img:
            result["valid_image"] = True
            result["width"], result["height"] = img.size

            if imagehash is not None:
                try:
                    result["phash"] = str(imagehash.phash(img))
                except Exception:  # pragma: no cover - defensive, hashing should rarely fail
                    result["phash"] = None

            exif = img.getexif()
            if exif:
                exif_sub = exif.get_ifd(ExifTags.IFD.Exif) if hasattr(exif, "get_ifd") else {}
                tagged = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
                tagged.update({ExifTags.TAGS.get(k, k): v for k, v in exif_sub.items()})
                dt_raw = tagged.get("DateTimeOriginal") or tagged.get("DateTime")
                if dt_raw:
                    try:
                        result["captured_at"] = datetime.strptime(
                            str(dt_raw), "%Y:%m:%d %H:%M:%S"
                        ).replace(tzinfo=timezone.utc).isoformat()
                    except ValueError:
                        pass

                gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo) if hasattr(exif, "get_ifd") else None
                if gps_ifd:
                    gps = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_ifd.items()}
                    lat = _exif_gps_to_degrees(gps.get("GPSLatitude"), gps.get("GPSLatitudeRef"))
                    lon = _exif_gps_to_degrees(gps.get("GPSLongitude"), gps.get("GPSLongitudeRef"))
                    if lat is not None and lon is not None:
                        result["gps_lat"], result["gps_lon"] = lat, lon
    except Exception as exc:  # noqa: BLE001 - any failure just means "couldn't verify"
        logger.info("Could not read image metadata from %s: %s", filepath, exc)
    return result


def find_duplicate_media(db, phash, exclude_report_id=None):
    """Look for an existing report (other than this one) whose photo hashes
    to something visually near-identical. Only compares against the last
    500 reports with a stored hash, which is plenty for a demo-scale table
    and avoids scanning the whole table on every upload."""
    if imagehash is None or not phash:
        return None
    try:
        this_hash = imagehash.hex_to_hash(phash)
    except ValueError:
        return None
    rows = db.execute(
        "SELECT id, media_phash, created_at FROM reports "
        "WHERE media_phash IS NOT NULL AND id != ? "
        "ORDER BY created_at DESC LIMIT 500",
        (exclude_report_id or "",),
    ).fetchall()
    for row in rows:
        try:
            other_hash = imagehash.hex_to_hash(row["media_phash"])
        except ValueError:
            continue
        if (this_hash - other_hash) <= DUPLICATE_HASH_DISTANCE:
            return {"report_id": row["id"], "created_at": row["created_at"]}
    return None


def _normalise_visual_label(label):
    value = re.sub(r"[^a-z_ ]", "", str(label or "").lower()).strip().replace(" ", "_")
    aliases = {
        "landslide": "landslide", "mudslide": "mudslide", "slope_failure": "slope_movement",
        "slope_movement": "slope_movement", "cracks": "cracks", "ground_cracks": "cracks",
        "rockfall": "rockfall", "falling_rocks": "rockfall", "flood": "flood",
        "flooding": "flood", "blocked_road": "road_blockage", "road_blockage": "road_blockage",
        "debris_on_road": "road_blockage", "no_hazard": "no_hazard", "unclear": "unclear",
    }
    return aliases.get(value, value)


def _gemini_models_to_try():
    """Try the configured model first, then stable multimodal fallbacks."""
    models = [GEMINI_MODEL, "gemini-3.6-flash", "gemini-3.6-flash"]
    out = []
    for model in models:
        model = str(model or "").strip()
        if model and model not in out:
            out.append(model)
    return out


def visual_identify_image(filepath, claimed_hazard):
    """Inspect the actual image with Gemini multimodal vision.

    Uses REST directly and reports the real HTTP/API error. Tries several
    currently supported multimodal models so a model-specific outage does not
    make the whole visual gate fail unnecessarily.
    """
    result = {
        "available": bool(GEMINI_API_KEY),
        "label": None,
        "confidence": None,
        "summary": None,
        "matched_claim": False,
        "model": None,
    }

    if not GEMINI_API_KEY or GEMINI_API_KEY.startswith("PASTE_YOUR_"):
        result["available"] = False
        result["error"] = (
            "Gemini API key is missing. Set GEMINI_API_KEY in 1.env locally "
            "or in your hosting provider's Environment Variables, then restart."
        )
        return result

    try:
        with open(filepath, "rb") as fh:
            image_b64 = base64.b64encode(fh.read()).decode("ascii")
        with Image.open(filepath) as img:
            mime = Image.MIME.get(img.format, "image/jpeg")
            if mime not in ("image/jpeg", "image/png", "image/webp", "image/gif"):
                mime = "image/jpeg"
    except Exception as exc:
        result["error"] = f"image read failed: {exc}"
        return result

    prompt = f"""You are the visual verification stage of a landslide early-warning field-report system.
Inspect the uploaded field photograph itself. Do not assume the reporter's claim is true.
Classify the visible scene into exactly one label from:
landslide, mudslide, slope_movement, cracks, rockfall, flood, road_blockage, no_hazard, unclear.
The reporter claims: {claimed_hazard!r}.
Return ONLY valid JSON with keys: label, confidence, summary, matched_claim.
confidence must be a number from 0 to 1.
matched_claim is true only when the visible scene supports the claimed hazard.
If the image is unrelated, staged-looking, too ambiguous, or does not show a hazard, use no_hazard or unclear.
Keep summary under 180 characters."""

    last_error = None
    for model in _gemini_models_to_try():
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": mime, "data": image_b64}},
                ]
            }],
            "generationConfig": {
                "responseMimeType": "application/json",
                **({} if str(model).startswith("gemini-3") else {"temperature": 0}),
            },
        }

        try:
            resp = requests.post(
                url,
                params={"key": GEMINI_API_KEY},
                json=payload,
                timeout=45,
            )

            if not resp.ok:
                try:
                    details = resp.json()
                    message = details.get("error", {}).get("message") or str(details)
                except Exception:
                    message = resp.text[:500]
                last_error = f"{model}: HTTP {resp.status_code}: {message}"
                logger.warning("Gemini vision failed on %s: %s", model, last_error)
                continue

            body = resp.json()
            candidates = body.get("candidates") or []
            if not candidates:
                last_error = f"{model}: Gemini returned no candidates: {body}"
                continue

            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(
                p.get("text", "") for p in parts if isinstance(p, dict)
            ).strip()

            if not text:
                last_error = f"{model}: Gemini returned an empty response: {body}"
                continue

            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
            parsed = json.loads(text)

            confidence = float(parsed.get("confidence", 0) or 0)
            confidence = max(0.0, min(1.0, confidence))

            result.update({
                "available": True,
                "model": model,
                "label": _normalise_visual_label(parsed.get("label")),
                "confidence": confidence,
                "summary": str(parsed.get("summary", ""))[:180],
                "matched_claim": bool(parsed.get("matched_claim")),
                "error": None,
            })
            return result

        except requests.RequestException as exc:
            last_error = f"{model}: network error: {exc}"
        except (ValueError, KeyError, TypeError) as exc:
            last_error = f"{model}: invalid Gemini response: {exc}"
        except Exception as exc:
            last_error = f"{model}: {exc}"

    result["error"] = last_error or "Gemini visual verification failed for an unknown reason."
    return result


def _live_verification_snapshot(lat, lon):
    """Collect the same live environmental feeds used by the EWS."""
    if lat is None or lon is None:
        return {"available": False, "error": "location required"}
    weather = fetch_open_meteo(lat, lon) or {}
    earthquakes = fetch_usgs_earthquakes()
    nearby_quakes = []
    for event in earthquakes:
        try:
            d = haversine_km(lat, lon, event["lat"], event["lon"])
            if d <= 250:
                nearby_quakes.append({"magnitude": event["magnitude"], "distance_km": round(d, 1), "time": event.get("time")})
        except (KeyError, TypeError, ValueError):
            continue
    nearby_quakes.sort(key=lambda x: x["distance_km"])
    return {
        "available": bool(weather),
        "rain_72h_mm": weather.get("rain_72h_mm"),
        "soil_moisture_pct": weather.get("soil_moisture_pct"),
        "soil_source": weather.get("source", "open_meteo"),
        "nearby_quakes": nearby_quakes[:5],
        "seismic_influence": round(calculate_seismic_influence(lat, lon, earthquakes), 4),
    }


def verify_image_against_live_data(db, filepath, claimed_lat, claimed_lon, hazard_type, created_at_iso, report_id=None, client_captured_at=None, client_gps_lat=None, client_gps_lon=None):
    """Strict gate: visual identification + live feed cross-check + metadata checks."""
    raw_hazard = str(hazard_type or "").lower()
    hazard_alias = {
        "ground / slope cracks": "cracks", "slope movement": "slope_movement",
        "blocked road": "road_blockage", "landslide": "landslide", "mudslide": "mudslide",
        "rockfall": "rockfall", "flood": "flood",
    }
    hazard_type = hazard_alias.get(raw_hazard, raw_hazard.replace(" ", "_"))
    base = verify_report_media(
        db, filepath, "image", claimed_lat, claimed_lon, hazard_type, created_at_iso,
        report_id=report_id, client_captured_at=client_captured_at,
        client_gps_lat=client_gps_lat, client_gps_lon=client_gps_lon,
    )
    visual = visual_identify_image(filepath, hazard_type)
    base["visual_label"] = visual.get("label")
    base["visual_confidence"] = visual.get("confidence")
    base["visual_summary"] = visual.get("summary")
    base["visual_error"] = visual.get("error")
    base["visual_matched_claim"] = visual.get("matched_claim")
    base["live_verification"] = _live_verification_snapshot(
        base.get("media_lat") if base.get("media_lat") is not None else claimed_lat,
        base.get("media_lon") if base.get("media_lon") is not None else claimed_lon,
    )
    flags = list(base.get("flags", []))

    if not visual.get("available"):
        flags.append("visual_ai_unavailable")
    elif visual.get("error"):
        flags.append("visual_ai_error")
    elif visual.get("label") in ("no_hazard", "unclear"):
        flags.append("visual_hazard_not_confirmed")
    elif (visual.get("confidence") or 0) < 0.65:
        flags.append("visual_confidence_low")
    elif not visual.get("matched_claim"):
        flags.append("visual_claim_mismatch")

    live = base["live_verification"]
    if not live.get("available"):
        flags.append("live_feed_unavailable")
    elif hazard_type in RAIN_SENSITIVE_HAZARD_TYPES:
        rain = live.get("rain_72h_mm")
        soil = live.get("soil_moisture_pct")
        if rain is not None and soil is not None and rain < DRY_CONDITIONS_RAIN_MM and soil < DRY_CONDITIONS_SOIL_PCT:
            flags.append("live_conditions_inconsistent")

    # Strict acceptance: every required gate must pass. No manual review queue
    # for unverified media; only accepted evidence is inserted as a field report.
    # Only image-integrity/location/duplicate failures and a failed visual gate
    # reject a submission. Temporary telemetry outages are warnings, not proof
    # that the photograph is fake. This prevents an Open-Meteo/USGS outage from
    # making otherwise valid field photos impossible to submit.
    hard_reject = {
        "visual_ai_unavailable", "visual_ai_error", "visual_hazard_not_confirmed",
        "visual_confidence_low", "visual_claim_mismatch",
        "live_conditions_inconsistent", "conditions_inconsistent",
        "location_mismatch", "duplicate_media", "corrupt_or_unreadable_image",
    }
    accepted = not any(flag in hard_reject for flag in flags)
    base["flags"] = flags
    base["status"] = "verified" if accepted else "rejected"
    return base


def verify_report_media(db, filepath, media_type, claimed_lat, claimed_lon,
                         hazard_type, created_at_iso, report_id=None,
                         client_captured_at=None, client_gps_lat=None, client_gps_lon=None):
    """Run the checks described above on one just-uploaded file. Returns a
    dict that gets stored on the report row: {status, flags, ...evidence}.
    status is one of 'auto_approved', 'needs_review', 'flagged'."""
    flags = []
    evidence = {
        "media_lat": None, "media_lon": None, "media_captured_at": None,
        "media_phash": None, "distance_km": None,
        "live_rain_72h_mm": None, "live_soil_moisture_pct": None,
        "gps_source": None, "capture_time_source": None,
    }

    try:
        claimed_lat = float(claimed_lat)
        claimed_lon = float(claimed_lon)
    except (TypeError, ValueError):
        claimed_lat = claimed_lon = None

    if media_type == "video":
        # Can't read EXIF from video with Pillow — record only what we can:
        # an exact-bytes duplicate check via file hash, done by the caller.
        flags.append("video_not_deep_inspected")
        return {"status": "needs_review", "flags": flags, **evidence}

    meta = extract_image_metadata(filepath)
    # Mobile browsers and messaging apps often strip EXIF GPS/time when an
    # image is selected or resized. Fall back to the browser-captured
    # location/time, while still preserving EXIF when it exists.
    gps_source = "exif"
    captured_source = "exif"
    if meta["gps_lat"] is None and meta["gps_lon"] is None:
        try:
            if client_gps_lat is not None and client_gps_lon is not None:
                meta["gps_lat"] = float(client_gps_lat)
                meta["gps_lon"] = float(client_gps_lon)
                gps_source = "browser"
        except (TypeError, ValueError):
            pass
    # If browser GPS permission is unavailable, use the report location as a
    # clearly-labelled fallback. This is NOT claimed as proof that the camera
    # was at that location; it simply lets the live-feed check run.
    if meta["gps_lat"] is None and meta["gps_lon"] is None:
        if claimed_lat is not None and claimed_lon is not None:
            meta["gps_lat"] = float(claimed_lat)
            meta["gps_lon"] = float(claimed_lon)
            gps_source = "report_location_fallback"

    if meta["captured_at"] is None and client_captured_at:
        try:
            parsed_client_time = datetime.fromisoformat(str(client_captured_at).replace("Z", "+00:00"))
            if parsed_client_time.tzinfo is None:
                parsed_client_time = parsed_client_time.replace(tzinfo=timezone.utc)
            meta["captured_at"] = parsed_client_time.astimezone(timezone.utc).isoformat()
        except ValueError:
            pass

    # Last-resort timestamp: if the browser did not provide one, use the
    # uploaded file's modification time. This is explicitly a fallback and
    # not presented as cryptographic proof of camera capture time.
    if meta["captured_at"] is None:
        try:
            meta["captured_at"] = datetime.fromtimestamp(
                os.path.getmtime(filepath), tz=timezone.utc
            ).isoformat()
            captured_source = "server_file_time"
        except OSError:
            pass

    evidence["media_lat"] = meta["gps_lat"]
    evidence["media_lon"] = meta["gps_lon"]
    evidence["media_captured_at"] = meta["captured_at"]
    evidence["media_phash"] = meta["phash"]
    evidence["gps_source"] = gps_source if meta["gps_lat"] is not None else None
    evidence["capture_time_source"] = captured_source if meta["captured_at"] is not None else None

    if not meta["valid_image"]:
        flags.append("corrupt_or_unreadable_image")
        return {"status": "flagged", "flags": flags, **evidence}

    if (meta["width"] or 0) < 150 or (meta["height"] or 0) < 150:
        flags.append("low_resolution")

    # 1. Location cross-check
    if meta["gps_lat"] is not None and claimed_lat is not None:
        dist = haversine_km(meta["gps_lat"], meta["gps_lon"], claimed_lat, claimed_lon)
        evidence["distance_km"] = round(dist, 2)
        if dist > LOCATION_MISMATCH_KM:
            flags.append("location_mismatch")
        elif dist > LOCATION_MATCH_KM:
            flags.append("location_mismatch_minor")
    else:
        flags.append("no_gps_data")

    # 2. Recency cross-check
    if meta["captured_at"]:
        try:
            captured = datetime.fromisoformat(meta["captured_at"])
            reported = datetime.fromisoformat((created_at_iso or now_iso()).replace("Z", "+00:00"))
            if reported.tzinfo is None:
                reported = reported.replace(tzinfo=timezone.utc)
            if captured < reported - timedelta(days=STALE_MEDIA_DAYS):
                flags.append("stale_media")
            elif captured > reported + timedelta(days=1):
                flags.append("future_timestamp")
        except ValueError:
            pass
    else:
        flags.append("no_capture_time")

    # 3. Duplicate-photo cross-check
    dup = find_duplicate_media(db, meta["phash"], exclude_report_id=report_id)
    if dup:
        flags.append("duplicate_media")
        evidence["duplicate_of_report_id"] = dup["report_id"]

    # 4. Live-conditions cross-check — only meaningful for rain-driven hazard
    # types, and only a soft signal (landslides can follow non-rain triggers
    # too), so it nudges toward review rather than an outright flag.
    if hazard_type in RAIN_SENSITIVE_HAZARD_TYPES and claimed_lat is not None:
        live = fetch_open_meteo(claimed_lat, claimed_lon)
        if live:
            evidence["live_rain_72h_mm"] = live.get("rain_72h_mm")
            evidence["live_soil_moisture_pct"] = live.get("soil_moisture_pct")
            rain = live.get("rain_72h_mm")
            soil = live.get("soil_moisture_pct")
            if (
                rain is not None and rain < DRY_CONDITIONS_RAIN_MM and
                (soil is None or soil < DRY_CONDITIONS_SOIL_PCT)
            ):
                flags.append("conditions_inconsistent")

    # NOTE: this pipeline is a triage aid, not a fake-image detector — it
    # cannot reliably tell a genuine landslide photo from a spammed/reused/
    # AI-generated one. So no photo is ever auto-published: a clean photo
    # (no flags raised) still goes to "needs_review" and waits in the admin
    # queue, it just carries no warning flags for the reviewer to weigh.
    #
    # The one exception is "conditions_inconsistent": that flag is backed by
    # a live, independently-sourced feed (Open-Meteo rain/soil data for the
    # claimed coordinates right now), not a guess about the image itself. If
    # someone claims a rain-driven hazard (landslide, slope movement, etc.)
    # in a spot that has recorded no rain and no wet soil, no human judgment
    # call is needed — the claim is contradicted by real-time ground truth,
    # so it's rejected and the file removed automatically (see _insert_report).
    if not flags:
        flags.append("awaiting_manual_approval")

    reject_flags = {"conditions_inconsistent"}
    hard_flags = {"corrupt_or_unreadable_image", "location_mismatch", "duplicate_media"}
    if any(f in reject_flags for f in flags):
        status = "rejected"
    elif any(f in hard_flags for f in flags):
        status = "flagged"
    else:
        status = "needs_review"

    return {"status": status, "flags": flags, **evidence}


def hash_file(filepath):
    """SHA-256 of raw file bytes — used as a cheap exact-duplicate check for
    videos, which the EXIF/perceptual-hash pipeline above can't inspect."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()




@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


@app.route("/api/<path:_any>", methods=["OPTIONS"])
def cors_preflight(_any):
    return "", 204


@app.errorhandler(413)
def too_large(_exc):
    return jsonify({"error": "file too large (30MB max)"}), 413


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH, timeout=10)
        g.db.row_factory = sqlite3.Row
        # WAL mode lets reads proceed while a write is in progress, which
        # matters now that gunicorn runs multiple worker processes against
        # the same SQLite file. busy_timeout avoids "database is locked"
        # errors under brief write contention instead of failing immediately.
        g.db.execute("PRAGMA journal_mode=WAL")
        g.db.execute("PRAGMA busy_timeout=10000")
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS reports (
            id TEXT PRIMARY KEY,
            client_id TEXT,
            lat REAL,
            lon REAL,
            district TEXT,
            hazard_type TEXT,
            severity INTEGER,
            remarks TEXT,
            media_filename TEXT,
            media_type TEXT,
            created_at TEXT,
            synced_at TEXT
        );

        CREATE TABLE IF NOT EXISTS alerts (
            id TEXT PRIMARY KEY,
            district TEXT,
            level TEXT,
            language TEXT,
            message TEXT,
            dispatched_at TEXT
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id TEXT PRIMARY KEY,
            district TEXT,
            rain_72h_mm REAL,
            soil_moisture_pct REAL,
            slope_deg REAL,
            seismic_influence REAL DEFAULT 0,
            probability REAL,
            created_at TEXT
        );

        CREATE TABLE IF NOT EXISTS sensor_readings (
            id TEXT PRIMARY KEY,
            sensor_id TEXT,
            sensor_type TEXT,
            district TEXT,
            lat REAL,
            lon REAL,
            value REAL,
            unit TEXT,
            recorded_at TEXT,
            received_at TEXT
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_reports_client_id
            ON reports (client_id) WHERE client_id IS NOT NULL;

        CREATE INDEX IF NOT EXISTS idx_reports_created_at
            ON reports (created_at DESC);

        CREATE INDEX IF NOT EXISTS idx_reports_verification_status
            ON reports (verification_status);

        CREATE INDEX IF NOT EXISTS idx_sensor_type_recorded
            ON sensor_readings (sensor_type, recorded_at DESC);

        CREATE INDEX IF NOT EXISTS idx_predictions_created_at
            ON predictions (created_at DESC);
        """
    )

    # Migrate older databases with media-upload and verification columns.
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(reports)").fetchall()}
    for col, coltype in (
        ("media_filename", "TEXT"), ("media_type", "TEXT"),
        ("verification_status", "TEXT"), ("verification_flags", "TEXT"),
        ("media_lat", "REAL"), ("media_lon", "REAL"),
        ("media_captured_at", "TEXT"), ("media_phash", "TEXT"),
        ("media_sha256", "TEXT"), ("distance_km", "REAL"),
        ("live_rain_72h_mm", "REAL"), ("live_soil_moisture_pct", "REAL"),
        ("duplicate_of_report_id", "TEXT"),
        ("visual_label", "TEXT"), ("visual_confidence", "REAL"),
        ("visual_summary", "TEXT"), ("visual_matched_claim", "INTEGER"),
        ("live_verification_json", "TEXT"),
        ("reviewed_by_admin", "INTEGER"), ("reviewed_at", "TEXT"),
    ):
        if col not in existing_cols:
            conn.execute(f"ALTER TABLE reports ADD COLUMN {col} {coltype}")

    # Migrate prediction records to store the additional seismic model input.
    prediction_cols = {
        row[1] for row in conn.execute("PRAGMA table_info(predictions)").fetchall()
    }
    if "seismic_influence" not in prediction_cols:
        conn.execute(
            "ALTER TABLE predictions ADD COLUMN seismic_influence REAL DEFAULT 0"
        )

    conn.commit()
    conn.close()


def _train_model():
    rng = np.random.default_rng(42)
    n = 6000

    rain_72h = np.clip(
        rng.gamma(shape=2.0, scale=35, size=n),
        0,
        400,
    )

    soil_moisture_pct = np.clip(
        rng.normal(loc=45, scale=18, size=n),
        5,
        100,
    )

    slope_deg = rng.uniform(5, 55, size=n)

    # Prototype seismic-influence feature.
    # This is deliberately a normalized influence index (0..1), not raw
    # earthquake magnitude. It is generated synthetically here because
    # the current project does not contain a field-validated earthquake/
    # landslide training dataset.
    seismic_influence = rng.beta(
        1.4,
        5.0,
        size=n,
    )

    duration_hr = 72.0
    mean_intensity = rain_72h / duration_hr
    caine_threshold = 14.82 * (duration_hr ** -0.39)
    rainfall_stress = mean_intensity / caine_threshold

    slope_term = (slope_deg - 30) / 10.0
    moisture_term = (soil_moisture_pct - 50) / 20.0
    rainfall_term = (rainfall_stress - 1.0) * 2.0

    logit = (
        1.15 * rainfall_term
        + 1.00 * moisture_term
        + 1.05 * slope_term
        + 0.60 * rainfall_term * moisture_term
        + 0.45 * seismic_influence
        - 1.2
    )

    true_prob = 1 / (1 + np.exp(-logit))
    labels = rng.binomial(1, true_prob)

    X = np.column_stack([
        rain_72h,
        soil_moisture_pct,
        slope_deg,
        seismic_influence,
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingClassifier(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.05,
            random_state=42,
        )),
    ])

    pipeline.fit(X_train, y_train)

    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred_default = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": round(
            float((y_pred_default == y_test).mean()),
            4,
        ),
        "precision": round(
            float(
                precision_score(
                    y_test,
                    y_pred_default,
                    zero_division=0,
                )
            ),
            4,
        ),
        "recall": round(
            float(
                recall_score(
                    y_test,
                    y_pred_default,
                    zero_division=0,
                )
            ),
            4,
        ),
        "f1": round(
            float(
                f1_score(
                    y_test,
                    y_pred_default,
                    zero_division=0,
                )
            ),
            4,
        ),
        "roc_auc": round(
            float(
                roc_auc_score(
                    y_test,
                    y_prob,
                )
            ),
            4,
        ),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }

    precision, recall, thresholds = precision_recall_curve(
        y_test,
        y_prob,
    )

    precision, recall = precision[:-1], recall[:-1]

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        beta = 2.0
        fbeta = (
            (1 + beta**2)
            * precision
            * recall
            / (
                beta**2 * precision
                + recall
            )
        )

    fbeta = np.nan_to_num(fbeta)
    best_idx = int(np.argmax(fbeta))

    alert_threshold = {
        "threshold": round(
            float(thresholds[best_idx]),
            4,
        ),
        "precision": round(
            float(precision[best_idx]),
            4,
        ),
        "recall": round(
            float(recall[best_idx]),
            4,
        ),
    }

    model_info = {
        "model_type": "gradient_boosting",
        "features": [
            "antecedent_rainfall_72h_mm",
            "soil_moisture_pct",
            "slope_deg",
            "seismic_influence",
        ],
        "metrics": metrics,
        "alert_threshold": alert_threshold,
        "note": (
            "Seismic influence is a prototype normalized feature. "
            "Replace synthetic training data with real NER-linked "
            "earthquake/landslide records before operational deployment."
        ),
    }

    return pipeline, model_info


_PIPELINE, LANDSLIDE_MODEL = _train_model()


def predict_probability(
    rain_72h_mm,
    soil_moisture_pct,
    slope_deg,
    seismic_influence=0.0,
):
    row = [[
        rain_72h_mm,
        soil_moisture_pct,
        slope_deg,
        seismic_influence,
    ]]

    return float(
        _PIPELINE.predict_proba(row)[0][1]
    )


def predict_probabilities_batch(rows):
    if not rows:
        return []

    X = np.array(
        rows,
        dtype=float,
    )

    return _PIPELINE.predict_proba(
        X
    )[:, 1].tolist()


# ---------------------------------------------------------------------------
# Live seismic activity (USGS)
# ---------------------------------------------------------------------------

SEISMIC_NER_BOUNDS = {
    "minlat": 21.0,
    "maxlat": 30.0,
    "minlon": 88.0,
    "maxlon": 98.5,
}

SEISMIC_CACHE_SECONDS = 300
_SEISMIC_CACHE = {
    "fetched_at": 0.0,
    "events": [],
}


def haversine_km(
    lat1,
    lon1,
    lat2,
    lon2,
):
    radius = 6371.0088

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    dphi = math.radians(
        lat2 - lat1
    )

    dlambda = math.radians(
        lon2 - lon1
    )

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1)
        * math.cos(phi2)
        * math.sin(dlambda / 2) ** 2
    )

    return (
        2
        * radius
        * math.atan2(
            math.sqrt(a),
            math.sqrt(
                max(
                    0.0,
                    1.0 - a,
                )
            ),
        )
    )


def fetch_usgs_earthquakes(force=False):
    now_ts = datetime.now(
        timezone.utc
    ).timestamp()

    if (
        not force
        and
        now_ts - _SEISMIC_CACHE["fetched_at"]
        < SEISMIC_CACHE_SECONDS
    ):
        return _SEISMIC_CACHE["events"]

    try:
        params = {
            "format": "geojson",
            "starttime": datetime.fromtimestamp(
                now_ts - 7 * 86400,
                tz=timezone.utc,
            ).strftime("%Y-%m-%dT%H:%M:%S"),
            "endtime": datetime.now(
                timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%S"),
            "minmagnitude": 2.5,
            "minlatitude": SEISMIC_NER_BOUNDS["minlat"],
            "maxlatitude": SEISMIC_NER_BOUNDS["maxlat"],
            "minlongitude": SEISMIC_NER_BOUNDS["minlon"],
            "maxlongitude": SEISMIC_NER_BOUNDS["maxlon"],
            "orderby": "time",
            "limit": 500,
        }

        response = requests.get(
            "https://earthquake.usgs.gov/fdsnws/event/1/query",
            params=params,
            headers={
                "User-Agent": "NER-Landslide-EWS/1.0"
            },
            timeout=10,
        )

        response.raise_for_status()

        payload = response.json()
        events = []

        for feature in payload.get(
            "features",
            [],
        ):
            geometry = feature.get(
                "geometry",
                {},
            )

            coords = geometry.get(
                "coordinates",
                [],
            )

            props = feature.get(
                "properties",
                {},
            )

            if len(coords) < 2:
                continue

            lon = float(coords[0])
            lat = float(coords[1])

            depth = (
                float(coords[2])
                if len(coords) > 2
                and coords[2] is not None
                else None
            )

            mag = props.get("mag")

            if mag is None:
                continue

            events.append({
                "id": feature.get("id"),
                "magnitude": float(mag),
                "lat": lat,
                "lon": lon,
                "depth_km": depth,
                "place": props.get("place")
                or "Northeast India",
                "time": (
                    datetime.fromtimestamp(
                        float(props.get("time")) / 1000.0,
                        tz=timezone.utc,
                    ).isoformat()
                    if props.get("time") is not None
                    else None
                ),
                "url": props.get("url"),
            })

        _SEISMIC_CACHE["fetched_at"] = now_ts
        _SEISMIC_CACHE["events"] = events

        return events

    except (
        requests.RequestException,
        ValueError,
        TypeError,
        KeyError,
    ) as exc:
        logger.warning(
            "USGS earthquake feed unavailable: %s",
            exc,
        )
        return _SEISMIC_CACHE["events"]


def calculate_seismic_influence(
    lat,
    lon,
    events=None,
):
    events = (
        fetch_usgs_earthquakes()
        if events is None
        else events
    )

    if not events:
        return 0.0

    now = datetime.now(
        timezone.utc
    )

    influence = 0.0

    for event in events:

        magnitude = float(
            event.get(
                "magnitude",
                0,
            )
        )

        distance = haversine_km(
            lat,
            lon,
            float(event["lat"]),
            float(event["lon"]),
        )

        time_str = event.get(
            "time"
        )

        try:
            event_time = datetime.fromisoformat(
                time_str.replace(
                    "Z",
                    "+00:00",
                )
            )
            age_days = max(
                0.0,
                (
                    now - event_time
                ).total_seconds()
                / 86400.0,
            )
        except (
            TypeError,
            ValueError,
        ):
            age_days = 7.0

        magnitude_factor = max(
            0.0,
            magnitude - 2.5,
        )

        distance_factor = math.exp(
            -distance / 150.0
        )

        recency_factor = math.exp(
            -age_days / 3.0
        )

        influence += (
            magnitude_factor
            * distance_factor
            * recency_factor
            / 4.0
        )

    return round(
        float(
            np.clip(
                influence,
                0.0,
                1.0,
            )
        ),
        4,
    )


def now_iso():
    return datetime.now(timezone.utc).isoformat()


@app.route("/api/health", methods=["GET"])
def health():
    db_ok = True
    try:
        get_db().execute("SELECT 1").fetchone()
    except sqlite3.Error:
        db_ok = False
    return jsonify({
        "status": "ok" if db_ok else "degraded",
        "version": APP_VERSION,
        "uptime_seconds": round((datetime.now(timezone.utc) - STARTED_AT).total_seconds()),
        "database": "ok" if db_ok else "unreachable",
        "model": LANDSLIDE_MODEL["model_type"],
        "model_features": LANDSLIDE_MODEL["features"],
        "seismic_feed": "USGS live",
        "sms_gateway": "configured" if SMS_GATEWAY_CONFIGURED else "not configured (dispatch logs only)",
        "imd_data_gov_key": "configured" if IMD_API_KEY else "not configured (using Open-Meteo fallback)",
    })


# ---------------------------------------------------------------------------
# Northeast India district registry
# ---------------------------------------------------------------------------
# The frontend used to own a small 34-district list.  The backend now owns the
# authoritative application registry so every dashboard/client can consume the
# same Northeast-wide district set. Coordinates are representative district-
# headquarters points (not polygon centroids) and are used for live weather,
# seismic influence and map navigation.
#
# District names are aligned to current 2026 state/IGOD listings.  Assam's
# Karimganj is represented by its current name, Sribhumi.
# ---------------------------------------------------------------------------

NE_DISTRICTS = [
    # Assam (35)
    ("baksa", "Baksa", "Assam", "Mushalpur", 26.75, 91.10),
    ("bajali", "Bajali", "Assam", "Pathsala", 26.49, 91.17),
    ("barpeta", "Barpeta", "Assam", "Barpeta", 26.32, 91.00),
    ("biswanath", "Biswanath", "Assam", "Biswanath Chariali", 26.73, 93.15),
    ("bongaigaon", "Bongaigaon", "Assam", "Bongaigaon", 26.48, 90.56),
    ("cachar", "Cachar", "Assam", "Silchar", 24.83, 92.78),
    ("charaideo", "Charaideo", "Assam", "Sonari", 27.02, 95.02),
    ("chirang", "Chirang", "Assam", "Kajalgaon", 26.52, 90.57),
    ("darrang", "Darrang", "Assam", "Mangaldai", 26.44, 92.03),
    ("dhemaji", "Dhemaji", "Assam", "Dhemaji", 27.48, 94.58),
    ("dhubri", "Dhubri", "Assam", "Dhubri", 26.02, 89.98),
    ("dibrugarh", "Dibrugarh", "Assam", "Dibrugarh", 27.47, 94.91),
    ("dima_hasao", "Dima Hasao", "Assam", "Haflong", 25.17, 93.02),
    ("goalpara", "Goalpara", "Assam", "Goalpara", 26.17, 90.62),
    ("golaghat", "Golaghat", "Assam", "Golaghat", 26.52, 93.97),
    ("hailakandi", "Hailakandi", "Assam", "Hailakandi", 24.68, 92.56),
    ("hojai", "Hojai", "Assam", "Hojai", 26.00, 92.86),
    ("jorhat", "Jorhat", "Assam", "Jorhat", 26.75, 94.20),
    ("kamrup", "Kamrup", "Assam", "Amingaon", 26.22, 91.72),
    ("kamrup_metropolitan", "Kamrup Metropolitan", "Assam", "Guwahati", 26.14, 91.74),
    ("karbi_anglong", "Karbi Anglong", "Assam", "Diphu", 25.84, 93.44),
    ("sribhumi", "Sribhumi", "Assam", "Karimganj", 24.87, 92.36),
    ("kokrajhar", "Kokrajhar", "Assam", "Kokrajhar", 26.40, 90.27),
    ("lakhimpur", "Lakhimpur", "Assam", "North Lakhimpur", 27.24, 94.11),
    ("majuli", "Majuli", "Assam", "Garamur", 27.00, 94.26),
    ("morigaon", "Morigaon", "Assam", "Morigaon", 26.25, 92.34),
    ("nagaon", "Nagaon", "Assam", "Nagaon", 26.35, 92.68),
    ("nalbari", "Nalbari", "Assam", "Nalbari", 26.44, 91.44),
    ("sivasagar", "Sivasagar", "Assam", "Sivasagar", 26.98, 94.64),
    ("sonitpur", "Sonitpur", "Assam", "Tezpur", 26.63, 92.80),
    ("south_salmara_mankachar", "South Salmara-Mankachar", "Assam", "Hatsingimari", 25.70, 89.87),
    ("tamulpur", "Tamulpur", "Assam", "Tamulpur", 26.64, 91.56),
    ("tinsukia", "Tinsukia", "Assam", "Tinsukia", 27.49, 95.36),
    ("udalguri", "Udalguri", "Assam", "Udalguri", 26.77, 92.10),
    ("west_karbi_anglong", "West Karbi Anglong", "Assam", "Hamren", 25.83, 92.76),

    # Arunachal Pradesh (28)
    ("anjaw", "Anjaw", "Arunachal Pradesh", "Hawai", 28.07, 96.99),
    ("bichom", "Bichom", "Arunachal Pradesh", "Napangphung", 27.31, 92.62),
    ("changlang", "Changlang", "Arunachal Pradesh", "Changlang", 27.12, 96.73),
    ("dibang_valley", "Dibang Valley", "Arunachal Pradesh", "Anini", 28.78, 95.90),
    ("east_kameng", "East Kameng", "Arunachal Pradesh", "Seppa", 27.33, 93.05),
    ("east_siang", "East Siang", "Arunachal Pradesh", "Pasighat", 28.07, 95.33),
    ("kamle", "Kamle", "Arunachal Pradesh", "Raga", 27.57, 93.63),
    ("keyi_panyor", "Keyi Panyor", "Arunachal Pradesh", "Yachuli", 27.55, 93.58),
    ("kra_daadi", "Kra Daadi", "Arunachal Pradesh", "Jamin", 27.57, 93.68),
    ("kurung_kumey", "Kurung Kumey", "Arunachal Pradesh", "Koloriang", 27.92, 93.35),
    ("lepa_rada", "Lepa-Rada", "Arunachal Pradesh", "Basar", 27.99, 94.69),
    ("lohit", "Lohit", "Arunachal Pradesh", "Tezu", 27.92, 96.16),
    ("longding", "Longding", "Arunachal Pradesh", "Longding", 26.87, 95.32),
    ("lower_dibang_valley", "Lower Dibang Valley", "Arunachal Pradesh", "Roing", 28.14, 95.84),
    ("lower_siang", "Lower Siang", "Arunachal Pradesh", "Likabali", 27.65, 94.70),
    ("lower_subansiri", "Lower Subansiri", "Arunachal Pradesh", "Ziro", 27.63, 93.83),
    ("namsai", "Namsai", "Arunachal Pradesh", "Namsai", 27.67, 95.86),
    ("pakke_kessang", "Pakke-Kessang", "Arunachal Pradesh", "Lemmi", 27.24, 93.30),
    ("papum_pare", "Papum Pare", "Arunachal Pradesh", "Yupia", 27.10, 93.62),
    ("shi_yomi", "Shi-Yomi", "Arunachal Pradesh", "Tato", 28.04, 94.25),
    ("siang", "Siang", "Arunachal Pradesh", "Boleng", 28.19, 94.61),
    ("tawang", "Tawang", "Arunachal Pradesh", "Tawang", 27.59, 91.87),
    ("tirap", "Tirap", "Arunachal Pradesh", "Khonsa", 27.02, 95.57),
    ("upper_siang", "Upper Siang", "Arunachal Pradesh", "Yingkiong", 28.61, 95.05),
    ("upper_subansiri", "Upper Subansiri", "Arunachal Pradesh", "Daporijo", 27.98, 94.22),
    ("west_kameng", "West Kameng", "Arunachal Pradesh", "Bomdila", 27.26, 92.40),
    ("west_siang", "West Siang", "Arunachal Pradesh", "Aalo", 28.17, 94.80),
    ("tawang", "Tawang", "Arunachal Pradesh", "Tawang", 27.59, 91.87),

    # Manipur (16)
    ("bishnupur", "Bishnupur", "Manipur", "Bishnupur", 24.63, 93.77),
    ("chandel", "Chandel", "Manipur", "Chandel", 24.32, 94.00),
    ("churachandpur", "Churachandpur", "Manipur", "Churachandpur", 24.33, 93.68),
    ("imphal_east", "Imphal East", "Manipur", "Porompat", 24.81, 93.95),
    ("imphal_west", "Imphal West", "Manipur", "Imphal", 24.82, 93.94),
    ("jiribam", "Jiribam", "Manipur", "Jiribam", 24.80, 93.12),
    ("kakching", "Kakching", "Manipur", "Kakching", 24.50, 94.01),
    ("kamjong", "Kamjong", "Manipur", "Kamjong", 24.95, 94.36),
    ("kangpokpi", "Kangpokpi", "Manipur", "Kangpokpi", 25.15, 93.98),
    ("noney", "Noney", "Manipur", "Noney", 24.79, 93.65),
    ("pherzawl", "Pherzawl", "Manipur", "Pherzawl", 24.25, 93.17),
    ("senapati", "Senapati", "Manipur", "Senapati", 25.27, 94.02),
    ("tamenglong", "Tamenglong", "Manipur", "Tamenglong", 24.98, 93.53),
    ("tengnoupal", "Tengnoupal", "Manipur", "Tengnoupal", 24.41, 94.16),
    ("thoubal", "Thoubal", "Manipur", "Thoubal", 24.64, 94.01),
    ("ukhrul", "Ukhrul", "Manipur", "Ukhrul", 25.10, 94.37),

    # Meghalaya (12)
    ("east_garo_hills", "East Garo Hills", "Meghalaya", "Williamnagar", 25.50, 90.62),
    ("east_jaintia_hills", "East Jaintia Hills", "Meghalaya", "Khliehriat", 25.55, 92.37),
    ("east_khasi_hills", "East Khasi Hills", "Meghalaya", "Shillong", 25.57, 91.88),
    ("eastern_west_khasi_hills", "Eastern West Khasi Hills", "Meghalaya", "Mairang", 25.56, 91.64),
    ("north_garo_hills", "North Garo Hills", "Meghalaya", "Resubelpara", 25.90, 90.62),
    ("ri_bhoi", "Ri Bhoi", "Meghalaya", "Nongpoh", 25.90, 91.88),
    ("south_garo_hills", "South Garo Hills", "Meghalaya", "Baghmara", 25.30, 90.59),
    ("south_west_garo_hills", "South West Garo Hills", "Meghalaya", "Ampati", 25.47, 89.93),
    ("south_west_khasi_hills", "South West Khasi Hills", "Meghalaya", "Mawkyrwat", 25.31, 91.22),
    ("west_garo_hills", "West Garo Hills", "Meghalaya", "Tura", 25.57, 90.22),
    ("west_jaintia_hills", "West Jaintia Hills", "Meghalaya", "Jowai", 25.45, 92.20),
    ("west_khasi_hills", "West Khasi Hills", "Meghalaya", "Nongstoin", 25.52, 91.27),

    # Mizoram (11)
    ("aizawl", "Aizawl", "Mizoram", "Aizawl", 23.73, 92.72),
    ("champhai", "Champhai", "Mizoram", "Champhai", 23.46, 93.33),
    ("hnahthial", "Hnahthial", "Mizoram", "Hnahthial", 22.97, 92.93),
    ("khawzawl", "Khawzawl", "Mizoram", "Khawzawl", 23.49, 93.14),
    ("kolasib", "Kolasib", "Mizoram", "Kolasib", 24.23, 92.68),
    ("lawngtlai", "Lawngtlai", "Mizoram", "Lawngtlai", 22.53, 92.90),
    ("lunglei", "Lunglei", "Mizoram", "Lunglei", 22.88, 92.73),
    ("mamit", "Mamit", "Mizoram", "Mamit", 23.93, 92.49),
    ("saitual", "Saitual", "Mizoram", "Saitual", 23.85, 93.09),
    ("serchhip", "Serchhip", "Mizoram", "Serchhip", 23.34, 92.85),
    ("siaha", "Siaha", "Mizoram", "Siaha", 22.49, 92.98),

    # Nagaland (17)
    ("chumoukedima", "Chumoukedima", "Nagaland", "Chumoukedima", 25.78, 93.78),
    ("dimapur", "Dimapur", "Nagaland", "Dimapur", 25.91, 93.73),
    ("kiphire", "Kiphire", "Nagaland", "Kiphire", 25.89, 94.78),
    ("kohima", "Kohima", "Nagaland", "Kohima", 25.67, 94.12),
    ("longleng", "Longleng", "Nagaland", "Longleng", 26.48, 94.82),
    ("meluri", "Meluri", "Nagaland", "Meluri", 25.49, 94.51),
    ("mokokchung", "Mokokchung", "Nagaland", "Mokokchung", 26.33, 94.53),
    ("mon", "Mon", "Nagaland", "Mon", 26.72, 95.03),
    ("niuland", "Niuland", "Nagaland", "Niuland", 25.70, 93.77),
    ("noklak", "Noklak", "Nagaland", "Noklak", 26.10, 95.02),
    ("peren", "Peren", "Nagaland", "Peren", 25.56, 93.67),
    ("phek", "Phek", "Nagaland", "Phek", 25.67, 94.47),
    ("shamator", "Shamator", "Nagaland", "Shamator", 26.07, 94.95),
    ("tseminyu", "Tseminyu", "Nagaland", "Tseminyu", 25.94, 94.16),
    ("tuensang", "Tuensang", "Nagaland", "Tuensang", 26.27, 94.82),
    ("wokha", "Wokha", "Nagaland", "Wokha", 26.10, 94.26),
    ("zunheboto", "Zunheboto", "Nagaland", "Zunheboto", 26.02, 94.53),

    # Sikkim (6)
    ("gangtok", "Gangtok", "Sikkim", "Gangtok", 27.33, 88.61),
    ("gyalshing", "Gyalshing", "Sikkim", "Gyalshing", 27.29, 88.26),
    ("mangan", "Mangan", "Sikkim", "Mangan", 27.50, 88.53),
    ("namchi", "Namchi", "Sikkim", "Namchi", 27.17, 88.36),
    ("pakyong", "Pakyong", "Sikkim", "Pakyong", 27.23, 88.61),
    ("soreng", "Soreng", "Sikkim", "Soreng", 27.17, 88.19),

    # Tripura (8)
    ("dhalai", "Dhalai", "Tripura", "Ambassa", 23.93, 91.85),
    ("gomati", "Gomati", "Tripura", "Udaipur", 23.53, 91.48),
    ("khowai", "Khowai", "Tripura", "Khowai", 24.07, 91.61),
    ("north_tripura", "North Tripura", "Tripura", "Dharmanagar", 24.38, 92.17),
    ("sepahijala", "Sepahijala", "Tripura", "Bishramganj", 23.67, 91.31),
    ("south_tripura", "South Tripura", "Tripura", "Belonia", 23.25, 91.45),
    ("unakoti", "Unakoti", "Tripura", "Kailashahar", 24.33, 92.00),
    ("west_tripura", "West Tripura", "Tripura", "Agartala", 23.83, 91.28),
]

# The model historically used curated representative slopes for the original
# dashboard districts. Preserve those values; new districts use 30 degrees as
# a neutral fallback until a terrain-derived slope service is wired in.
_CURATED_SLOPES = {
    "kamrup_metropolitan": 15, "dima_hasao": 32, "karbi_anglong": 28,
    "cachar": 12, "sonitpur": 22, "east_khasi_hills": 38,
    "west_khasi_hills": 34, "ri_bhoi": 30, "west_garo_hills": 32,
    "east_jaintia_hills": 33, "papum_pare": 30, "west_kameng": 42,
    "upper_subansiri": 40, "lower_dibang_valley": 38, "tawang": 45,
    "kohima": 35, "dimapur": 14, "mokokchung": 33, "phek": 34,
    "imphal_west": 10, "churachandpur": 34, "ukhrul": 36,
    "senapati": 33, "aizawl": 40, "lunglei": 38, "champhai": 36,
    "kolasib": 32, "west_tripura": 14, "dhalai": 28, "gomati": 20,
    "gangtok": 42, "mangan": 46, "namchi": 40, "gyalshing": 41,
}

# Remove accidental duplicate IDs while preserving first occurrence.
NE_DISTRICTS = [
    {
        "id": row[0], "name": row[1], "state": row[2], "headquarters": row[3],
        "lat": float(row[4]), "lon": float(row[5]),
        "slope": float(_CURATED_SLOPES.get(row[0], 30.0)),
        "coordinate_source": "district headquarters",
    }
    for row in NE_DISTRICTS
]
_seen_district_ids = set()
NE_DISTRICTS = [d for d in NE_DISTRICTS if not (d["id"] in _seen_district_ids or _seen_district_ids.add(d["id"]))]
NE_DISTRICT_BY_ID = {d["id"]: d for d in NE_DISTRICTS}
NE_DISTRICT_BY_NAME = {d["name"].casefold(): d for d in NE_DISTRICTS}

# Short-lived whole-response cache for the expensive regional dashboard feed.
# This prevents a fresh browser navigation (or returning from GIS) from
# waiting on 100+ weather lookups before the dashboard can become usable.
_REGIONAL_DASHBOARD_CACHE = {"payload": None, "fetched_at": 0.0}
_REGIONAL_DASHBOARD_CACHE_SECONDS = 300
_REGIONAL_DASHBOARD_LOCK = threading.Lock()


def _district_registry_response():
    return [dict(d) for d in NE_DISTRICTS]


@app.route("/api/districts", methods=["GET"])
def list_districts():
    """Return the complete Northeast India district registry used by the app."""
    state = request.args.get("state", "").strip().casefold()
    rows = NE_DISTRICTS
    if state:
        rows = [d for d in rows if d["state"].casefold() == state]
    return jsonify({
        "region": "Northeast India",
        "states": sorted({d["state"] for d in rows}),
        "count": len(rows),
        "districts": [dict(d) for d in rows],
        "coordinate_note": "Coordinates represent district headquarters and are used as the district-level prediction/map point.",
    })


@app.route("/api/regional-districts", methods=["GET"])
def regional_districts():
    """Return live regional risk data, using a short cache for fast navigation."""
    now_ts = datetime.now(timezone.utc).timestamp()
    cached = _REGIONAL_DASHBOARD_CACHE.get("payload")
    if cached is not None and now_ts - _REGIONAL_DASHBOARD_CACHE.get("fetched_at", 0) < _REGIONAL_DASHBOARD_CACHE_SECONDS:
        response = jsonify(cached)
        response.headers["X-Regional-Cache"] = "HIT"
        return response

    # Only one request computes the expensive cold-cache payload. Other
    # requests wait briefly for the same result rather than spawning another
    # 100+ weather-call batch.
    with _REGIONAL_DASHBOARD_LOCK:
        now_ts = datetime.now(timezone.utc).timestamp()
        cached = _REGIONAL_DASHBOARD_CACHE.get("payload")
        if cached is not None and now_ts - _REGIONAL_DASHBOARD_CACHE.get("fetched_at", 0) < _REGIONAL_DASHBOARD_CACHE_SECONDS:
            response = jsonify(cached)
            response.headers["X-Regional-Cache"] = "HIT"
            return response

        seismic_events = fetch_usgs_earthquakes()
        soil_sensor_rows = _fetch_soil_sensor_rows(get_db())
        rows = []

        with ThreadPoolExecutor(max_workers=16) as pool:
            weather_by_district = dict(
                zip(
                    (d["id"] for d in NE_DISTRICTS),
                    pool.map(lambda d: fetch_open_meteo(d["lat"], d["lon"]) or {}, NE_DISTRICTS),
                )
            )

        for d in NE_DISTRICTS:
            weather = weather_by_district.get(d["id"], {})
            rain = _to_float(weather.get("rain_72h_mm"))
            soil = _to_float(weather.get("soil_moisture_pct"))
            sensor = get_latest_soil_sensor(d["lat"], d["lon"], rows=soil_sensor_rows)
            if sensor:
                soil = float(sensor["soil_moisture_pct"])

            probability = predict_probability(
                rain if rain is not None else 0.0,
                soil if soil is not None else 45.0,
                d["slope"],
                calculate_seismic_influence(d["lat"], d["lon"], seismic_events),
            )
            hazard_index = round(probability * 100)
            level = "critical" if hazard_index >= 70 else ("elevated" if hazard_index >= 40 else "watch")
            rows.append({
                **d,
                "rain_72h_mm": rain,
                "soil_moisture_pct": soil,
                "soil_moisture_source": "ground_sensor" if sensor else weather.get("source"),
                "soil_sensor_id": sensor["sensor_id"] if sensor else None,
                "seismic_influence": calculate_seismic_influence(d["lat"], d["lon"], seismic_events),
                "probability": probability,
                "hazard_index": hazard_index,
                "level": level,
            })

        rows.sort(key=lambda r: (-r["hazard_index"], r["state"], r["name"]))
        critical = sum(r["hazard_index"] >= 70 for r in rows)
        elevated = sum(40 <= r["hazard_index"] < 70 for r in rows)
        overall = round(sum(r["hazard_index"] for r in rows) / len(rows)) if rows else 0
        payload = {
            "region": "Northeast India",
            "district_count": len(rows),
            "overall_risk": overall,
            "critical_count": critical,
            "elevated_count": elevated,
            "watch_count": len(rows) - critical - elevated,
            "districts": rows,
            "seismic_events_used": len(seismic_events),
            "generated_at": now_iso(),
        }
        _REGIONAL_DASHBOARD_CACHE["payload"] = payload
        _REGIONAL_DASHBOARD_CACHE["fetched_at"] = datetime.now(timezone.utc).timestamp()

    response = jsonify(payload)
    response.headers["X-Regional-Cache"] = "MISS"
    return response



# ---------------------------------------------------------------------------
# Weather: IMD-first with an automatic, clearly-labeled Open-Meteo fallback
# ---------------------------------------------------------------------------
# India Meteorological Department does not publish a simple free public REST
# API. The closest official route is IMD's datasets mirrored on the
# data.gov.in Open Government Data platform, which requires a (free)
# registered API key. To avoid silently pretending we have official IMD
# access we don't have, this only calls data.gov.in when DATA_GOV_IN_API_KEY
# is set; otherwise it (openly) falls back to Open-Meteo, same as the rest of
# this dashboard, and says so in the response's "source" field.
IMD_API_KEY = os.environ.get("DATA_GOV_IN_API_KEY", "").strip()
IMD_RESOURCE_ID = os.environ.get(
    "DATA_GOV_IN_IMD_RESOURCE_ID",
    "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69",  # IMD city rainfall/weather resource on data.gov.in
)


def fetch_imd_observation(lat, lon):
    """Best-effort fetch from IMD data mirrored on data.gov.in. Returns None
    on any failure or if no key is configured, so callers can fall back."""
    if not IMD_API_KEY:
        return None
    try:
        resp = requests.get(
            f"https://api.data.gov.in/resource/{IMD_RESOURCE_ID}",
            params={"api-key": IMD_API_KEY, "format": "json", "limit": 5},
            timeout=6,
        )
        resp.raise_for_status()
        records = resp.json().get("records", [])
        if not records:
            return None
        return {"source": "imd_data_gov_in", "records": records}
    except (requests.RequestException, ValueError):
        return None


SOIL_SENSOR_CACHE_SECONDS = 20
_SOIL_SENSOR_CACHE = {
    "fetched_at": 0.0,
    "rows": [],
}


def _fetch_soil_sensor_rows(db, force=False):
    """Fetch (and briefly cache) the raw soil-moisture sensor rows.

    This used to run fresh on every call to get_latest_soil_sensor(), which
    meant a single /api/predict/grid request (~140 grid points) issued ~140
    identical 500-row queries in a row. A short in-process cache means one
    query serves an entire grid request (and the handful of requests that
    land within the same few seconds), while still staying fresh enough for
    live sensor data.
    """
    now_ts = datetime.now(timezone.utc).timestamp()
    if (
        not force
        and now_ts - _SOIL_SENSOR_CACHE["fetched_at"] < SOIL_SENSOR_CACHE_SECONDS
    ):
        return _SOIL_SENSOR_CACHE["rows"]

    rows = db.execute(
        """
        SELECT *
        FROM sensor_readings
        WHERE sensor_type IN (
            'soil_moisture',
            'soil-moisture',
            'soil moisture',
            'soil'
        )
        ORDER BY recorded_at DESC
        LIMIT 500
        """
    ).fetchall()

    _SOIL_SENSOR_CACHE["fetched_at"] = now_ts
    _SOIL_SENSOR_CACHE["rows"] = rows
    return rows


def get_latest_soil_sensor(
    lat,
    lon,
    radius_km=50.0,
    rows=None,
):
    """Return a nearby live soil-moisture sensor reading, if one exists."""
    if rows is None:
        db = get_db()
        rows = _fetch_soil_sensor_rows(db)

    now = datetime.now(
        timezone.utc
    )

    candidates = []

    for row in rows:

        if row["lat"] is None or row["lon"] is None:
            continue

        try:
            recorded = datetime.fromisoformat(
                row["recorded_at"]
            )

            if (
                now - recorded
            ).total_seconds() >= SENSOR_STALE_AFTER_SECONDS:
                continue

            sensor_lat = float(
                row["lat"]
            )

            sensor_lon = float(
                row["lon"]
            )

            distance = haversine_km(
                lat,
                lon,
                sensor_lat,
                sensor_lon,
            )

            if distance > radius_km:
                continue

            value = float(
                row["value"]
            )

            unit = str(
                row["unit"] or ""
            ).lower()

            if (
                "%"
                in unit
            ):
                soil_pct = value

            elif value <= 1.0:
                soil_pct = value * 100.0

            else:
                soil_pct = value

            soil_pct = float(
                np.clip(
                    soil_pct,
                    0.0,
                    100.0,
                )
            )

            candidates.append({
                "distance_km": distance,
                "soil_moisture_pct": soil_pct,
                "sensor_id": row["sensor_id"],
                "district": row["district"],
                "recorded_at": row["recorded_at"],
            })

        except (
            TypeError,
            ValueError,
        ):
            continue

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: x["distance_km"]
    )

    # Prefer the closest live sensor.
    return candidates[0]


OPEN_METEO_CACHE_SECONDS = 600
_OPEN_METEO_CACHE = {}


def fetch_open_meteo(lat, lon):
    # /api/regional-districts calls this once per NE district (100+) on
    # every load/refresh; without caching that's 100+ sequential live HTTP
    # calls to Open-Meteo before the page can render. A 10-minute cache
    # (rain/soil data doesn't meaningfully change minute to minute) turns
    # that into a handful of real requests, with the rest served instantly.
    cache_key = (round(lat, 2), round(lon, 2))
    now_ts = datetime.now(timezone.utc).timestamp()
    cached = _OPEN_METEO_CACHE.get(cache_key)
    if cached and now_ts - cached["fetched_at"] < OPEN_METEO_CACHE_SECONDS:
        return cached["payload"]

    result = _fetch_open_meteo_uncached(lat, lon)
    if result is not None:
        _OPEN_METEO_CACHE[cache_key] = {"fetched_at": now_ts, "payload": result}
    return result


def _fetch_open_meteo_uncached(lat, lon):
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": (
                    "precipitation,"
                    "soil_moisture_0_to_1cm"
                ),
                "past_days": 3,
                "forecast_days": 1,
            },
            timeout=8,
        )

        resp.raise_for_status()

        data = resp.json()
        hourly = data.get(
            "hourly",
            {},
        )

        precip = hourly.get(
            "precipitation"
        ) or []

        soil = hourly.get(
            "soil_moisture_0_to_1cm"
        ) or []

        rain_72h = round(
            float(
                sum(
                    value
                    for value in precip
                    if value is not None
                )
            ),
            2,
        )

        soil_pct = (
            round(
                float(soil[-1]) * 100,
                1,
            )
            if soil
            and soil[-1] is not None
            else None
        )

        return {
            "source": "open_meteo",
            "rain_72h_mm": rain_72h,
            "soil_moisture_pct": soil_pct,
        }

    except (
        requests.RequestException,
        ValueError,
        KeyError,
    ):
        return None


WEATHER_CACHE_SECONDS = 300
_WEATHER_CACHE = {}


def _weather_cache_key(lat, lon):
    # Round to ~1km so nearby repeat requests (dashboard refresh, multiple
    # users looking at the same district) share a cache entry instead of
    # each paying the full IMD/Open-Meteo round trip.
    return (round(lat, 2), round(lon, 2))


RAINFALL_WATCH_CACHE_SECONDS = 300
_RAINFALL_WATCH_CACHE = {"fetched_at": 0.0, "payload": None}


def _fetch_rainfall_watch_series():
    """Return the latest 72 hours of regional rainfall in 4-hour buckets."""
    now_ts = datetime.now(timezone.utc).timestamp()
    cached = _RAINFALL_WATCH_CACHE.get("payload")
    if cached and now_ts - _RAINFALL_WATCH_CACHE["fetched_at"] < RAINFALL_WATCH_CACHE_SECONDS:
        return cached

    # Regional centroid. District risk calculations continue to use their
    # individual coordinates; this series is only for the regional chart.
    lat, lon = 25.9, 93.0
    resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "hourly": "precipitation",
            "past_days": 3,
            "forecast_days": 1,
            "timezone": "Asia/Kolkata",
        },
        timeout=10,
    )
    resp.raise_for_status()
    hourly = resp.json().get("hourly") or {}
    times = hourly.get("time") or []
    precip = hourly.get("precipitation") or []

    pairs = []
    for ts, value in zip(times, precip):
        try:
            if value is not None:
                pairs.append((str(ts), float(value)))
        except (TypeError, ValueError):
            continue

    # Exactly 18 complete four-hour bars = 72 hours.
    usable = len(pairs) - (len(pairs) % 4)
    bins = []
    for i in range(max(0, usable - 72), usable, 4):
        chunk = pairs[i:i + 4]
        if len(chunk) != 4:
            continue
        try:
            label = datetime.fromisoformat(chunk[0][0]).strftime("%d %b %H:%M")
        except ValueError:
            label = chunk[0][0]
        bins.append({
            "label": label,
            "value": round(sum(v for _, v in chunk), 2),
        })

    payload = {
        "labels": [b["label"] for b in bins],
        "values": [b["value"] for b in bins],
        "source": "Open-Meteo",
        "window_hours": 4,
        "hours": 72,
        "generated_at": now_iso(),
    }
    _RAINFALL_WATCH_CACHE["fetched_at"] = now_ts
    _RAINFALL_WATCH_CACHE["payload"] = payload
    return payload


@app.route("/api/rainfall-watch", methods=["GET"])
def rainfall_watch():
    try:
        return jsonify(_fetch_rainfall_watch_series())
    except (requests.RequestException, ValueError, KeyError) as exc:
        logger.warning("Rainfall watch feed unavailable: %s", exc)
        return jsonify({
            "error": "rainfall upstream unavailable",
            "source": None,
            "labels": [],
            "values": [],
        }), 502


@app.route("/api/weather", methods=["GET"])
def weather():
    try:
        lat = float(
            request.args.get("lat")
        )
        lon = float(
            request.args.get("lon")
        )
    except (
        TypeError,
        ValueError,
    ):
        return jsonify({
            "error":
                "lat and lon query params are required"
        }), 400

    now_ts = datetime.now(timezone.utc).timestamp()
    cache_key = _weather_cache_key(lat, lon)
    cached = _WEATHER_CACHE.get(cache_key)
    if cached and now_ts - cached["fetched_at"] < WEATHER_CACHE_SECONDS:
        return jsonify(cached["payload"])

    imd = fetch_imd_observation(
        lat,
        lon,
    )

    if imd is not None:

        sensor = get_latest_soil_sensor(
            lat,
            lon,
        )

        if sensor:
            imd["soil_moisture_pct"] = sensor[
                "soil_moisture_pct"
            ]
            imd["soil_source"] = "ground_sensor"
            imd["soil_sensor_id"] = sensor[
                "sensor_id"
            ]
            imd["soil_sensor_distance_km"] = round(
                sensor["distance_km"],
                2,
            )

        _WEATHER_CACHE[cache_key] = {"fetched_at": now_ts, "payload": imd}
        return jsonify(imd)

    fallback = fetch_open_meteo(
        lat,
        lon,
    )

    if fallback is None:
        return jsonify({
            "error":
                "weather upstream unavailable",
            "source":
                None,
        }), 502

    sensor = get_latest_soil_sensor(
        lat,
        lon,
    )

    if sensor:
        fallback["soil_moisture_pct"] = sensor[
            "soil_moisture_pct"
        ]
        fallback["soil_source"] = "ground_sensor"
        fallback["soil_sensor_id"] = sensor[
            "sensor_id"
        ]
        fallback["soil_sensor_distance_km"] = round(
            sensor["distance_km"],
            2,
        )

    else:
        fallback["soil_source"] = "open_meteo"

    fallback["imd_available"] = False
    fallback["note"] = (
        "IMD access not configured (set DATA_GOV_IN_API_KEY) — "
        "serving live Open-Meteo data; nearby live ground "
        "soil-moisture sensors override the weather soil value."
    )

    _WEATHER_CACHE[cache_key] = {"fetched_at": now_ts, "payload": fallback}
    return jsonify(fallback)


@app.route("/", methods=["GET"])
def landing_page():
    return render_template("index.html")


@app.route("/dashboard.html", methods=["GET"])
@app.route("/dashboard", methods=["GET"])
def dashboard_page():
    return render_template("dashboard.html")


@app.route("/dashboard-3d.html", methods=["GET"])
@app.route("/dashboard-3d", methods=["GET"])
def dashboard_3d_page():
    return render_template("dashboard-3d.html")


@app.route("/gis.html", methods=["GET"])
@app.route("/gis", methods=["GET"])
def gis_page():
    return render_template("gis.html")


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    error = None
    if request.method == "POST":
        if request.form.get("password", "") == ADMIN_PASSWORD:
            session["is_admin"] = True
            dest = request.form.get("next") or request.args.get("next") or url_for("admin_panel")
            return redirect(dest)
        error = "Incorrect password."
    return render_template("admin_login.html", error=error, next=request.args.get("next", ""))


@app.route("/admin/logout", methods=["GET", "POST"])
def admin_logout():
    session.pop("is_admin", None)
    return redirect(url_for("admin_login"))


@app.route("/admin", methods=["GET"])
@admin_required
def admin_panel():
    return render_template("admin.html")


TILE_SOURCES = {
    "satellite": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    "topo": "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
}


@app.route("/api/tile/<layer>/<int:z>/<int:x>/<int:y>", methods=["GET"])
def tile_proxy(layer, z, x, y):
    if layer not in TILE_SOURCES:
        return jsonify({"error": "unknown layer"}), 404
    if not (0 <= z <= 18):
        return jsonify({"error": "zoom out of range"}), 400

    if layer == "satellite":
        url = TILE_SOURCES[layer].format(z=z, x=x, y=y)
    else:
        subdomain = "abc"[(x + y) % 3]
        url = TILE_SOURCES[layer].format(s=subdomain, z=z, x=x, y=y)

    try:
        upstream = requests.get(
            url,
            headers={"User-Agent": "NER-Landslide-EWS/1.0"},
            timeout=8,
        )
        upstream.raise_for_status()
    except requests.RequestException:
        return jsonify({"error": "tile fetch failed"}), 502

    return Response(
        upstream.content,
        mimetype=upstream.headers.get("Content-Type", "image/png"),
        headers={"Cache-Control": "public, max-age=604800"},
    )


# ---------------------------------------------------------------------------
# NASA Earth Observation imagery
# ---------------------------------------------------------------------------
# The frontend used to call api.nasa.gov directly from the browser using the
# public DEMO_KEY, which is capped at 30 requests/hour and 50/day and shared
# across every app on the internet using it — so it looks broken most of the
# time even when nothing is wrong. Proxying through the backend lets us use a
# real per-app key (1,000 req/hour) via NASA_API_KEY, keeps the key out of
# the page source, and lets us tell the frontend *why* a request failed
# (rate limit vs. no imagery available for that lat/lon/date) instead of one
# generic "unavailable" message.
NASA_API_KEY = os.environ.get("NASA_API_KEY", "DEMO_KEY").strip() or "DEMO_KEY"


@app.route("/api/nasa-imagery", methods=["GET"])
@admin_required
def nasa_imagery():
    try:
        lat = float(request.args.get("lat"))
        lon = float(request.args.get("lon"))
    except (TypeError, ValueError):
        return jsonify({"error": "lat and lon query params are required"}), 400
    date = request.args.get("date", "").strip()

    params = {"lat": lat, "lon": lon, "dim": 0.20, "api_key": NASA_API_KEY}
    if date:
        params["date"] = date

    try:
        upstream = requests.get(
            "https://api.nasa.gov/planetary/earth/imagery",
            params=params,
            timeout=45,
        )
    except requests.RequestException as exc:
        logger.warning("NASA imagery upstream unreachable: %s", exc)
        return jsonify({"error": "nasa upstream unreachable", "detail": str(exc)}), 502

    # This branch was previously unreachable dead code (it sat after an
    # earlier `return`, inside the exception handler, so it could never
    # actually run) — a real 429 from NASA is a normal HTTP response, not a
    # request exception, so it has to be checked here against the response
    # that actually came back.
    if upstream.status_code == 429:
        return jsonify({
            "error": "rate_limited",
            "detail": "NASA API rate limit reached for this key.",
            "using_demo_key": NASA_API_KEY == "DEMO_KEY",
        }), 429

    content_type = upstream.headers.get("Content-Type", "")
    if upstream.status_code == 200 and "image" in content_type:
        return Response(
            upstream.content,
            mimetype=content_type,
            headers={"Cache-Control": "public, max-age=86400"},
        )

    # NASA returns a JSON body instead of an image both when there's no
    # imagery for this lat/lon/date, and on other failures (bad key, bad
    # params). Surface the real message either way instead of assuming.
    detail = None
    if "json" in content_type:
        try:
            body = upstream.json()
            detail = body.get("msg") or body.get("error")
        except ValueError:
            pass
    if upstream.status_code == 404 or (detail and "not found" in str(detail).lower()):
        return jsonify({
            "error": "no_imagery",
            "detail": detail or "No satellite imagery available for that location/date.",
        }), 404

    return jsonify({
        "error": "nasa_upstream_error",
        "status": upstream.status_code,
        "detail": detail or upstream.text[:200],
    }), 502


@app.route("/api/model/info", methods=["GET"])
def model_info():
    return jsonify(LANDSLIDE_MODEL)


@app.route("/api/predict", methods=["POST"])
def predict():
    body = request.get_json(
        force=True,
        silent=True,
    ) or {}

    try:
        rain = float(
            body.get(
                "rain_72h_mm",
                0,
            )
        )

        soil = float(
            body.get(
                "soil_moisture_pct",
                0,
            )
        )

        slope = float(
            body.get(
                "slope_deg",
                0,
            )
        )

    except (
        TypeError,
        ValueError,
    ):
        return jsonify({
            "error":
                "invalid numerical inputs"
        }), 400

    lat = body.get("lat")
    lon = body.get("lon")

    sensor = None

    if lat is not None and lon is not None:
        try:
            lat = float(lat)
            lon = float(lon)

            sensor = get_latest_soil_sensor(
                lat,
                lon,
            )

            if sensor:
                soil = sensor[
                    "soil_moisture_pct"
                ]

        except (
            TypeError,
            ValueError,
        ):
            lat = None
            lon = None

    try:
        seismic = float(
            body.get(
                "seismic_influence",
                0,
            )
        )
    except (
        TypeError,
        ValueError,
    ):
        seismic = 0.0

    if (
        lat is not None
        and lon is not None
        and "seismic_influence" not in body
    ):
        seismic = calculate_seismic_influence(
            lat,
            lon,
        )

    seismic = float(
        np.clip(
            seismic,
            0.0,
            1.0,
        )
    )

    prob = predict_probability(
        rain,
        soil,
        slope,
        seismic,
    )

    record_id = str(
        uuid.uuid4()
    )

    db = get_db()

    db.execute(
        """
        INSERT INTO predictions (
            id,
            district,
            rain_72h_mm,
            soil_moisture_pct,
            slope_deg,
            seismic_influence,
            probability,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record_id,
            body.get(
                "district",
                "unknown",
            ),
            rain,
            soil,
            slope,
            seismic,
            prob,
            now_iso(),
        ),
    )

    db.commit()

    return jsonify({
        "id": record_id,
        "probability": prob,
        "hazard_index": round(
            prob * 100
        ),
        "seismic_influence": seismic,
        "soil_moisture_source": (
            "ground_sensor"
            if sensor
            else "request/weather_value"
        ),
        "soil_sensor_id": (
            sensor["sensor_id"]
            if sensor
            else None
        ),
        "model_metrics":
            LANDSLIDE_MODEL["metrics"],
    })


@app.route("/api/predict/grid", methods=["POST"])
def predict_grid():
    body = request.get_json(
        force=True,
        silent=True,
    ) or {}

    points = body.get(
        "points",
        [],
    )

    if (
        not isinstance(points, list)
        or not points
    ):
        return jsonify({
            "error":
                "points must be a non-empty list"
        }), 400

    # Fetch the USGS feed and the soil-sensor rows once for the whole grid,
    # instead of re-querying the database for every single grid point.
    seismic_events = fetch_usgs_earthquakes()
    soil_sensor_rows = _fetch_soil_sensor_rows(get_db())

    rows = []
    coords = []
    enriched = []

    for p in points:

        try:
            rain = float(
                p.get(
                    "rain_72h_mm",
                    0,
                )
            )

            soil = float(
                p.get(
                    "soil_moisture_pct",
                    0,
                )
            )

            slope = float(
                p.get(
                    "slope_deg",
                    0,
                )
            )

            lat = float(
                p.get("lat")
            )

            lon = float(
                p.get("lon")
            )

        except (
            TypeError,
            ValueError,
        ):
            return jsonify({
                "error":
                    "malformed point data"
            }), 400

        sensor = get_latest_soil_sensor(
            lat,
            lon,
            rows=soil_sensor_rows,
        )

        if sensor:
            soil = sensor[
                "soil_moisture_pct"
            ]

        if "seismic_influence" in p:
            try:
                seismic = float(
                    p["seismic_influence"]
                )
            except (
                TypeError,
                ValueError,
            ):
                seismic = calculate_seismic_influence(
                    lat,
                    lon,
                    seismic_events,
                )
        else:
            seismic = calculate_seismic_influence(
                lat,
                lon,
                seismic_events,
            )

        seismic = float(
            np.clip(
                seismic,
                0.0,
                1.0,
            )
        )

        rows.append([
            rain,
            soil,
            slope,
            seismic,
        ])

        coords.append(
            (lat, lon)
        )

        enriched.append({
            "soil": soil,
            "seismic_influence":
                seismic,
            "sensor_id": (
                sensor["sensor_id"]
                if sensor
                else None
            ),
        })

    probs = predict_probabilities_batch(
        rows
    )

    results = []

    for (
        (lat, lon),
        prob,
        metadata,
    ) in zip(
        coords,
        probs,
        enriched,
    ):

        results.append({
            "lat": lat,
            "lon": lon,
            "probability": prob,
            "hazard_index": round(
                prob * 100
            ),
            "soil_moisture_pct":
                metadata["soil"],
            "seismic_influence":
                metadata["seismic_influence"],
            "soil_sensor_id":
                metadata["sensor_id"],
        })

    return jsonify({
        "results": results,
        "model_features":
            LANDSLIDE_MODEL["features"],
        "seismic_events_used":
            len(seismic_events),
    })


# ---------------------------------------------------------------------------
# Seismic activity endpoint
# ---------------------------------------------------------------------------

@app.route("/api/seismic", methods=["GET"])
def seismic():
    try:
        lat = float(
            request.args.get(
                "lat",
                25.5,
            )
        )

        lon = float(
            request.args.get(
                "lon",
                92.5,
            )
        )

    except (
        TypeError,
        ValueError,
    ):
        return jsonify({
            "error":
                "lat and lon must be numeric"
        }), 400

    try:
        radius_km = float(
            request.args.get(
                "radius_km",
                300,
            )
        )

        radius_km = max(
            1.0,
            min(
                radius_km,
                1000.0,
            ),
        )

    except (
        TypeError,
        ValueError,
    ):
        radius_km = 300.0

    events = fetch_usgs_earthquakes()

    nearby = []

    for event in events:

        distance = haversine_km(
            lat,
            lon,
            event["lat"],
            event["lon"],
        )

        if distance <= radius_km:

            item = dict(
                event
            )

            item["distance_km"] = round(
                distance,
                1,
            )

            nearby.append(
                item
            )

    nearby.sort(
        key=lambda e: (
            e["distance_km"],
            -e["magnitude"],
        )
    )

    influence = calculate_seismic_influence(
        lat,
        lon,
        events,
    )

    return jsonify({
        "source": "USGS",
        "location": {
            "lat": lat,
            "lon": lon,
        },
        "radius_km": radius_km,
        "events_last_7_days":
            len(events),
        "nearby_events":
            len(nearby),
        "seismic_influence":
            influence,
        "events":
            nearby[:100],
        "note": (
            "Seismic influence is a normalized "
            "distance/recency/magnitude heuristic "
            "used as an AI feature; it is not a "
            "seismological warning threshold."
        ),
    })


# ---------------------------------------------------------------------------
# Historical landslide data + location intelligence
# ---------------------------------------------------------------------------
# Primary source: NASA COOLR (Cooperative Open Online Landslide Repository).
# The service is a public ArcGIS REST feature layer maintained by NASA.
# A local CSV/JSON file can still be supplied through HISTORICAL_LANDSLIDE_DATA
# and is preferred when present. Otherwise NASA COOLR is queried for the NER.

NASA_COOLR_QUERY_URL = (
    "https://maps.nccs.nasa.gov/mapping/rest/services/"
    "COOLR/COOLR_Reports_Point/MapServer/0/query"
)

# Official NASA Open Data fallback. This is a real historical catalog, not
# generated data. NASA publishes the CSV as a public resource.
NASA_GLC_CSV_URL = (
    "https://data.nasa.gov/docs/legacy/Global_Landslide_Catalog_Export/"
    "Global_Landslide_Catalog_Export_rows.csv"
)
NASA_GLC_JSON_URL = (
    "https://data.nasa.gov/docs/legacy/Global_Landslide_Catalog_Export/"
    "Global_Landslide_Catalog_Export_rows.json"
)

NER_BOUNDS = {
    "lon_min": 87.0,
    "lat_min": 20.0,
    "lon_max": 98.0,
    "lat_max": 29.5,
}

NER_STATES = {
    "assam", "arunachal pradesh", "meghalaya", "manipur",
    "mizoram", "nagaland", "tripura", "sikkim",
}

HISTORICAL_DATA_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "historical_landslides.csv"),
    os.path.join(os.path.dirname(__file__), "historical_landslides.json"),
    os.path.join(os.path.dirname(__file__), "historical-landslides.csv"),
    os.path.join(os.path.dirname(__file__), "historical-landslides.json"),
    os.path.join(os.path.dirname(__file__), "data", "historical_landslides.csv"),
    os.path.join(os.path.dirname(__file__), "data", "historical_landslides.json"),
    os.path.join(os.path.dirname(__file__), "data", "historical-landslides.csv"),
    os.path.join(os.path.dirname(__file__), "data", "historical-landslides.json"),
]

HISTORICAL_CACHE_SECONDS = 300
_HISTORICAL_CACHE = {"loaded_at": 0.0, "source": None, "events": []}
_HISTORICAL_REMOTE_LOCK = threading.Lock()
_HISTORICAL_REMOTE_STATE = {"loading": False, "ready": False, "error": None, "started_at": 0.0}

def _merge_historical_events(*groups):
    """Merge historical inventories without duplicating the same event."""
    merged = []
    seen = set()
    for group in groups:
        for event in group or []:
            lat = event.get("lat")
            lon = event.get("lon")
            key = str(event.get("id") or event.get("event_id") or "").strip().lower()
            if not key:
                key = f"{event.get('event_date','')}|{event.get('locality','')}|{lat}|{lon}"
            if key in seen:
                continue
            seen.add(key)
            merged.append(event)
    return merged

def _refresh_remote_historical_in_background():
    """Fetch *all* available NASA historical records progressively.

    COOLR is paginated by ArcGIS.  We publish each page into the in-memory
    cache as soon as it arrives, so the browser can render thousands of
    points progressively instead of waiting for one giant request.  The NASA
    Global Landslide Catalog is then merged as a second, broader fallback
    inventory.  There is intentionally no 20,000-record ceiling here.
    """
    with _HISTORICAL_REMOTE_LOCK:
        if _HISTORICAL_REMOTE_STATE["loading"]:
            return
        _HISTORICAL_REMOTE_STATE.update({"loading": True, "ready": False, "error": None, "started_at": datetime.now(timezone.utc).timestamp()})

    def publish(events, source):
        merged = _merge_historical_events(_HISTORICAL_CACHE.get("events", []), events)
        _HISTORICAL_CACHE["loaded_at"] = datetime.now(timezone.utc).timestamp()
        _HISTORICAL_CACHE["source"] = source
        _HISTORICAL_CACHE["events"] = merged
        logger.info("Historical inventory progress: %d events", len(merged))

    def worker():
        try:
            local_events, local_source = _load_local_historical_landslides()
            publish(local_events, local_source or "Historical inventory")

            # First source: NASA COOLR, page-by-page.
            coolr = _fetch_nasa_coolr_historical(progress_callback=lambda page: publish(page, "Local inventory + NASA COOLR" if local_events else "NASA COOLR"))
            publish(coolr, "Local inventory + NASA COOLR" if local_events else "NASA COOLR")

            # Second source: official NASA Global Landslide Catalog.  It is
            # useful when COOLR has sparse coverage in a particular region.
            try:
                glc = _fetch_nasa_glc_historical()
                # Publish in chunks so the client visibly receives progress.
                for start in range(0, len(glc), 500):
                    publish(glc[start:start + 500], "Local + NASA COOLR + NASA Global Landslide Catalog")
            except Exception as glc_exc:
                logger.warning("NASA GLC enrichment failed: %s", glc_exc)

            _HISTORICAL_REMOTE_STATE.update({"loading": False, "ready": True, "error": None})
            logger.info("Background historical inventory complete: %d events", len(_HISTORICAL_CACHE.get("events", [])))
        except Exception as exc:
            logger.warning("Background NASA historical refresh failed: %s", exc)
            _HISTORICAL_REMOTE_STATE.update({"loading": False, "ready": False, "error": str(exc)})

    threading.Thread(target=worker, name="historical-inventory-refresh", daemon=True).start()

def _start_historical_background_refresh():
    """Start remote enrichment after the first fast local response.

    Only skip starting a new refresh while one is actively in flight.
    Once a refresh has completed (``ready``), a later stale-cache reload
    should be allowed to kick off a fresh enrichment pass rather than being
    locked out forever.
    """
    if _HISTORICAL_REMOTE_STATE["loading"]:
        return
    _refresh_remote_historical_in_background()


def _first_value(record, names, default=None):
    lower = {str(k).strip().lower(): v for k, v in record.items()}
    for name in names:
        value = lower.get(name.lower())
        if value is not None and str(value).strip() != "":
            return value
    return default


def _to_float(value):
    if value is None or value == "":
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _to_int(value):
    number = _to_float(value)
    return int(number) if number is not None else 0


def _derive_historical_severity(raw):
    """Derive a display severity only when COOLR has no explicit severity field."""
    explicit = _first_value(raw, ["severity", "severity_level", "level", "impact", "risk"])
    if explicit:
        return str(explicit).title()

    fatalities = _to_int(_first_value(raw, ["fatality_count", "fatalities", "deaths", "death"], 0))
    size = str(_first_value(raw, ["landslide_size", "size"], "")).lower()

    if fatalities > 0:
        return "High"
    if any(word in size for word in ("large", "medium", "huge")):
        return "Moderate"
    if "small" in size:
        return "Low"
    return "Unknown"


def _derive_road_affected(raw):
    explicit = _first_value(raw, ["road_affected", "road_impact", "road_damage"])
    if explicit is not None:
        value = str(explicit).strip().lower()
        return 1 if value in {"1", "true", "yes", "y"} else 0

    text = " ".join(str(_first_value(raw, [name], "")) for name in (
        "event_title", "event_description", "location_description", "comments"
    )).lower()
    return 1 if any(word in text for word in (
        "road", "highway", "roadway", "blocked", "route", "nh-", "nh "
    )) else 0


def _normalise_historical_event(raw, index=0, source_default="Historical dataset"):
    lat = _to_float(_first_value(raw, [
        "lat", "latitude", "y", "lat_dd", "latitude_dd"
    ]))
    lon = _to_float(_first_value(raw, [
        "lon", "lng", "longitude", "x", "lon_dd", "longitude_dd"
    ]))

    district = _first_value(raw, [
        "district", "district_name", "location", "place", "city", "area",
        "gazetteer_closest_point", "admin_division_name"
    ], "Unknown")
    state = _first_value(raw, ["state", "state_name", "admin_division_name"], "")
    locality = _first_value(raw, [
        "locality", "location_description", "event_title", "place", "city",
        "gazetteer_closest_point"
    ], district)
    event_date = _first_value(raw, [
        "date", "event_date", "event_date_utc", "datetime", "timestamp", "year"
    ])
    severity = _derive_historical_severity(raw)
    fatalities = _to_int(_first_value(raw, [
        "fatalities", "fatality_count", "deaths", "death", "fatality", "casualties"
    ], 0))
    injuries = _to_int(_first_value(raw, [
        "injuries", "injury_count", "injured", "injury"
    ], 0))
    rainfall = _to_float(_first_value(raw, [
        "rainfall", "rainfall_mm", "rain_24h_mm", "rain_72h_mm"
    ]))
    trigger = _first_value(raw, [
        "trigger", "landslide_trigger", "cause"
    ], "Not recorded")
    area = _first_value(raw, [
        "landslide_type", "landslide_category", "type", "hazard_type", "event_type"
    ], "landslide")
    description = _first_value(raw, [
        "description", "event_description", "remarks", "notes", "details", "comment", "comments"
    ], "")
    source_url = _first_value(raw, ["source_url", "source_link", "reference_url", "url"], "")
    source_org = _first_value(raw, ["source_org", "source_name", "source", "data_source", "reference", "agency"], source_default)
    event_id = _first_value(raw, ["id", "event_id", "landslide_id", "objectid"], f"hist-{index + 1}")

    return {
        "id": str(event_id),
        "lat": lat,
        "lon": lon,
        "locality": str(locality) if locality is not None else "Recorded event",
        "district": str(district) if district is not None else "Unknown",
        "state": str(state) if state is not None else "",
        "event_date": str(event_date) if event_date is not None else None,
        "date": str(event_date) if event_date is not None else None,
        "severity": severity,
        "trigger": str(trigger) if trigger is not None else "Not recorded",
        "fatalities": fatalities,
        "injuries": injuries,
        "rainfall_mm": rainfall,
        "landslide_type": str(area) if area is not None else "landslide",
        "description": str(description),
        "source_org": str(source_org),
        "source_url": str(source_url) if source_url else None,
        "road_affected": _derive_road_affected(raw),
    }


def _inside_ner(event):
    lat = event.get("lat")
    lon = event.get("lon")
    return (
        lat is not None and lon is not None and
        NER_BOUNDS["lat_min"] <= lat <= NER_BOUNDS["lat_max"] and
        NER_BOUNDS["lon_min"] <= lon <= NER_BOUNDS["lon_max"]
    )


def _load_local_historical_landslides():
    import csv
    import json

    configured = os.environ.get("HISTORICAL_LANDSLIDE_DATA", "").strip()
    candidates = ([configured] if configured else []) + HISTORICAL_DATA_CANDIDATES
    seen = set()

    for path in candidates:
        if not path or path in seen or not os.path.isfile(path):
            continue
        seen.add(path)
        try:
            if path.lower().endswith(".csv"):
                with open(path, "r", encoding="utf-8-sig", newline="") as handle:
                    rows = list(csv.DictReader(handle))
            else:
                with open(path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if isinstance(payload, dict):
                    payload = payload.get("events", payload.get("records", payload.get("data", [])))
                if not isinstance(payload, list):
                    raise ValueError("historical JSON must contain a list or events/records/data list")
                rows = [row for row in payload if isinstance(row, dict)]

            events = [
                _normalise_historical_event(row, i, os.path.basename(path))
                for i, row in enumerate(rows)
            ]
            return [event for event in events if _inside_ner(event)], os.path.basename(path)
        except (OSError, ValueError, csv.Error) as exc:
            logger.warning("Historical landslide file could not be loaded (%s): %s", path, exc)

    return [], None


def _fetch_nasa_coolr_historical(progress_callback=None):
    """Fetch every NASA COOLR point inside the NER bounding box.

    ``progress_callback`` receives each page immediately after it is decoded.
    This makes the server-side cache grow progressively while the dashboard
    stays responsive.
    """
    fields = (
        "objectid,source_name,source_link,event_id,event_date,event_title,"
        "event_description,location_description,landslide_category,landslide_trigger,"
        "landslide_size,fatality_count,injury_count,comments,event_import_source,"
        "latitude,longitude,country_name,country_code,admin_division_name,"
        "gazetteer_closest_point"
    )

    all_features = []
    offset = 0

    while True:
        params = {
            "where": "1=1",
            "geometry": f"{NER_BOUNDS['lon_min']},{NER_BOUNDS['lat_min']},{NER_BOUNDS['lon_max']},{NER_BOUNDS['lat_max']}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": fields,
            "returnGeometry": "true",
            "outSR": "4326",
            "resultOffset": offset,
            "resultRecordCount": 2000,
            "f": "json",
        }

        response = requests.get(
            NASA_COOLR_QUERY_URL,
            params=params,
            timeout=25,
            headers={"User-Agent": "BHUDRISTI-NER-EWS/1.0"},
        )
        response.raise_for_status()
        payload = response.json()

        if payload.get("error"):
            raise RuntimeError(payload["error"].get("message", "NASA COOLR query failed"))

        features = payload.get("features", [])
        all_features.extend(features)
        if progress_callback and features:
            page_events = []
            for page_index, feature in enumerate(features):
                attrs = feature.get("attributes") or {}
                geometry = feature.get("geometry") or {}
                raw = dict(attrs)
                if raw.get("latitude") is None:
                    raw["latitude"] = geometry.get("y")
                if raw.get("longitude") is None:
                    raw["longitude"] = geometry.get("x")
                event = _normalise_historical_event(raw, offset + page_index, "NASA COOLR")
                country = str(attrs.get("country_code") or attrs.get("country_name") or "").strip().lower()
                if country and country not in {"in", "ind", "india"}:
                    continue
                if _inside_ner(event):
                    page_events.append(event)
            progress_callback(page_events)

        # ArcGIS can expose a transfer-limit flag even when the returned page
        # is shorter than requested. Keep following the server's pagination
        # until it explicitly says there are no more records.
        if not payload.get("exceededTransferLimit") or not features:
            break
        offset += len(features)

    events = []
    for index, feature in enumerate(all_features):
        attrs = feature.get("attributes") or {}
        geometry = feature.get("geometry") or {}
        raw = dict(attrs)
        if raw.get("latitude") is None:
            raw["latitude"] = geometry.get("y")
        if raw.get("longitude") is None:
            raw["longitude"] = geometry.get("x")

        event = _normalise_historical_event(raw, index, "NASA COOLR")
        country = str(attrs.get("country_code") or attrs.get("country_name") or "").strip().lower()
        admin = str(attrs.get("admin_division_name") or "").strip().lower()

        # Prefer India records. If NASA leaves the country/admin field blank,
        # keep the point when its coordinates are safely inside the NER bounds.
        if country and country not in {"in", "ind", "india"}:
            continue
        if admin and admin not in NER_STATES:
            # NASA's admin field is not consistently populated/named, so do
            # not discard an otherwise valid India point solely on this field.
            pass
        if _inside_ner(event):
            events.append(event)

    return events


def _fetch_nasa_glc_historical():
    """Fetch NASA's official Global Landslide Catalog export as a fallback.

    COOLR is preferred because it is the newer/live repository. The GLC export
    is used when COOLR is unavailable, then filtered to the NER by coordinates.
    """
    last_error = None

    for url, is_json in ((NASA_GLC_CSV_URL, False), (NASA_GLC_JSON_URL, True)):
        try:
            response = requests.get(
                url,
                timeout=45,
                headers={"User-Agent": "BHUDRISTI-NER-EWS/1.0"},
            )
            response.raise_for_status()

            if is_json:
                payload = response.json()
                if isinstance(payload, dict):
                    payload = payload.get("data", payload.get("rows", payload.get("events", [])))
                rows = [row for row in payload if isinstance(row, dict)] if isinstance(payload, list) else []
            else:
                text = response.content.decode("utf-8-sig", errors="replace")
                rows = list(csv.DictReader(io.StringIO(text)))

            events = []
            for index, row in enumerate(rows):
                event = _normalise_historical_event(
                    row, index, "NASA Global Landslide Catalog"
                )
                if _inside_ner(event):
                    events.append(event)

            if events:
                logger.info(
                    "Loaded %d NER historical landslides from NASA GLC fallback",
                    len(events),
                )
                return events

            logger.warning("NASA GLC returned %d rows but none were inside NER bounds", len(rows))
            return []

        except Exception as exc:
            last_error = exc
            logger.warning("NASA GLC fallback failed (%s): %s", url, exc)

    raise RuntimeError(f"NASA GLC fallback unavailable: {last_error}")


def _load_historical_landslides(force=False):
    now_ts = datetime.now(timezone.utc).timestamp()
    if (
        not force
        and _HISTORICAL_CACHE["source"] is not None
        and now_ts - _HISTORICAL_CACHE["loaded_at"] < HISTORICAL_CACHE_SECONDS
    ):
        return _HISTORICAL_CACHE["events"], _HISTORICAL_CACHE["source"]

    # Once we already have a populated cache (e.g. enriched by the NASA
    # background refresh), a stale cache-timer should never wipe it back
    # down to just the local file. Re-loading local data is only meaningful
    # the very first time, before anything has been cached yet.
    if _HISTORICAL_CACHE["source"] is not None and _HISTORICAL_CACHE["events"]:
        _HISTORICAL_CACHE["loaded_at"] = now_ts
        _start_historical_background_refresh()
        return _HISTORICAL_CACHE["events"], _HISTORICAL_CACHE["source"]

    local_events, local_source = _load_local_historical_landslides()
    if local_source:
        # Fast path: never make the dashboard wait for NASA. Enrich the local
        # inventory in a daemon thread and let subsequent requests see it.
        _HISTORICAL_CACHE["loaded_at"] = now_ts
        _HISTORICAL_CACHE["source"] = local_source
        _HISTORICAL_CACHE["events"] = local_events
        _start_historical_background_refresh()
        return local_events, local_source

    try:
        events = _fetch_nasa_coolr_historical()
        source = "NASA COOLR"
        logger.info("Loaded %d historical landslide events from NASA COOLR", len(events))
    except Exception as exc:
        logger.warning("NASA COOLR historical feed unavailable: %s", exc)
        try:
            events = _fetch_nasa_glc_historical()
            source = "NASA Global Landslide Catalog"
        except Exception as glc_exc:
            logger.warning("NASA GLC historical fallback unavailable: %s", glc_exc)
            events, source = [], "Historical sources unavailable"

    _HISTORICAL_CACHE["loaded_at"] = now_ts
    _HISTORICAL_CACHE["source"] = source
    _HISTORICAL_CACHE["events"] = events
    return events, source


def _historical_event_distance(event, lat, lon):
    if event.get("lat") is None or event.get("lon") is None:
        return None
    return haversine_km(lat, lon, event["lat"], event["lon"])


@app.route("/api/historical-status", methods=["GET"])
def historical_status():
    """Diagnostic endpoint for the historical data pipeline."""
    force = request.args.get("refresh", "0").lower() in {"1", "true", "yes"}
    events, source = _load_historical_landslides(force=force)
    return jsonify({
        "status": "ok" if events else "empty",
        "source": source,
        "count": len(events),
        "ner_bounds": NER_BOUNDS,
        "coolr_url": NASA_COOLR_QUERY_URL,
        "glc_url": NASA_GLC_CSV_URL,
        "message": (
            "Real historical records loaded." if events else
            "No historical records could be loaded from local data, NASA COOLR, or NASA GLC."
        ),
    })


@app.route("/api/historical-landslides", methods=["GET"])
def historical_landslides():
    """Return historical landslide events, paginated.

    The dashboard walks through this endpoint page by page using `offset`
    and `limit`, stopping only once `has_more` comes back false. Earlier
    this endpoint ignored `offset` entirely and always returned a bare
    array of the closest N events with no `source`/`has_more`/`next_offset`
    fields — so the frontend's pagination loop treated every response as
    "no more pages" after the very first fetch, silently capping the map
    at whatever fit in the first page.
    """
    events, source = _load_historical_landslides()

    try:
        lat_arg = request.args.get("lat")
        lon_arg = request.args.get("lon")
        lat = float(lat_arg) if lat_arg not in (None, "") else None
        lon = float(lon_arg) if lon_arg not in (None, "") else None
    except (TypeError, ValueError):
        return jsonify({"error": "lat and lon must be numeric"}), 400

    try:
        radius_km = max(1.0, min(float(request.args.get("radius_km", 100)), 1000.0))
    except (TypeError, ValueError):
        radius_km = 100.0

    filtered = []
    for event in events:
        distance = _historical_event_distance(event, lat, lon) if lat is not None and lon is not None else None
        if distance is not None and distance > radius_km:
            continue
        item = dict(event)
        item["distance_km"] = round(distance, 2) if distance is not None else None
        filtered.append(item)

    filtered.sort(key=lambda item: (
        item["distance_km"] is None,
        item["distance_km"] if item["distance_km"] is not None else 0,
        str(item.get("event_date") or ""),
    ))

    try:
        limit = max(1, min(int(request.args.get("limit", 500)), 2000))
    except (TypeError, ValueError):
        limit = 500

    try:
        offset = max(0, int(request.args.get("offset", 0)))
    except (TypeError, ValueError):
        offset = 0

    total = len(filtered)
    page = filtered[offset:offset + limit]
    next_offset = offset + len(page)
    has_more = next_offset < total

    return jsonify({
        "events": page,
        "source": source,
        "total": total,
        "offset": offset,
        "limit": limit,
        "next_offset": next_offset,
        "has_more": has_more,
        "background_loading": bool(_HISTORICAL_REMOTE_STATE["loading"]),
        "background_ready": bool(_HISTORICAL_REMOTE_STATE["ready"]),
        "background_error": _HISTORICAL_REMOTE_STATE["error"],
    })


@app.route("/api/location-intelligence", methods=["GET"])
def location_intelligence():
    """Combine live weather, seismic activity, model prediction and history."""
    try:
        lat = float(request.args.get("lat"))
        lon = float(request.args.get("lon"))
    except (TypeError, ValueError):
        return jsonify({"error": "lat and lon query params are required"}), 400

    district = request.args.get("district", "").strip()

    weather_data = fetch_open_meteo(lat, lon) or {
        "source": None,
        "rain_72h_mm": None,
        "soil_moisture_pct": None,
    }

    rain = _to_float(weather_data.get("rain_72h_mm"))
    soil = _to_float(weather_data.get("soil_moisture_pct"))
    rain_for_model = rain if rain is not None else 0.0
    soil_for_model = soil if soil is not None else 45.0

    try:
        slope = float(request.args.get("slope_deg", 30))
    except (TypeError, ValueError):
        slope = 30.0
    slope = max(0.0, min(slope, 90.0))

    seismic_events = fetch_usgs_earthquakes()
    seismic_influence = calculate_seismic_influence(lat, lon, seismic_events)
    probability = predict_probability(rain_for_model, soil_for_model, slope, seismic_influence)

    historical_events, historical_source = _load_historical_landslides()
    nearby_history = []
    for event in historical_events:
        distance = _historical_event_distance(event, lat, lon)
        if distance is None or distance > 100.0:
            continue
        item = dict(event)
        item["distance_km"] = round(distance, 2)
        nearby_history.append(item)
    nearby_history.sort(key=lambda item: item["distance_km"])

    within_25 = [e for e in nearby_history if e["distance_km"] <= 25.0]
    selected_history = within_25 if within_25 else nearby_history[:1]
    fallback_used = not bool(within_25) and bool(selected_history)

    fatalities = sum(int(e.get("fatalities") or 0) for e in within_25)
    road_events = sum(int(e.get("road_affected") or 0) for e in within_25)

    latest = None
    if selected_history:
        latest = sorted(
            selected_history,
            key=lambda e: str(e.get("event_date") or e.get("date") or ""),
            reverse=True,
        )[0]

    if probability >= 0.70:
        risk_level = "high"
    elif probability >= 0.40:
        risk_level = "moderate"
    else:
        risk_level = "low"

    return jsonify({
        "location": {"lat": lat, "lon": lon, "district": district or None},
        "risk": {
            "probability": probability,
            "hazard_index": round(probability * 100),
            "level": risk_level,
        },
        "model_inputs": {
            "rain_72h_mm": rain,
            "soil_moisture_pct": soil,
            "slope_deg": slope,
            "seismic_influence": seismic_influence,
        },
        "weather": weather_data,
        "seismic": {
            "source": "USGS",
            "nearby_events": sum(
                1 for event in seismic_events
                if haversine_km(lat, lon, event["lat"], event["lon"]) <= 300
            ),
            "influence": seismic_influence,
        },
        "historical": {
            "source": historical_source,
            "count": len(within_25),
            "nearby_events_25km": len(within_25),
            "nearby_events_100km": len(nearby_history),
            "fatalities": fatalities,
            "road_events": road_events,
            "fallback_used": fallback_used,
            "nearest": nearby_history[0] if nearby_history else None,
            "latest": latest,
            "events": selected_history[:100],
        },
        "reports": {"count": 0},
        "model_features": LANDSLIDE_MODEL["features"],
        "note": "Historical events are sourced from NASA COOLR, the official NASA Global Landslide Catalog fallback, or the configured local historical dataset.",
    })


# ---------------------------------------------------------------------------
# Road connectivity / disruption status
# ---------------------------------------------------------------------------

@app.route("/api/road-status", methods=["GET"])
def road_status():
    db = get_db()

    rows = db.execute(
        """
        SELECT *
        FROM reports
        WHERE created_at >= datetime('now', '-24 hours')
          AND (
              lower(COALESCE(hazard_type, ''))
              LIKE '%road%'
              OR
              lower(COALESCE(remarks, ''))
              LIKE '%road%'
              OR
              lower(COALESCE(remarks, ''))
              LIKE '%blocked%'
          )
        ORDER BY created_at DESC
        LIMIT 200
        """
    ).fetchall()

    blocked = 0
    restricted = 0
    recent = []

    for row in rows:

        text_blob = (
            f"{row['hazard_type'] or ''} "
            f"{row['remarks'] or ''}"
        ).lower()

        severity = int(
            row["severity"] or 0
        )

        if (
            "blocked" in text_blob
            or severity >= 4
        ):
            blocked += 1

        elif (
            "restrict" in text_blob
            or "damaged" in text_blob
        ):
            restricted += 1

        recent.append({
            "district":
                row["district"],
            "lat":
                row["lat"],
            "lon":
                row["lon"],
            "severity":
                severity,
            "hazard_type":
                row["hazard_type"],
            "remarks":
                row["remarks"],
            "created_at":
                row["created_at"],
        })

    if blocked:
        status = "blocked_reported"

    elif restricted:
        status = "restricted_reported"

    else:
        status = "no_recent_disruption_reported"

    return jsonify({
        "status": status,
        "blocked_reports_24h": blocked,
        "restricted_reports_24h": restricted,
        "reports_24h": len(rows),
        "latest_reports": recent[:50],
        "note": (
            "Road connectivity is based on actual field "
            "reports received by Pravah. No report is not "
            "treated as proof that a road is open."
        ),
    })


@app.route("/api/verification-status", methods=["GET"])
def verification_status():
    """Small diagnostic endpoint for the field-report photo verifier."""
    configured = bool(GEMINI_API_KEY) and not GEMINI_API_KEY.startswith("PASTE_YOUR_")
    if not configured:
        return jsonify({"configured": False, "model": GEMINI_MODEL, "message": "GEMINI_API_KEY is missing."}), 503
    # Do not expose the key. A lightweight text request confirms that the key
    # and selected model can actually be reached from the running server.
    try:
        text = _gemini_request(
            GEMINI_MODEL,
            [{"parts": [{"text": "Reply with the single word OK."}]}],
            {"responseMimeType": "text/plain", **({} if GEMINI_MODEL.startswith("gemini-3") else {"temperature": 0})},
        )
        return jsonify({"configured": True, "reachable": True, "model": GEMINI_MODEL, "response": text[:40]})
    except Exception as exc:
        logger.warning("Gemini verification status check failed: %s", exc)
        return jsonify({"configured": True, "reachable": False, "model": GEMINI_MODEL, "error": str(exc)}), 503


@app.route("/api/reports", methods=["GET"])
def list_reports():
    """List field reports, most recent first.

    Public callers (dashboard, GIS map — anyone without an admin session)
    only ever see a report's media once an admin has approved it: reports
    with no media attachment always show, but a report with a photo/video
    stays invisible here while its verification_status is 'needs_review'
    or 'flagged' — i.e. while it's sitting in the admin queue. This is the
    single choke point that keeps unapproved images off the public map, so
    it applies regardless of the `?status=` filter below.

    Logged-in admins (checked via the same session used by /admin) bypass
    that restriction and can filter by verification_status (auto_approved /
    needs_review / flagged) to pull just the reports that need a look."""
    db = get_db()
    is_admin = bool(session.get("is_admin"))
    status = request.args.get("status", "").strip()

    if is_admin and status:
        rows = db.execute(
            "SELECT * FROM reports WHERE verification_status = ? "
            "ORDER BY created_at DESC LIMIT 200",
            (status,),
        ).fetchall()
    elif is_admin:
        # Rejected reports are never exposed to the public field-report list; ?status=rejected
        # still surfaces them for an audit trail.
        rows = db.execute(
            "SELECT * FROM reports WHERE verification_status IS NULL "
            "OR verification_status != 'rejected' "
            "ORDER BY created_at DESC LIMIT 200"
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT * FROM reports WHERE media_filename IS NULL "
            "OR verification_status = 'verified' "
            "ORDER BY created_at DESC LIMIT 200"
        ).fetchall()

    out = []
    for r in rows:
        d = dict(r)
        if d.get("verification_flags"):
            try:
                d["verification_flags"] = json.loads(d["verification_flags"])
            except (TypeError, ValueError):
                pass
        out.append(d)
    return jsonify(out)


def _insert_report(db, body, media_filename=None, media_type=None, media_path=None):
    """Insert one field report. Dedupes on client_id (the id the mobile/web
    client generates offline) so replaying a synced report is a no-op rather
    than a duplicate row. If a media file was attached, it's run through the
    verification pipeline (see verify_report_media above) before the row is
    written. Returns (report_id, was_new)."""
    client_id = body.get("client_id") or str(uuid.uuid4())
    existing = db.execute(
        "SELECT id FROM reports WHERE client_id = ?", (client_id,)
    ).fetchone()
    if existing is not None:
        return existing["id"], False

    report_id = str(uuid.uuid4())
    created_at = body.get("created_at") or now_iso()

    verification = {"status": "no_media", "flags": []}
    media_sha256 = None
    if media_filename and media_path:
        if media_type == "video":
            media_sha256 = hash_file(media_path)
            dup_row = db.execute(
                "SELECT id FROM reports WHERE media_sha256 = ? AND media_sha256 IS NOT NULL LIMIT 1",
                (media_sha256,),
            ).fetchone()
            verification = {
                "status": "flagged" if dup_row else "needs_review",
                "flags": (["duplicate_media"] if dup_row else []) + ["video_not_deep_inspected"],
                "duplicate_of_report_id": dup_row["id"] if dup_row else None,
            }
        else:
            try:
                verification = verify_image_against_live_data(
                    db, media_path, body.get("lat"), body.get("lon"),
                    body.get("hazard_type"), created_at, report_id=report_id,
                    client_captured_at=body.get("client_captured_at"),
                    client_gps_lat=body.get("client_gps_lat"),
                    client_gps_lon=body.get("client_gps_lon"),
                )
            except Exception as exc:  # noqa: BLE001 - never let verification crash a submission
                logger.warning("Media verification failed for %s: %s", media_filename, exc)
                verification = {"status": "needs_review", "flags": ["verification_error"]}

            if verification.get("status") == "rejected":
                try:
                    os.remove(media_path)
                except OSError as exc:
                    logger.warning("Could not delete rejected media %s: %s", media_path, exc)
                raise ValueError(json.dumps({
                    "message": "Image verification failed. The report was NOT added to Field Reports.",
                    "verification": verification,
                }))

    db.execute(
        "INSERT INTO reports (id, client_id, lat, lon, district, hazard_type, severity, remarks, "
        "media_filename, media_type, created_at, synced_at, verification_status, verification_flags, "
        "media_lat, media_lon, media_captured_at, media_phash, media_sha256, distance_km, "
        "live_rain_72h_mm, live_soil_moisture_pct, duplicate_of_report_id, "
        "visual_label, visual_confidence, visual_summary, visual_matched_claim, live_verification_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            report_id, client_id, body.get("lat"), body.get("lon"), body.get("district"),
            body.get("hazard_type"), int(body.get("severity", 3) or 3), body.get("remarks", ""),
            media_filename, media_type,
            created_at, now_iso(),
            verification.get("status"), json.dumps(verification.get("flags", [])),
            verification.get("media_lat"), verification.get("media_lon"),
            verification.get("media_captured_at"), verification.get("media_phash"),
            media_sha256, verification.get("distance_km"),
            verification.get("live_rain_72h_mm"), verification.get("live_soil_moisture_pct"),
            verification.get("duplicate_of_report_id"),
            verification.get("visual_label"), verification.get("visual_confidence"),
            verification.get("visual_summary"), int(bool(verification.get("visual_matched_claim"))) if verification.get("visual_matched_claim") is not None else None,
            json.dumps(verification.get("live_verification") or {}),
        ),
    )
    return report_id, True


@app.route("/api/gemini-status", methods=["GET"])
def gemini_status():
    """Safe diagnostic endpoint; never returns the API key."""
    return jsonify({
        "configured": bool(GEMINI_API_KEY) and not GEMINI_API_KEY.startswith("PASTE_YOUR_"),
        "model": GEMINI_MODEL,
        "models_to_try": _gemini_models_to_try(),
        "key_source": "GEMINI_API_KEY/GOOGLE_API_KEY" if GEMINI_API_KEY else None,
    })


@app.route("/api/reports", methods=["POST"])
def create_report():
    """Create one field report. Accepts either JSON (text-only, e.g. offline
    queue replay) or multipart/form-data with an optional `media` file field
    (a geo-tagged photo/video of cracks, slope movement, a blocked road,
    etc.) attached alongside the same fields. A photo attachment is run
    through automated verification (location/time/duplicate/live-weather
    cross-checks — see verify_report_media) before being stored; the result
    is returned here so a client can show "submitted, pending review" etc."""
    media_filename, media_type, media_path = None, None, None

    if request.content_type and "multipart/form-data" in request.content_type:
        body = request.form.to_dict()
        try:
            media_filename, media_type = save_report_media(request.files.get("media"))
            if not media_filename:
                return jsonify({"error": "A field photo is required. The image must pass visual AI + live-feed verification before it can enter Field Reports."}), 400
            if media_type != "image":
                return jsonify({"error": "Only images can be verified into Field Reports right now. Please upload a photo."}), 400
            media_path = os.path.join(app.static_folder, media_filename)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
    else:
        body = request.get_json(force=True, silent=True) or {}
        return jsonify({"error": "A verified field photo is required. Text-only reports are not added to Field Reports."}), 400

    db = get_db()
    try:
        report_id, is_new = _insert_report(db, body, media_filename, media_type, media_path)
        db.commit()
    except ValueError as exc:
        db.rollback()
        try:
            payload = json.loads(str(exc))
            return jsonify(payload), 422
        except (TypeError, ValueError):
            return jsonify({"error": str(exc)}), 422

    response = {"id": report_id, "status": "saved" if is_new else "already_synced"}
    if is_new:
        row = db.execute(
            "SELECT verification_status, verification_flags FROM reports WHERE id = ?",
            (report_id,),
        ).fetchone()
        if row:
            response["verification_status"] = row["verification_status"]
            try:
                response["verification_flags"] = json.loads(row["verification_flags"] or "[]")
            except (TypeError, ValueError):
                response["verification_flags"] = []
            full = db.execute(
                "SELECT visual_label, visual_confidence, visual_summary, visual_matched_claim, live_verification_json "
                "FROM reports WHERE id = ?", (report_id,)
            ).fetchone()
            if full:
                response["verification"] = {
                    "visual_label": full["visual_label"],
                    "visual_confidence": full["visual_confidence"],
                    "visual_summary": full["visual_summary"],
                    "visual_matched_claim": bool(full["visual_matched_claim"]) if full["visual_matched_claim"] is not None else None,
                    "live": json.loads(full["live_verification_json"] or "{}"),
                    "flags": response["verification_flags"],
                }
    return jsonify(response), 201


@app.route("/api/reports/<report_id>/review", methods=["POST"])
@admin_required
def review_report(report_id):
    """Admin-only. Manually approve or reject a report after looking at the
    automated verification flags — the automated pipeline only triages, a
    human still makes the final call on anything not auto_approved."""
    body = request.get_json(force=True, silent=True) or {}
    decision = body.get("decision")
    if decision not in ("approve", "reject"):
        return jsonify({"error": "decision must be 'approve' or 'reject'"}), 400

    db = get_db()
    row = db.execute("SELECT id FROM reports WHERE id = ?", (report_id,)).fetchone()
    if row is None:
        return jsonify({"error": "report not found"}), 404

    new_status = "verified" if decision == "approve" else "rejected"
    db.execute(
        "UPDATE reports SET verification_status = ?, reviewed_by_admin = 1, reviewed_at = ? "
        "WHERE id = ?",
        (new_status, now_iso(), report_id),
    )
    db.commit()
    return jsonify({"id": report_id, "verification_status": new_status})




@app.route("/api/reports/sync", methods=["POST"])
def sync_reports():
    """Bulk-upload an offline field-report queue collected while a device had
    no connectivity. Each report is deduped by client_id (generated on the
    device at capture time), so a queue can be safely retried/resent."""
    body = request.get_json(force=True, silent=True) or {}
    reports = body.get("reports", [])
    if not isinstance(reports, list):
        return jsonify({"error": "reports must be a list"}), 400

    db = get_db()
    saved, duplicates, failed = [], [], []
    for r in reports:
        if not isinstance(r, dict):
            failed.append({"error": "malformed report entry"})
            continue
        try:
            report_id, is_new = _insert_report(db, r)
            (saved if is_new else duplicates).append({
                "client_id": r.get("client_id"), "id": report_id,
            })
        except (sqlite3.Error, TypeError, ValueError) as exc:
            failed.append({"client_id": r.get("client_id"), "error": str(exc)})
    db.commit()

    return jsonify({
        "synced": len(saved),
        "already_synced": len(duplicates),
        "failed": len(failed),
        "saved": saved,
        "duplicates": duplicates,
        "errors": failed,
    })


# ---------------------------------------------------------------------------
# Alerts: in-app/web alert feed + a pluggable SMS gateway
# ---------------------------------------------------------------------------
# There's no free universal SMS API, so this doesn't fabricate delivery.
# Every dispatch is always recorded (powering the in-app/web early-warning
# feed below). If TWILIO_* or MSG91_* env vars are set, it also attempts a
# real SMS send through that provider; otherwise it logs the message as
# "simulated" and the dispatch still succeeds as an app-based alert.
TWILIO_SID = os.environ.get("TWILIO_ACCOUNT_SID", "").strip()
TWILIO_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "").strip()
TWILIO_FROM = os.environ.get("TWILIO_FROM_NUMBER", "").strip()
MSG91_KEY = os.environ.get("MSG91_AUTH_KEY", "").strip()
MSG91_SENDER = os.environ.get("MSG91_SENDER_ID", "").strip()
SMS_GATEWAY_CONFIGURED = bool((TWILIO_SID and TWILIO_TOKEN and TWILIO_FROM) or MSG91_KEY)


def dispatch_sms(district, message, recipients=None):
    """Best-effort SMS send. Returns a status dict; never raises — a gateway
    failure should not block the alert from being logged for the app feed."""
    recipients = recipients or []
    if not recipients:
        return {"channel": "sms", "status": "skipped", "detail": "no recipients configured"}

    if TWILIO_SID and TWILIO_TOKEN and TWILIO_FROM:
        try:
            sent = 0
            for to in recipients:
                resp = requests.post(
                    f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_SID}/Messages.json",
                    data={"From": TWILIO_FROM, "To": to, "Body": message},
                    auth=(TWILIO_SID, TWILIO_TOKEN),
                    timeout=8,
                )
                if resp.ok:
                    sent += 1
            return {"channel": "sms", "status": "sent", "provider": "twilio", "count": sent}
        except requests.RequestException as exc:
            logger.warning("Twilio dispatch failed: %s", exc)
            return {"channel": "sms", "status": "failed", "provider": "twilio", "error": str(exc)}

    if MSG91_KEY:
        try:
            resp = requests.post(
                "https://control.msg91.com/api/v5/flow/",
                headers={"authkey": MSG91_KEY, "Content-Type": "application/json"},
                json={"sender": MSG91_SENDER, "short_url": "0",
                      "mobiles": ",".join(recipients), "message": message},
                timeout=8,
            )
            return {
                "channel": "sms", "status": "sent" if resp.ok else "failed",
                "provider": "msg91", "count": len(recipients),
            }
        except requests.RequestException as exc:
            logger.warning("MSG91 dispatch failed: %s", exc)
            return {"channel": "sms", "status": "failed", "provider": "msg91", "error": str(exc)}

    logger.info("SMS gateway not configured — simulated dispatch for %s: %s", district, message)
    return {
        "channel": "sms", "status": "simulated",
        "detail": "No SMS gateway configured (set TWILIO_* or MSG91_* env vars) — "
                   "alert was logged for the in-app/web feed only.",
    }


@app.route("/api/alerts", methods=["GET"])
def list_alerts():
    db = get_db()
    district = request.args.get("district")
    if district:
        rows = db.execute(
            "SELECT * FROM alerts WHERE district = ? ORDER BY dispatched_at DESC LIMIT 100",
            (district,),
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT * FROM alerts ORDER BY dispatched_at DESC LIMIT 100"
        ).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route("/api/alerts/dispatch", methods=["POST"])
@admin_required
def dispatch_alert():
    body = request.get_json(force=True, silent=True) or {}
    district = body.get("district", "unknown")
    level = body.get("level", "watch")
    language = body.get("language", "en")
    message = body.get("message", "")
    recipients = body.get("recipients", [])
    if not message:
        return jsonify({"error": "message is required"}), 400

    alert_id = str(uuid.uuid4())
    dispatched_at = now_iso()

    sms_result = dispatch_sms(district, message, recipients)

    db = get_db()
    db.execute(
        "INSERT INTO alerts (id, district, level, language, message, dispatched_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (alert_id, district, level, language, message, dispatched_at),
    )
    db.commit()

    return jsonify({
        "id": alert_id, "district": district, "level": level, "language": language,
        "message": message, "dispatched_at": dispatched_at,
        "app_feed": "published",
        "sms": sms_result,
    }), 201


# ---------------------------------------------------------------------------
# Ground sensor ingestion (rain gauges, piezometers, extensometers, etc.)
# ---------------------------------------------------------------------------
# No physical sensors are wired up in this deployment. This endpoint exists
# so real hardware/telemetry gateways have somewhere to POST readings to; the
# dashboard/API report honestly when no live sensor has reported recently
# instead of inventing fixed "IoT sensor" points, per this project's README.
SENSOR_STALE_AFTER_SECONDS = 3 * 3600


@app.route("/api/sensors", methods=["POST"])
def ingest_sensor_reading():
    body = request.get_json(force=True, silent=True) or {}
    try:
        sensor_id = str(body["sensor_id"])
        sensor_type = str(body["sensor_type"])
        value = float(body["value"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "sensor_id, sensor_type, and numeric value are required"}), 400

    reading_id = str(uuid.uuid4())
    db = get_db()
    db.execute(
        "INSERT INTO sensor_readings (id, sensor_id, sensor_type, district, lat, lon, value, unit, recorded_at, received_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            reading_id, sensor_id, sensor_type, body.get("district"),
            body.get("lat"), body.get("lon"), value, body.get("unit", ""),
            body.get("recorded_at") or now_iso(), now_iso(),
        ),
    )
    db.commit()
    return jsonify({"id": reading_id, "status": "ingested"}), 201


@app.route("/api/sensors/latest", methods=["GET"])
def latest_sensor_readings():
    db = get_db()
    rows = db.execute(
        "SELECT * FROM sensor_readings ORDER BY recorded_at DESC LIMIT 500"
    ).fetchall()

    latest_by_sensor = {}
    for r in rows:
        if r["sensor_id"] not in latest_by_sensor:
            latest_by_sensor[r["sensor_id"]] = dict(r)

    now = datetime.now(timezone.utc)
    live = []
    for reading in latest_by_sensor.values():
        try:
            recorded = datetime.fromisoformat(reading["recorded_at"])
            is_live = (now - recorded).total_seconds() < SENSOR_STALE_AFTER_SECONDS
        except (TypeError, ValueError):
            is_live = False
        reading["is_live"] = is_live
        live.append(reading)

    return jsonify({
        "sensors_reporting": sum(1 for r in live if r["is_live"]),
        "readings": live,
        "note": (
            "No physical sensors are connected in this deployment — this "
            "reflects only readings actually POSTed to /api/sensors. When a "
            "live soil-moisture sensor is present near a prediction point, "
            "its reading overrides the weather-model soil moisture input."
            if not live else None
        ),
    })


init_db()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
