from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from urllib.parse import quote
from urllib.request import Request
from src.ats.resilience import fetch_json

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
KEYWORDS = {
    "energy": ["oil", "brent", "opec", "lng", "gas", "pipeline", "refinery", "hormuz", "bab el mandeb", "shipping"],
    "trade": ["tariff", "export restriction", "sanction", "trade war", "supply chain", "container", "freight"],
    "rates": ["fed", "federal reserve", "rbi", "interest rate", "treasury yield", "inflation"],
    "geopolitical": ["war", "missile", "ceasefire", "conflict", "escalation", "military", "sanctions"],
}


def parse_event_time(value: str | None, fallback: str) -> str:
    if not value:
        return fallback
    m = re.fullmatch(r"(\d{8})(\d{6})?", str(value))
    if m:
        fmt = "%Y%m%d%H%M%S" if m.group(2) else "%Y%m%d"
        return datetime.strptime(m.group(1) + (m.group(2) or "000000"), fmt).replace(tzinfo=timezone.utc).isoformat()
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).astimezone(timezone.utc).isoformat()
    except ValueError:
        return fallback


def classify(title: str) -> tuple[str, float]:
    t = title.lower()
    scores = {k: sum(1 for w in ws if w in t) for k, ws in KEYWORDS.items()}
    cat = max(scores, key=scores.get)
    sev = min(1.0, max(scores.values(), default=0) / 3.0 + (0.2 if any(x in t for x in ["attack", "war", "sanction", "strike", "closure"]) else 0))
    return (cat if max(scores.values(), default=0) > 0 else "other", sev)


class GDELTEventProvider:
    def fetch(self, query="India OR NIFTY OR oil OR OPEC OR Iran OR China OR tariff OR RBI OR Fed", maxrecords=40):
        params = f"?query={quote(query)}&mode=artlist&format=json&maxrecords={maxrecords}&sort=datedesc"
        req = Request(GDELT_URL + params, headers={"User-Agent": "ATS-v2/2.0"})
        payload = fetch_json(req, timeout=15)
        now = datetime.now(timezone.utc).isoformat()
        out = []
        for a in payload.get("articles", []):
            title = a.get("title") or ""
            url = a.get("url") or ""
            event_time = parse_event_time(a.get("seendate"), now)
            category, sev = classify(title)
            eid = hashlib.sha1((url or title).encode()).hexdigest()
            novelty = 1.0 if category in {"geopolitical", "energy", "trade"} else 0.0
            out.append((eid, event_time, event_time, now, "GDELT", category, sev, novelty, title, json.dumps({"url": url, "domain": a.get("domain"), "query": query})))
        return out
