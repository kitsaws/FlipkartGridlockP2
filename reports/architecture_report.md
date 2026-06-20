# AI-Driven Parking Intelligence System
**System Architecture & Pipeline Report**

## 1. Executive Summary
We have built a **Spatio-Temporal Enforcement Priority Intelligence System**. Instead of attempting to build a black-box predictive AI (which often fails in law enforcement contexts), we built a deterministic, mathematical rules-engine. The system ingests messy historical parking tickets and processes them through spatial grids and temporal buckets to produce a ranked "Hit List" of enforcement targets.

---

## 2. Data Evolution: Raw to Refined

### The Original Data
The raw dataset was highly granular and noisy. Every row was a single parking ticket with:
- Exact GPS Coordinates (Latitude / Longitude)
- Messy, mixed-format string timestamps (`created_datetime`)
- Complex JSON arrays describing infractions (e.g., `["WRONG PARKING", "OBSTRUCTING DRIVER"]`)
- Data Leakage columns (timestamps showing when data was synced to the cloud, which happens *after* the event).

### How We Modified It
We built a Data Cleaning and Feature Engineering pipeline (`02_data_cleaning.py` & `03_feature_engineering.py`) that performed the following transformations:
1. **Leakage Removal**: Dropped any post-event columns so the system only evaluates what happened *at the time* of the violation.
2. **Temporal Extraction**: Parsed timestamps into `hour`, `day_of_week`, and binary flags like `is_night` and `is_peak_hour`.
3. **Violation Parsing**: We used "Multi-Hot Encoding" to burst the JSON string arrays into individual boolean columns (e.g., `is_wrong_parking = 1`, `is_double_parking = 0`).

---

## 3. Why did we use Uber's H3 Spatial Grid?
Raw GPS coordinates (Lat/Lon) are continuous. If Car A parks at `12.9716, 77.5946` and Car B parks at `12.9717, 77.5947`, a computer sees them as two entirely different locations.

**The Solution:** We used the `h3` library (Resolution 8, roughly 460-meter edge length). H3 overlays a mathematically perfect hexagonal grid across the entire planet. 
- By mapping every GPS coordinate to its parent H3 Hexagon, we essentially "binned" the continuous map into discrete city blocks.
- This allowed us to group thousands of nearby, isolated violations into single, high-density **Neighborhood Profiles**, making it possible to calculate things like "Density" and "Repeat Offenders" for a specific physical space.

---

## 4. The Aggregation Strategy
After applying H3 grids, we completely collapsed the dataset. We grouped the data by `[h3_cell_id, hour]`. 
This meant we threw away the chronological timeline (e.g., January 1st, January 2nd) and compressed everything into **Zone-Time Profiles** (e.g., "Hexagon A at 5:00 AM"). This allows the system to evaluate how consistently chaotic a specific hour is across all weeks.

---

## 5. Hotspot Clustering (HDBSCAN)
We used HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) to find distinct Hotspots. 

**The Challenge:**
Because H3 cells are perfectly spaced, HDBSCAN originally couldn't find distinct clusters—it just saw a massive, uniform grid covering the city. 

**The Fix:**
Before clustering, we implemented a **Density Filter**. We calculated the total violation volume for every cell and threw away the low-density "bridge" streets. This caused the continuous city grid to shatter into separate, ultra-dense "Islands". We then ran HDBSCAN on these islands. 
As a result, the system perfectly encapsulated the massive cluster in the center of Bangalore, while simultaneously finding smaller, isolated hotspots on the city periphery.

---

## 6. The Enforcement Priority Engine
This is the core of the system (`06_priority_engine.py`). For every Zone-Time profile, the engine computes a definitive **Priority Score (0-100)** using a weighted formula:

> **Priority Score = (40% Density) + (30% Persistence) + (20% Severity) + (10% Repeat Offenders)**

1. **Volume/Density (40%)**: Simply the raw number of violations in that zone at that hour.
2. **Weekly Persistence (30%)**: How many unique calendar weeks did violations occur here? A zone with 100 tickets in 1 week is a fluke. A zone with 10 tickets per week for 10 weeks is a systemic hotspot.
3. **Severity Score (20%)**: We weighted the infractions. "Wrong Parking" has a baseline multiplier of 1.0. "Double Parking" or "Parking on Footpath" is 2.0. "Obstructing Driver" or "Near Hospital" is 3.0. A zone full of severe infractions scores much higher than a zone full of harmless infractions.
4. **Repeat Offender Ratio (10%)**: The ratio of unique vehicles to total tickets. If the same vehicles are constantly illegally parked in a zone, it proves the current deterrents are failing.

### Final Output
The system spits out the absolute Top 50 Unique Hexagons (with their peak hours) as an actionable "Hit List" for traffic commanders, accompanied by geographic heatmaps and Temporal Analytics charts.
