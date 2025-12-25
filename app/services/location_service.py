# app/services/location_service.py - Complete: All Features + 300ms Architecture
import httpx
import asyncio
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from app.core.cache import get_cached, set_cached

logger = logging.getLogger(__name__)

class LocationService:
    BASE_HEADERS = {"User-Agent": "FarmAI/1.0"}

    # ---------------------------------------------------
    # Helper: Clean names for Wikipedia
    # ---------------------------------------------------
    def clean_name(self, name: str | None):
        if not name:
            return None
        name = name.split(",")[0].strip()
        name = name.replace(" ", "_")
        name = name.title()
        return name

    # ---------------------------------------------------
    # Async HTTP Helper (The engine for speed)
    # ---------------------------------------------------
    async def _get(self, client: httpx.AsyncClient, url: str, params: dict = None, timeout: float = 2.0):
        try:
            return await client.get(url, params=params, timeout=timeout)
        except Exception as e:
            # logger.warning(f"API Fetch failed for {url}: {e}")
            return None

    # ---------------------------------------------------
    # 1. Reverse Geocoding (Nominatim)
    # ---------------------------------------------------
    async def reverse_geocode(self, client: httpx.AsyncClient, lat: float, lon: float):
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {
            "lat": lat,
            "lon": lon,
            "format": "json",
            "zoom": 14,
            "addressdetails": 1
        }
        res = await self._get(client, url, params=params, timeout=1.5)
        
        if not res or res.status_code != 200:
            return {}

        data = res.json()
        addr = data.get("address", {})

        return {
            "display_name": data.get("display_name"),
            "city": addr.get("city") or addr.get("town") or addr.get("village"),
            "district": addr.get("state_district"),
            "state": addr.get("state"),
            "country": addr.get("country")
        }

    # ---------------------------------------------------
    # 2. Wikipedia Summary
    # ---------------------------------------------------
    async def get_wikipedia_summary(self, client: httpx.AsyncClient, name: str | None):
        if not name:
            return None

        safe = self.clean_name(name)
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{safe}"
        
        # Fast timeout for Wiki, it usually responds quick
        res = await self._get(client, url, timeout=1.0)
        
        if res and res.status_code == 200:
            return res.json().get("extract")
        return None

    # ---------------------------------------------------
    # 3. Wikipedia Fallback Search
    # ---------------------------------------------------
    async def wikipedia_search(self, client: httpx.AsyncClient, query: str):
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json"
        }
        
        res = await self._get(client, url, params=params, timeout=1.5)
        
        if res and res.status_code == 200:
            results = res.json().get("query", {}).get("search", [])
            if results:
                return results[0]["title"]
        return None

    # ---------------------------------------------------
    # 4. Nearest POI (Preserving Tiered Logic)
    # ---------------------------------------------------
    async def get_nearest_poi(self, client: httpx.AsyncClient, lat: float, lon: float):
        # Tier 1 — HIGH IMPORTANCE POIs
        TIER_1 = [
            "wd:Q33506",      # museum
            "wd:Q7692360",    # science centre
            "wd:Q570116",     # tourist attraction
            "wd:Q4989906",    # monument
            "wd:Q12973014",   # heritage site
            "wd:Q839954",     # religious building
            "wd:Q23442",      # archaeological site
        ]

        # Tier 2 — MEDIUM IMPORTANCE POIs
        TIER_2 = [
            "wd:Q860861",     # park
            "wd:Q41176",      # educational institution
            "wd:Q29468",      # government building
            "wd:Q327333",     # library
            "wd:Q207694",     # public venue
        ]

        # Tier 3 — LOW IMPORTANCE POIs
        TIER_3 = [
            "wd:Q41176",      # generic public building
            "wd:Q483110",     # infrastructure type
        ]

        async def run_query(category_list, radius="0.45"):
            if not category_list: return None
            categories = " ".join(category_list)
            
            # Optimized SPARQL: Use direct P31 if possible, but keep original structure for compatibility
            QUERY = f"""
            SELECT ?itemLabel WHERE {{
                SERVICE wikibase:around {{
                    ?item wdt:P625 ?coord .
                    bd:serviceParam wikibase:center "Point({lon} {lat})"^^geo:wktLiteral .
                    bd:serviceParam wikibase:radius "{radius}" .
                }}
                VALUES ?category {{ {categories} }}
                ?item wdt:P31/wdt:P279* ?category .
                SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
            }} LIMIT 1
            """
            # Strict timeout (800ms) to prevent hanging
            res = await self._get(client, "https://query.wikidata.org/sparql", 
                                  params={"query": QUERY, "format": "json"}, timeout=0.8)
            
            if res and res.status_code == 200:
                try:
                    results = res.json()["results"]["bindings"]
                    if results:
                        return results[0]["itemLabel"]["value"]
                except:
                    pass
            return None

        # Execute Tiers sequentially (because priority matters)
        # But since we have Redis caching, this cost is only paid ONCE per 24h.
        poi = await run_query(TIER_1)
        if poi: return poi

        poi = await run_query(TIER_2)
        if poi: return poi

        poi = await run_query(TIER_3)
        if poi: return poi

        return None

    # ---------------------------------------------------
    # 5. Nearby Monuments (Preserving Exclusions)
    # ---------------------------------------------------
    async def get_nearby_monuments(self, client: httpx.AsyncClient, lat: float, lon: float):
        QUERY = f"""
        SELECT ?itemLabel WHERE {{
            SERVICE wikibase:around {{
                ?item wdt:P625 ?coord .
                bd:serviceParam wikibase:center "Point({lon} {lat})"^^geo:wktLiteral .
                bd:serviceParam wikibase:radius "5" .
            }}
            VALUES ?category {{
                wd:Q4989906 wd:Q570116 wd:Q839954 wd:Q12973014 wd:Q860861
            }}
            ?item wdt:P31/wdt:P279* ?category .
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }} LIMIT 50
        """
        
        res = await self._get(client, "https://query.wikidata.org/sparql", 
                              params={"query": QUERY, "format": "json"}, timeout=2.0)
        
        monuments = []
        exclude_keywords = ["grave", "cemetery", "tomb", "burial"]
        seen = set()

        if res and res.status_code == 200:
            try:
                for entry in res.json()["results"]["bindings"]:
                    label = entry["itemLabel"]["value"]
                    
                    # Filter logic from original code
                    if label.startswith("Q"): continue
                    if any(k in label.lower() for k in exclude_keywords): continue
                    
                    if label not in seen:
                        seen.add(label)
                        monuments.append(label)
            except:
                pass
        
        return monuments[:20]

    # ---------------------------------------------------
    # MAIN PIPELINE: Parallel + Cached + Full Features
    # ---------------------------------------------------
    async def build_location_context(self, lat: float, lon: float, weather_service=None):
        # 1. CHECK CACHE FIRST (The Speed Layer)
        lat_r = round(lat, 3)
        lon_r = round(lon, 3)
        cache_key = f"loc_ctx:{lat_r}:{lon_r}"

        cached_ctx = await get_cached(cache_key)
        if cached_ctx:
            # We trust the cache for 24h. This is how we hit 50ms.
            return cached_ctx

        # 2. EXECUTE LOGIC (The Logic Layer)
        async with httpx.AsyncClient(headers=self.BASE_HEADERS) as client:
            
            # Start independent tasks in parallel
            tasks = [
                self.reverse_geocode(client, lat, lon),       # Task 0
                self.get_nearest_poi(client, lat, lon),       # Task 1
                self.get_nearby_monuments(client, lat, lon)   # Task 2
            ]

            if weather_service:
                tasks.append(weather_service.get_current_weather(lat, lon, client))
            else:
                tasks.append(asyncio.sleep(0)) # No-op

            # Wait for all
            results = await asyncio.gather(*tasks)
            basic, poi, monuments, weather = results[0], results[1], results[2], results[3]
            
            # 3. Handle Fallback Logic (Sequential but fast)
            summary = None
            
            # Priority 1: Summary of POI
            if poi:
                summary = await self.get_wikipedia_summary(client, poi)
            
            # Priority 2: Summary of City/District
            if not summary:
                place = basic.get("city") or basic.get("district") or basic.get("state")
                summary = await self.get_wikipedia_summary(client, place)
            
            # Priority 3: Search Wikipedia
            if not summary:
                search_place = basic.get("city") or basic.get("district")
                if search_place:
                    search_title = await self.wikipedia_search(client, search_place)
                    if search_title:
                        summary = await self.get_wikipedia_summary(client, search_title)
            
            # Priority 4: Default text
            if not summary:
                summary = f"Location near ({lat}, {lon})."

            result = {
                "generated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
                "coordinates": {"lat": lat, "lon": lon},
                "location_info": basic,
                "nearest_place": poi,
                "description": summary,
                "weather": weather,
                "nearby_monuments": monuments
            }

            # 4. SAVE TO CACHE (24 Hours)
            await set_cached(cache_key, result, expire=86400)

            return result

# Global Instance
location_service = LocationService()

async def get_location_context(lat: float, lon: float):
    return await location_service.build_location_context(lat, lon)