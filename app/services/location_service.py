import requests
from datetime import datetime


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
    # Reverse Geocoding (City / District / State)
    # ---------------------------------------------------
    def reverse_geocode(self, lat: float, lon: float):
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {
            "lat": lat,
            "lon": lon,
            "format": "json",
            "zoom": 14,
            "addressdetails": 1
        }

        data = requests.get(url, params=params, headers=self.BASE_HEADERS).json()
        addr = data.get("address", {})

        return {
            "display_name": data.get("display_name"),
            "city": addr.get("city") or addr.get("town") or addr.get("village"),
            "district": addr.get("state_district"),
            "state": addr.get("state"),
            "country": addr.get("country")
        }

    # ---------------------------------------------------
    # Wikipedia Summary (for POI or City)
    # ---------------------------------------------------
    def get_wikipedia_summary(self, name: str | None):
        if not name:
            return None

        safe = self.clean_name(name)
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{safe}"

        try:
            res = requests.get(url, headers=self.BASE_HEADERS)
            if res.status_code == 200:
                return res.json().get("extract")
        except:
            pass

        return None

    # ---------------------------------------------------
    # Wikipedia Fallback Search
    # ---------------------------------------------------
    def wikipedia_search(self, query: str):
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json"
        }

        try:
            res = requests.get(url, params=params, headers=self.BASE_HEADERS).json()
            results = res.get("query", {}).get("search", [])
            if results:
                return results[0]["title"]
        except:
            pass

        return None

    # ---------------------------------------------------
    # Find Nearest POI within 250m
    # ---------------------------------------------------
    def get_nearest_poi(self, lat: float, lon: float):
        # --------------------------------------------
        # Tier 1 — HIGH IMPORTANCE POIs
        # --------------------------------------------
        TIER_1 = [
            "wd:Q33506",      # museum
            "wd:Q7692360",    # science centre
            "wd:Q570116",     # tourist attraction
            "wd:Q4989906",    # monument
            "wd:Q12973014",   # heritage site
            "wd:Q839954",     # religious building
            "wd:Q23442",      # archaeological site
        ]

        # --------------------------------------------
        # Tier 2 — MEDIUM IMPORTANCE POIs
        # --------------------------------------------
        TIER_2 = [
            "wd:Q860861",     # park
            "wd:Q41176",      # educational institution
            "wd:Q29468",      # government building
            "wd:Q327333",     # library
            "wd:Q207694",     # public venue
        ]

        # --------------------------------------------
        # Tier 3 — LOW IMPORTANCE POIs
        # --------------------------------------------
        TIER_3 = [
            "wd:Q41176",      # generic public building
            "wd:Q483110",     # infrastructure type
        ]

        # --------------------------------------------
        # Helper: SPARQL Runner
        # --------------------------------------------
        def run_query(category_list):
            if not category_list:
                return None

            categories = " ".join(category_list)

            QUERY = f"""
            SELECT ?itemLabel WHERE {{
                SERVICE wikibase:around {{
                    ?item wdt:P625 ?coord .
                    bd:serviceParam wikibase:center "Point({lon} {lat})"^^geo:wktLiteral .
                    bd:serviceParam wikibase:radius "0.45" .   # 450 meters
                }}

                VALUES ?category {{ {categories} }}
                ?item wdt:P31/wdt:P279* ?category .

                SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
            }}
            LIMIT 5
            """

            try:
                res = requests.get(
                    "https://query.wikidata.org/sparql",
                    params={"query": QUERY, "format": "json"}
                ).json()

                results = res["results"]["bindings"]
                if not results:
                    return None

                return results[0]["itemLabel"]["value"]
            except:
                return None

        # --------------------------------------------
        # Run By Priority: Tier 1 → Tier 2 → Tier 3
        # --------------------------------------------
        poi = run_query(TIER_1)
        if poi:
            return poi

        poi = run_query(TIER_2)
        if poi:
            return poi

        poi = run_query(TIER_3)
        if poi:
            return poi

        return None


    # ---------------------------------------------------
    # Nearby monuments, attractions, temples, parks
    # ---------------------------------------------------
    def get_nearby_monuments(self, lat: float, lon: float):
        QUERY = f"""
        SELECT ?itemLabel WHERE {{
            SERVICE wikibase:around {{
                ?item wdt:P625 ?coord .
                bd:serviceParam wikibase:center "Point({lon} {lat})"^^geo:wktLiteral .
                bd:serviceParam wikibase:radius "5" .
            }}

            VALUES ?category {{
                wd:Q4989906      # monument
                wd:Q570116       # tourist attraction
                wd:Q839954       # religious building
                wd:Q12973014     # heritage site
                wd:Q860861       # park
            }}

            ?item wdt:P31/wdt:P279* ?category .
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        LIMIT 50
        """

        try:
            response = requests.get(
                "https://query.wikidata.org/sparql",
                params={"query": QUERY, "format": "json"}
            ).json()

            # Filter and deduplicate
            seen = set()
            filtered_monuments = []
            
            # Keywords to exclude
            exclude_keywords = ["grave", "cemetery", "tomb", "burial"]
            
            for entry in response["results"]["bindings"]:
                label = entry["itemLabel"]["value"]
                
                # Skip Q-IDs without labels
                if label.startswith("Q"):
                    continue
                
                # Skip if label contains excluded keywords
                if any(keyword in label.lower() for keyword in exclude_keywords):
                    continue
                
                # Skip duplicates
                if label not in seen:
                    seen.add(label)
                    filtered_monuments.append(label)
            
            # Return only top 20 meaningful monuments
            return filtered_monuments[:20]
        except:
            return []

    # ---------------------------------------------------
    # MAIN PIPELINE: Build Full Location Context
    # ---------------------------------------------------
    def build_location_context(self, lat: float, lon: float, weather_service=None):
        # 1. Basic administrative location
        basic = self.reverse_geocode(lat, lon)

        # 2. Nearest landmark / POI
        poi = self.get_nearest_poi(lat, lon)

        # 3. Wikipedia summary for POI (highest priority)
        summary = None
        if poi:
            summary = self.get_wikipedia_summary(poi)

        # 4. If no POI summary → fall back to city/district/state
        if not summary:
            place = basic.get("city") or basic.get("district") or basic.get("state")
            summary = self.get_wikipedia_summary(place)

        # 5. If still no summary → Wikipedia search
        if not summary:
            search_title = self.wikipedia_search(place)
            if search_title:
                summary = self.get_wikipedia_summary(search_title)

        # 6. Last fallback (safe)
        if not summary:
            summary = f"{place} is located near ({lat}, {lon})."

        # 7. Weather (optional)
        weather = None
        if weather_service:
            weather = weather_service.get_current_weather(lat, lon)

        # 8. Nearby monuments & attractions (5km)
        monuments = self.get_nearby_monuments(lat, lon)

        # 9. Final output
        return {
            "generated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "coordinates": {"lat": lat, "lon": lon},
            "location_info": basic,
            "nearest_place": poi,
            "description": summary,
            "weather": weather,
            "nearby_monuments": monuments
        }
