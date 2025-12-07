import os
from supabase import create_client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

async def get_latest_sensor_data(lat, lon, radius_km=1.0):
    """
    Fetch the latest sensor reading near the coordinates.
    """
    # You can enhance this later with Haversine filtering
    resp = supabase.table("sensor_data") \
        .select("*") \
        .order("timestamp", desc=True) \
        .limit(1) \
        .execute()

    if resp.data and len(resp.data) > 0:
        row = resp.data[0]
        return {
            "soil_moisture": row["soil_moisture"],
            "ec": row["ec"],
            "soil_temperature": row["soil_temperature"],
            "n": row["n"],
            "p": row["p"],
            "k": row["k"],
            "ph": row["ph"],
            "timestamp": row["timestamp"]
        }
    
    return None
