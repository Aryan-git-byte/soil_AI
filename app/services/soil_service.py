# app/services/soil_service.py - Production Ready (No Files Required)
import os
import logging

logger = logging.getLogger(__name__)

# Try to import rasterio, but don't fail if unavailable
try:
    import rasterio
    from rasterio.windows import Window
    RASTERIO_AVAILABLE = True
except ImportError as e:
    RASTERIO_AVAILABLE = False
    logger.warning(f"⚠️ Rasterio not available: {e}")

# Load environment variables
SAND_FILE = os.getenv("SOIL_SAND_FILE")
CLAY_FILE = os.getenv("SOIL_CLAY_FILE")
SILT_FILE = os.getenv("SOIL_SILT_FILE")
TEXTURE_FILE = os.getenv("SOIL_TEXTURE_FILE")

# ✅ FIX: Only validate files if they're configured AND exist
SOIL_FILES_CONFIGURED = all([SAND_FILE, CLAY_FILE, SILT_FILE, TEXTURE_FILE])
SOIL_FILES_EXIST = False

if SOIL_FILES_CONFIGURED:
    SOIL_FILES_EXIST = all([
        os.path.exists(SAND_FILE),
        os.path.exists(CLAY_FILE),
        os.path.exists(SILT_FILE),
        os.path.exists(TEXTURE_FILE)
    ])

# Initialize datasets only if everything is available
sand_ds = None
clay_ds = None
silt_ds = None
texture_ds = None

if RASTERIO_AVAILABLE and SOIL_FILES_EXIST:
    try:
        sand_ds = rasterio.open(SAND_FILE)
        clay_ds = rasterio.open(CLAY_FILE)
        silt_ds = rasterio.open(SILT_FILE)
        texture_ds = rasterio.open(TEXTURE_FILE)
        logger.info("✓ Soil datasets loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load soil datasets: {e}")
        sand_ds = clay_ds = silt_ds = texture_ds = None
else:
    if not RASTERIO_AVAILABLE:
        logger.info("ℹ️ Soil analysis disabled: rasterio not available")
    elif not SOIL_FILES_CONFIGURED:
        logger.info("ℹ️ Soil analysis disabled: file paths not configured")
    else:
        logger.info("ℹ️ Soil analysis disabled: data files not found")

# USDA Texture Class Mapping
TEXTURE_MAP = {
    1: "Clay",
    2: "Silty Clay",
    3: "Sandy Clay",
    4: "Clay Loam",
    5: "Silty Clay Loam",
    6: "Sandy Clay Loam",
    7: "Loam",
    8: "Silt Loam",
    9: "Sandy Loam",
    10: "Silt",
    11: "Sand"
}


def get_pixel(ds, lat, lon):
    """Extract pixel value from raster dataset"""
    if ds is None:
        return None
    
    try:
        row, col = ds.index(lon, lat)
        window = Window(col, row, 1, 1)
        value = ds.read(1, window=window)[0, 0]

        if value == ds.nodata:
            return None
        
        return float(value)
    except Exception as e:
        logger.debug(f"Failed to get pixel at ({lat}, {lon}): {e}")
        return None


def get_soil_physical(lat: float, lon: float):
    """
    Get soil physical properties at given coordinates.
    Returns dict with sand/clay/silt percentages and texture class.
    """
    # Check if soil data is available
    if not all([sand_ds, clay_ds, silt_ds, texture_ds]):
        return {
            "sand_percent": None,
            "clay_percent": None,
            "silt_percent": None,
            "texture": "Not Available",
            "depth_cm": "0-5",
            "source": "OpenLandMap 250m (0–5 cm Depth)",
            "available": False,
            "note": "Soil data files not available on this server"
        }
    
    try:
        sand = get_pixel(sand_ds, lat, lon)
        clay = get_pixel(clay_ds, lat, lon)
        silt = get_pixel(silt_ds, lat, lon)
        texture_code = get_pixel(texture_ds, lat, lon)
        
        texture = None
        try:
            texture = TEXTURE_MAP.get(int(texture_code), "Unknown") if texture_code is not None else "Unknown"
        except Exception:
            texture = "Unknown"

        return {
            "sand_percent": sand,
            "clay_percent": clay,
            "silt_percent": silt,
            "texture": texture,
            "depth_cm": "0-5",
            "source": "OpenLandMap 250m (0–5 cm Depth)",
            "available": True
        }
    except Exception as e:
        logger.error(f"Error getting soil data for ({lat}, {lon}): {e}")
        return {
            "sand_percent": None,
            "clay_percent": None,
            "silt_percent": None,
            "texture": "Error",
            "depth_cm": "0-5",
            "source": "OpenLandMap 250m (0–5 cm Depth)",
            "available": False,
            "error": str(e)
        }


def classify_indian_soil_type(
    sand_percent: float,
    clay_percent: float,
    silt_percent: float,
    texture: str,
    lat: float = None,
    lon: float = None
) -> dict:
    """
    Classifies soil into Indian soil types based on physical properties and location.
    """
    
    if sand_percent is None or clay_percent is None or silt_percent is None:
        return {
            "indian_soil_type": "Data Not Available",
            "confidence": "N/A",
            "description": "Soil classification unavailable - missing soil composition data",
            "characteristics": [],
            "available": False
        }
    
    soil_type = "Unknown"
    confidence = "Medium"
    description = ""
    characteristics = []
    
    # Geographic regions (rough lat/lon boundaries)
    is_northern_plains = lat and 24 <= lat <= 30 and 75 <= lon <= 88
    is_deccan_plateau = lat and 15 <= lat <= 24 and 74 <= lon <= 81
    is_western_coast = lat and 8 <= lat <= 20 and 72 <= lon <= 77
    is_eastern_coast = lat and 10 <= lat <= 20 and 78 <= lon <= 85
    is_northwest_arid = lat and 24 <= lat <= 30 and 68 <= lon <= 75
    is_northeast = lat and 22 <= lat <= 29 and 88 <= lon <= 97
    is_himalayan = lat and lat >= 28
    
    # 1. BLACK SOIL (Regur) - High clay, dark color, Deccan region
    if clay_percent >= 40 and texture in ["Clay", "Silty Clay", "Clay Loam"]:
        if is_deccan_plateau:
            soil_type = "Black Soil (Regur)"
            confidence = "High"
            description = "Deep, clayey black soil rich in lime, iron, and magnesium. Excellent moisture retention."
            characteristics = [
                "Ideal for cotton cultivation",
                "High water retention capacity",
                "Self-ploughing nature",
                "Rich in calcium and magnesium",
                "pH typically 7.2-8.5 (slightly alkaline)"
            ]
        else:
            soil_type = "Black Soil (Regur)"
            confidence = "Medium"
            description = "Clayey soil with black soil characteristics, though outside typical Deccan region."
            characteristics = [
                "High clay content",
                "Good for cotton and other crops",
                "Requires careful water management"
            ]
    
    # 2. ALLUVIAL SOIL - Balanced composition, plains regions
    elif 20 <= clay_percent <= 40 and 20 <= sand_percent <= 60 and texture in ["Loam", "Silt Loam", "Clay Loam", "Sandy Loam"]:
        if is_northern_plains or is_eastern_coast:
            soil_type = "Alluvial Soil"
            confidence = "High"
            description = "Fertile soil deposited by rivers, rich in nutrients. Most productive agricultural soil in India."
            characteristics = [
                "High fertility and productivity",
                "Good for cereals (wheat, rice, sugarcane)",
                "Well-balanced texture",
                "Rich in potash but deficient in phosphorus",
                "Renewed by annual flooding"
            ]
        else:
            soil_type = "Alluvial Soil"
            confidence = "Medium"
            description = "Balanced soil composition similar to alluvial characteristics."
            characteristics = [
                "Moderate fertility",
                "Good drainage",
                "Suitable for diverse crops"
            ]
    
    # 3. ARID/DESERT SOIL - Very high sand content
    elif sand_percent >= 70 and clay_percent < 15:
        if is_northwest_arid:
            soil_type = "Arid/Desert Soil"
            confidence = "High"
            description = "Sandy soil with low organic matter, found in arid and semi-arid regions."
            characteristics = [
                "Low water retention",
                "Poor in organic matter",
                "Requires irrigation for cultivation",
                "High salinity in some areas",
                "Suitable for drought-resistant crops (bajra, jowar)"
            ]
        else:
            soil_type = "Sandy Soil"
            confidence = "Medium"
            description = "High sand content soil with low water retention."
            characteristics = [
                "Low fertility",
                "Requires frequent irrigation",
                "Good drainage but nutrient-poor"
            ]
    
    # 4. RED & YELLOW SOIL - Moderate sand/clay, peninsular region
    elif 30 <= sand_percent <= 60 and 15 <= clay_percent <= 35:
        if is_deccan_plateau or is_eastern_coast or is_western_coast:
            soil_type = "Red & Yellow Soil"
            confidence = "High"
            description = "Formed from weathering of ancient crystalline rocks. Reddish color due to iron oxide."
            characteristics = [
                "Good for groundnut, pulses, and millets",
                "Deficient in nitrogen and phosphorus",
                "Requires fertilization",
                "Porous and friable texture",
                "Red when well-drained, yellow when waterlogged"
            ]
        else:
            soil_type = "Red & Yellow Soil"
            confidence = "Medium"
            description = "Sandy to loamy soil with moderate fertility."
            characteristics = [
                "Moderate fertility",
                "Suitable for various crops with proper management"
            ]
    
    # 5. LATERITE SOIL - High clay but leached, high rainfall areas
    elif clay_percent >= 30 and (is_western_coast or is_northeast):
        soil_type = "Laterite Soil"
        confidence = "High"
        description = "Formed in high rainfall areas with alternate wet and dry periods. Rich in iron and aluminum."
        characteristics = [
            "Brick-like when dried",
            "Poor fertility due to leaching",
            "Acidic pH (5-6)",
            "Suitable for tea, coffee, cashew, rubber",
            "Rich in iron oxide and aluminum",
            "Poor in nitrogen, lime, and potash"
        ]
    
    # 6. FOREST/MOUNTAIN SOIL - Himalayan or hilly regions
    elif is_himalayan or (lat and lat >= 27):
        soil_type = "Forest/Mountain Soil"
        confidence = "Medium"
        description = "Soil found in hilly and mountainous regions. Composition varies with altitude and forest cover."
        characteristics = [
            "Rich in organic matter",
            "Varies greatly with altitude",
            "Good for horticulture and forestry",
            "Prone to erosion on slopes",
            "Acidic to neutral pH"
        ]
    
    # 7. SALINE & ALKALINE SOIL - Would need EC or pH data for accurate detection
    elif is_northern_plains and clay_percent > 25:
        soil_type = "Potentially Saline/Alkaline Soil"
        confidence = "Low"
        description = "Region prone to salinity. Requires EC and pH testing for confirmation."
        characteristics = [
            "May have high salt content",
            "Requires proper drainage and leaching",
            "Suitable for salt-tolerant crops",
            "EC and pH testing recommended"
        ]
    
    # 8. DEFAULT - Based purely on texture
    else:
        soil_type = f"{texture} (Unclassified)"
        confidence = "Low"
        description = f"Soil texture is {texture}. Location-based classification inconclusive."
        characteristics = [
            f"Sand: {sand_percent:.1f}%",
            f"Clay: {clay_percent:.1f}%",
            f"Silt: {silt_percent:.1f}%"
        ]
    
    return {
        "indian_soil_type": soil_type,
        "confidence": confidence,
        "description": description,
        "characteristics": characteristics,
        "classification_factors": {
            "sand_percent": round(sand_percent, 1),
            "clay_percent": round(clay_percent, 1),
            "silt_percent": round(silt_percent, 1),
            "texture": texture,
            "region_considered": lat is not None and lon is not None
        },
        "available": True
    }
