import rasterio
from rasterio.windows import Window

# ----------------------------
# FILE PATHS (update for your PC)
# ----------------------------
SAND_FILE = r"D:\Downloads\Soil-Data\sand.wfraction_usda.3a1a1a_m_250m_b0cm_19500101_20171231_go_epsg.4326_v0.2.tif"
CLAY_FILE = r"D:\Downloads\Soil-Data\sol_clay.wfraction_usda.3a1a1a_m_250m_b0..0cm_1950..2017_v0.2.tif"
SILT_FILE = r"D:\Downloads\Soil-Data\sol_silt.wfraction_usda.3a1a1a_m_250m_b0..0cm_1950..2017_v0.2.tif"
TEXTURE_FILE = r"D:\Downloads\Soil-Data\sol_texture.class_usda.tt_m_250m_b0..0cm_1950..2017_v0.2.tif"

# ----------------------------
# LOAD DATASETS ONCE
# ----------------------------
sand_ds = rasterio.open(SAND_FILE)
clay_ds = rasterio.open(CLAY_FILE)
silt_ds = rasterio.open(SILT_FILE)
texture_ds = rasterio.open(TEXTURE_FILE)

# ----------------------------
# USDA Texture Class Mapping
# ----------------------------
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

# ----------------------------
# READ 1 PIXEL SAFELY
# ----------------------------
def get_pixel(ds, lat, lon):
    row, col = ds.index(lon, lat)
    window = Window(col, row, 1, 1)
    value = ds.read(1, window=window)[0, 0]

    if value == ds.nodata:
        return None
    
    return float(value)

# ----------------------------
# MAIN FUNCTION
# ----------------------------
def get_soil_physical(lat: float, lon: float):
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
        "source": "OpenLandMap 250m (0–5 cm Depth)"
    }
