# 🌾 FarmBot Nova - AI-Powered Agricultural Assistant

**FarmBot Nova** is an intelligent farming assistant designed specifically for Indian agriculture. It combines AI vision, conversational memory, real-time sensor data, weather intelligence, and soil analysis to provide personalized farming advice.

---

## 🎯 Key Features

### 🤖 **AI-Powered Chat with Memory**
- Conversational AI that remembers your farming context
- Maintains separate conversation threads per user
- Supports both text and image queries in the same conversation

### 📸 **Vision Analysis**
- Crop disease identification
- Plant health assessment
- Pest and weed detection
- Soil quality analysis from images

### 🌍 **Location Intelligence**
- Real-time weather data integration
- Geographic context (city, district, state)
- Nearby landmarks and agricultural monuments
- Wikipedia-powered location insights

### 🌱 **Soil Analysis**
- Physical soil properties (sand, clay, silt percentages)
- USDA texture classification
- **Indian soil type classification** (Alluvial, Black/Regur, Red & Yellow, Laterite, etc.)
- Regional soil characteristics and crop recommendations

### 📊 **Real-Time Sensor Integration**
- Soil moisture, temperature, and EC monitoring
- NPK (Nitrogen, Phosphorus, Potassium) levels
- pH monitoring
- GPS-tagged sensor data

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (index.html)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Chat UI    │  │ Image Upload │  │  Location    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            ↓ HTTP/REST
┌─────────────────────────────────────────────────────────────┐
│                  FastAPI Backend (Python)                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Routers (API Endpoints)                  │   │
│  │  • /api/ai/ask (unified text + image)                │   │
│  │  • /api/location/context                             │   │
│  │  • /api/ai/conversations                             │   │
│  └──────────────────────────────────────────────────────┘   │
│                            ↓                                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                   Services Layer                      │   │
│  │  • AI Service (OpenRouter/Amazon Nova 2 Lite)        │   │
│  │  • Location Service (OSM, Wikipedia, Wikidata)       │   │
│  │  • Weather Service (OpenWeather API)                 │   │
│  │  • Soil Service (Raster datasets + classification)   │   │
│  │  • Sensor Service (Supabase)                         │   │
│  │  • Conversation Service (Supabase)                   │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    External Services                         │
│  • OpenRouter API (AI Vision + Text)                        │
│  • Supabase (Database for sensors + conversations)          │
│  • OpenWeather API (Real-time weather)                      │
│  • OpenStreetMap (Geocoding)                                │
│  • Wikipedia/Wikidata (Location context)                    │
│  • Local Raster Files (Soil data)                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
FarmBot-Nova/
│
├── app/
│   ├── __init__.py
│   │
│   ├── main.py                      # FastAPI app entry point + CORS
│   │
│   ├── core/
│   │   └── config.py                # Environment variables & API keys
│   │
│   ├── routers/                     # API Endpoints
│   │   ├── ai.py                    # Unified AI chat (text + image)
│   │   ├── location.py              # Location context API
│   │   └── image.py                 # (Legacy, can be removed)
│   │
│   └── services/                    # Business Logic
│       ├── ai_service.py            # AI query processing
│       ├── conversation_service.py  # Chat history management
│       ├── location_service.py      # Geocoding + Wikipedia
│       ├── weather_service.py       # OpenWeather integration
│       ├── soil_service.py          # Soil analysis + classification
│       ├── sensor_service.py        # Supabase sensor data
│       └── image_service.py         # (Legacy, can be removed)
│
├── index.html                       # Frontend chat interface
├── requirements.txt                 # Python dependencies
├── .env                             # Environment variables (not in repo)
├── .gitignore
└── README.md                        # This file
```

---

## 🚀 Getting Started

### **Prerequisites**

- Python 3.8+
- Node.js (for frontend, optional)
- Supabase account (free tier works)
- OpenRouter API key (free tier available)
- OpenWeather API key (free tier works)

---

### **1. Clone the Repository**

```bash
git clone https://github.com/yourusername/farmbot-nova.git
cd farmbot-nova
```

---

### **2. Install Dependencies**

```bash
pip install -r requirements.txt
```

**Dependencies:**
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `httpx` - Async HTTP client
- `python-dotenv` - Environment variable management
- `supabase` - Database client
- `rasterio` - Raster data processing (soil datasets)
- `requests` - HTTP requests

---

### **3. Set Up Environment Variables**

Create a `.env` file in the root directory:

```env
# OpenRouter API Keys (supports multiple for fallback)
OPENROUTER_API_KEY_1=your_openrouter_key_1
OPENROUTER_API_KEY_2=your_openrouter_key_2
OPENROUTER_API_KEY_3=your_openrouter_key_3

# OpenWeather API Key
OPENWEATHER_API_KEY=your_openweather_key

# Supabase Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_anon_key
```

---

### **4. Set Up Supabase Database**

Create two tables in your Supabase project:

#### **Table 1: `sensor_data`**
```sql
CREATE TABLE sensor_data (
    id SERIAL PRIMARY KEY,
    latitude NUMERIC NOT NULL,
    longitude NUMERIC NOT NULL,
    soil_moisture NUMERIC,
    ec NUMERIC,
    soil_temperature NUMERIC,
    n NUMERIC,
    p NUMERIC,
    k NUMERIC,
    ph NUMERIC,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);
```

#### **Table 2: `conversation_history`**
```sql
CREATE TABLE conversation_history (
    id SERIAL PRIMARY KEY,
    auth_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,  -- 'user' or 'assistant'
    content TEXT NOT NULL,
    metadata JSONB,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

-- Add index for faster queries
CREATE INDEX idx_conversation_id ON conversation_history(conversation_id);
CREATE INDEX idx_auth_id ON conversation_history(auth_id);
```

---

### **5. Download Soil Raster Data (Optional but Recommended)**

Download soil datasets from [OpenLandMap](https://openlandmap.org/):

- `sand.wfraction_usda.3a1a1a_m_250m_b0cm_19500101_20171231_go_epsg.4326_v0.2.tif`
- `sol_clay.wfraction_usda.3a1a1a_m_250m_b0..0cm_1950..2017_v0.2.tif`
- `sol_silt.wfraction_usda.3a1a1a_m_250m_b0..0cm_1950..2017_v0.2.tif`
- `sol_texture.class_usda.tt_m_250m_b0..0cm_1950..2017_v0.2.tif`

Update file paths in `app/services/soil_service.py`:

```python
SAND_FILE = r"path/to/sand.tif"
CLAY_FILE = r"path/to/clay.tif"
SILT_FILE = r"path/to/silt.tif"
TEXTURE_FILE = r"path/to/texture.tif"
```

---

### **6. Run the Backend**

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Server will start at: `http://localhost:8000`

---

### **7. Open the Frontend**

Simply open `index.html` in your browser, or serve it with:

```bash
# Using Python's built-in server
python -m http.server 3000
```

Then visit: `http://localhost:3000`

---

## 🔌 API Endpoints

### **1. Unified AI Chat** (Text + Image)
```http
POST /api/ai/ask
Content-Type: multipart/form-data

Parameters:
- query (string, required): User's question
- auth_id (string, required): User identifier
- conversation_id (string, optional): Conversation thread ID
- lat (float, optional): Latitude override
- lon (float, optional): Longitude override
- image (file, optional): Image for analysis

Response:
{
  "answer": "AI response with farming advice",
  "context_used": { ... },
  "conversation_id": "conv_user123_abc456",
  "message_count": 5,
  "had_image": true
}
```

### **2. Get User Conversations**
```http
GET /api/ai/conversations?auth_id=user123

Response:
{
  "auth_id": "user123",
  "conversations": [
    {
      "conversation_id": "conv_user123_abc456",
      "last_message": "2024-12-07T10:30:00Z"
    }
  ],
  "total": 1
}
```

### **3. Get Conversation History**
```http
GET /api/ai/conversation/history?conversation_id=conv_user123_abc456&limit=50

Response:
{
  "conversation_id": "conv_user123_abc456",
  "messages": [
    {
      "role": "user",
      "content": "What crops grow in clay soil?",
      "timestamp": "2024-12-07T10:25:00Z"
    },
    {
      "role": "assistant",
      "content": "Clay soil is excellent for rice, wheat...",
      "timestamp": "2024-12-07T10:25:05Z"
    }
  ],
  "total": 2
}
```

### **4. Delete Conversation**
```http
DELETE /api/ai/conversation?conversation_id=conv_user123_abc456

Response:
{
  "success": true,
  "conversation_id": "conv_user123_abc456",
  "message": "Conversation deleted"
}
```

### **5. Get Location Context**
```http
GET /api/location/context?lat=25.5941&lon=85.1376

Response:
{
  "generated_at": "2024-12-07T10:30:00Z",
  "coordinates": {"lat": 25.5941, "lon": 85.1376},
  "location_info": {
    "city": "Patna",
    "district": "Patna",
    "state": "Bihar",
    "country": "India"
  },
  "nearest_place": "Patna Museum",
  "description": "Patna is the capital of Bihar...",
  "weather": {
    "temperature": 28,
    "humidity": 65,
    "weather": "Clear"
  },
  "soil_physical": {
    "sand_percent": 35.2,
    "clay_percent": 28.5,
    "silt_percent": 36.3,
    "texture": "Loam"
  },
  "indian_soil_classification": {
    "indian_soil_type": "Alluvial Soil",
    "confidence": "High",
    "description": "Fertile soil deposited by rivers...",
    "characteristics": [...]
  },
  "nearby_monuments": ["Patna Museum", "Golghar", ...]
}
```

---

## 🧠 How It Works

### **Conversation Flow**

```
1. User enters Auth ID → Creates/resumes user session
2. User types question OR uploads image
3. Frontend sends to /api/ai/ask (always same endpoint)
4. Backend:
   a. Fetches sensor data from Supabase
   b. Gets location context (weather, soil, Wikipedia)
   c. Retrieves conversation history
   d. Builds AI prompt with full context
   e. Sends to Amazon Nova 2 Lite via OpenRouter
   f. Saves user message + AI response to database
5. Frontend displays answer with markdown formatting
6. User continues conversation (AI remembers context)
```

---

### **Indian Soil Classification Logic**

FarmBot uses a **multi-factor classification system**:

1. **Physical Properties:** Sand, clay, silt percentages from raster data
2. **Geographic Location:** Lat/lon mapped to Indian regions
3. **USDA Texture Class:** Loam, clay, sandy loam, etc.

**Example Classification:**
```python
Input:
- Clay: 42%
- Sand: 25%
- Silt: 33%
- Location: Deccan Plateau (Lat: 18.5, Lon: 77.2)

Output:
- Type: "Black Soil (Regur)"
- Confidence: "High"
- Characteristics: [
    "Ideal for cotton cultivation",
    "High water retention capacity",
    "Rich in calcium and magnesium"
  ]
```

---

## 🎨 Frontend Features

### **Chat Interface**
- Dark theme optimized for readability
- Markdown support (headings, lists, code blocks)
- Collapsible context data viewer
- Image preview before sending
- Real-time message updates

### **User Experience**
- Auth ID management (persist across sessions)
- Multiple conversation threads
- Image + text in same conversation
- Location override option
- Keyboard shortcuts (Ctrl+Enter to send)

---

## 🔧 Configuration & Customization

### **Change AI Model**
Edit `app/services/ai_service.py`:
```python
"model": "amazon/nova-2-lite-v1:free"  # Change to any OpenRouter model
```

### **Adjust Context Size**
Edit conversation history limit:
```python
history = await conversation_service.get_conversation_history(
    conversation_id, 
    limit=20  # Change to 10, 50, etc.
)
```

### **Modify System Prompt**
Edit `SYSTEM_PROMPT` in `app/services/ai_service.py` to change AI behavior.

---

## 🐛 Troubleshooting

### **"No sensor data available"**
- Check Supabase connection
- Verify `sensor_data` table has records
- Or provide `lat` and `lon` in the request

### **"All OpenRouter keys failed"**
- Verify API keys in `.env`
- Check OpenRouter account limits
- Ensure internet connectivity

### **"Soil data not available"**
- Download raster files from OpenLandMap
- Update file paths in `soil_service.py`
- Or continue without soil data (system will work but with limited features)

### **CORS Errors**
- Check `allow_origins` in `main.py`
- In production, replace `["*"]` with your domain

---

## 📊 Database Schema

### **conversation_history**
| Column | Type | Description |
|--------|------|-------------|
| id | SERIAL | Primary key |
| auth_id | TEXT | User identifier |
| conversation_id | TEXT | Thread identifier |
| role | TEXT | 'user' or 'assistant' |
| content | TEXT | Message content |
| metadata | JSONB | Context data, coordinates, image info |
| timestamp | TIMESTAMPTZ | Message creation time |

### **sensor_data**
| Column | Type | Description |
|--------|------|-------------|
| id | SERIAL | Primary key |
| latitude | NUMERIC | GPS latitude |
| longitude | NUMERIC | GPS longitude |
| soil_moisture | NUMERIC | % moisture |
| ec | NUMERIC | Electrical conductivity |
| soil_temperature | NUMERIC | °C |
| n, p, k | NUMERIC | NPK levels |
| ph | NUMERIC | Soil pH |
| timestamp | TIMESTAMPTZ | Reading time |

---

## 🌟 Advanced Features

### **Multi-Turn Conversations**
```
User: "What's the best crop for my region?"
Bot: "Based on your alluvial soil in Patna, rice and wheat are ideal..."

User: [uploads image of wheat]
Bot: "Your wheat looks healthy! The growth stage suggests..."

User: "When should I harvest?"
Bot: "Based on the wheat we discussed, harvest in 3-4 weeks..."
```

### **Context-Aware Responses**
The AI considers:
- Previous messages in conversation
- Current weather conditions
- Soil type and texture
- Regional characteristics
- Sensor readings (if available)
- Uploaded images

---

## 🚧 Roadmap

- [ ] Multi-language support (Hindi, Tamil, Telugu, etc.)
- [ ] Voice input/output
- [ ] Crop price prediction
- [ ] Disease outbreak alerts
- [ ] Mobile app (React Native)
- [ ] WhatsApp bot integration
- [ ] Offline mode with cached data

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **OpenRouter** - AI API aggregation
- **Amazon Nova 2 Lite** - Vision-capable AI model
- **Supabase** - Backend-as-a-Service
- **OpenWeather** - Weather data API
- **OpenLandMap** - Global soil property maps
- **OpenStreetMap** - Geocoding services
- **Wikipedia/Wikidata** - Location context

---

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Email: support@farmbotnova.com
- Documentation: [docs.farmbotnova.com](https://docs.farmbotnova.com)

---

## ⭐ Star History

If you find FarmBot Nova useful, please consider giving it a star! ⭐

---

**Built with ❤️ for Indian farmers**

🌾 Happy Farming! 🚜