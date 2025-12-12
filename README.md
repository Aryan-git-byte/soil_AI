# 🌾 FarmBot Nova - AI-Powered Agricultural Assistant

**FarmBot Nova** is an intelligent farming assistant designed specifically for Indian agriculture. It combines AI vision, conversational memory, real-time sensor data, weather intelligence, soil analysis, and **RAG-powered knowledge retrieval** to provide personalized farming advice.

---

## 🎯 Key Features

### 🤖 **AI-Powered Chat with Memory**
- Conversational AI that remembers your farming context
- Maintains separate conversation threads per user
- Supports both text and image queries in the same conversation
- Powered by Amazon Nova 2 Lite via OpenRouter API

### 📚 **RAG-Powered Knowledge Retrieval**
- **Hybrid search system** combining vector database (Qdrant) and web search (Tavily)
- Searches through curated agricultural knowledge base (ICAR, FAO, NCERT documents)
- Automatically triggers web search for recent/location-specific queries
- Cites sources in responses (e.g., "According to ICAR wheat guide...")
- **Pure semantic search** - no crop/region filtering for maximum relevance

### 📸 **Vision Analysis**
- Crop disease identification
- Plant health assessment
- Pest and weed detection
- Soil quality analysis from images
- Multi-modal conversations (text + image in same thread)

### 🌍 **Location Intelligence**
- Real-time weather data integration
- Geographic context (city, district, state)
- Nearby landmarks and agricultural monuments
- Wikipedia-powered location insights
- Smart POI detection (museums, heritage sites, monuments)

### 🌱 **Soil Analysis**
- Physical soil properties (sand, clay, silt percentages)
- USDA texture classification
- **Indian soil type classification** (Alluvial, Black/Regur, Red & Yellow, Laterite, etc.)
- Regional soil characteristics and crop recommendations
- Raster-based analysis using OpenLandMap datasets

### 📊 **Real-Time Sensor Integration**
- Soil moisture, temperature, and EC monitoring
- NPK (Nitrogen, Phosphorus, Potassium) levels
- pH monitoring
- GPS-tagged sensor data from Supabase

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (index.html)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Chat UI    │  │ Image Upload │  │  Location    │      │
│  │   + RAG UI   │  │              │  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            ↓ HTTP/REST
┌─────────────────────────────────────────────────────────────┐
│                  FastAPI Backend (Python)                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Routers (API Endpoints)                  │   │
│  │  • /api/ai/ask (unified text + image + RAG)          │   │
│  │  • /api/location/context                             │   │
│  │  • /api/ai/conversations                             │   │
│  └──────────────────────────────────────────────────────┘   │
│                            ↓                                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                   Services Layer                      │   │
│  │  • AI Service (OpenRouter/Amazon Nova 2 Lite)        │   │
│  │  • RAG Service (Qdrant + Tavily hybrid search)       │   │
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
│  • Qdrant Cloud (Vector Database for RAG)                   │
│  • Tavily API (Web Search)                                   │
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
│   │   ├── ai.py                    # Unified AI chat (text + image + RAG)
│   │   ├── location.py              # Location context API
│   │   └── image.py                 # (Legacy, standalone image analysis)
│   │
│   └── services/                    # Business Logic
│       ├── ai_service.py            # AI query processing + RAG integration
│       ├── rag_service.py           # ⭐ Hybrid RAG (Qdrant + Tavily)
│       ├── conversation_service.py  # Chat history management
│       ├── location_service.py      # Geocoding + Wikipedia
│       ├── weather_service.py       # OpenWeather integration
│       ├── soil_service.py          # Soil analysis + Indian classification
│       ├── sensor_service.py        # Supabase sensor data
│       └── image_service.py         # (Legacy image analysis)
│
├── knowledge_pipeline/              # ⭐ RAG Data Ingestion Scripts
│   ├── scripts/
│   │   └── script.ipynb             # PDF → Qdrant pipeline (URL & ZIP)
│   └── sources/
│       └── links.txt                # Processed & to-be-processed PDF links
│
├── index.html                       # Frontend chat interface with RAG UI
├── test_RAG.py                      # ⭐ RAG diagnostic script
├── requirements.txt                 # Python dependencies
├── .env                             # Environment variables (not in repo)
├── .gitignore
├── .hintrc                          # Linter config
└── README.md                        # This file
```

---

## 🚀 Getting Started

### **Prerequisites**

- Python 3.8+
- Supabase account (free tier works)
- OpenRouter API key (free tier available)
- OpenWeather API key (free tier works)
- **Qdrant Cloud account** (free tier: 1GB storage)
- **Tavily API key** (free tier: 1000 searches/month)

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
- `sentence-transformers` - Embedding model for RAG
- `qdrant-client` - Vector database client
- `pymupdf` - PDF text extraction

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

# ⭐ Qdrant Configuration (for RAG)
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION=farmbot_knowledge

# ⭐ Tavily API Key (for web search)
TAVILY_API_KEY=your_tavily_api_key
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

-- Add indexes for faster queries
CREATE INDEX idx_conversation_id ON conversation_history(conversation_id);
CREATE INDEX idx_auth_id ON conversation_history(auth_id);
```

---

### **5. Set Up RAG Knowledge Base** ⭐

#### **Step 1: Create Qdrant Collection**

Sign up at [Qdrant Cloud](https://cloud.qdrant.io/) (free tier available).

The collection will be auto-created on first run, but you can manually create it:

```python
from qdrant_client import QdrantClient, models

client = QdrantClient(url="YOUR_QDRANT_URL", api_key="YOUR_API_KEY")

client.create_collection(
    collection_name="farmbot_knowledge",
    vectors_config=models.VectorParams(
        size=384,  # MiniLM embedding dimension
        distance=models.Distance.COSINE
    )
)
```

#### **Step 2: Ingest Knowledge**

Use the provided Jupyter notebook in `knowledge_pipeline/scripts/script.ipynb`:

**Option A: Ingest from URLs**
```python
# Paste PDF links (one per line)
links = """
https://ncert.nic.in/textbook/pdf/iess204.pdf
https://www.fao.org/4/i2800e/i2800e07.pdf
...
"""
# Script will download → extract text → chunk → embed → upload to Qdrant
```

**Option B: Ingest from ZIP file**
```python
# Extract PDFs from ZIP → process → upload
ZIP_PATH = "/path/to/publications.zip"
```

The notebook handles:
- PDF download/extraction
- Text cleaning
- Chunking (800 words per chunk)
- Embedding generation (MiniLM-L6-v2)
- Batch upload to Qdrant

See `knowledge_pipeline/sources/links.txt` for processed sources.

#### **Step 3: Verify RAG Setup**

Run the diagnostic script:

```bash
python test_RAG.py
```

This checks:
- ✅ Environment variables
- ✅ Qdrant connection
- ✅ Embedding model
- ✅ Vector search
- ✅ Tavily web search
- ✅ Hybrid search pipeline

---

### **6. Download Soil Raster Data (Optional but Recommended)**

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

### **7. Run the Backend**

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Server will start at: `http://localhost:8000`

---

### **8. Open the Frontend**

Simply open `index.html` in your browser, or serve it with:

```bash
# Using Python's built-in server
python -m http.server 3000
```

Then visit: `http://localhost:3000`

---

## 🔌 API Endpoints

### **1. Unified AI Chat** (Text + Image + RAG)
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
  "had_image": true,
  "rag_info": {
    "success": true,
    "rag_chunks_count": 8,
    "web_results_count": 3,
    "used_web_search": true,
    "rag_sources": [...],
    "web_sources": [...]
  }
}
```

### **2. Get User Conversations**
```http
GET /api/ai/conversations?auth_id=user123
```

### **3. Get Conversation History**
```http
GET /api/ai/conversation/history?conversation_id=conv_user123_abc456&limit=50
```

### **4. Delete Conversation**
```http
DELETE /api/ai/conversation?conversation_id=conv_user123_abc456
```

### **5. Get Location Context**
```http
GET /api/location/context?lat=25.5941&lon=85.1376
```

---

## 🧠 How RAG Works

### **Hybrid Search Pipeline**

```
User Query: "Best wheat varieties for Bihar"
                    ↓
        ┌───────────────────────┐
        │  Query Analysis       │
        │  - Detect location    │
        │  - Detect time refs   │
        └───────────────────────┘
                    ↓
        ┌───────────────────────┐
        │  Vector Search        │
        │  (Qdrant)             │
        │  - Embed query        │
        │  - Semantic search    │
        │  - No filters         │
        │  → Top 8 chunks       │
        └───────────────────────┘
                    ↓
        ┌───────────────────────┐
        │  Decision Logic       │
        │  Trigger web if:      │
        │  - Location keywords  │
        │  - Time keywords      │
        │  - RAG count < 3      │
        └───────────────────────┘
                    ↓
        ┌───────────────────────┐
        │  Web Search           │
        │  (Tavily)             │
        │  - Agricultural sites │
        │  - Recent results     │
        │  → Top 5 articles     │
        └───────────────────────┘
                    ↓
        ┌───────────────────────┐
        │  Context Assembly     │
        │  - Format RAG chunks  │
        │  - Format web results │
        │  - Cite sources       │
        └───────────────────────┘
                    ↓
        ┌───────────────────────┐
        │  AI Response          │
        │  (Amazon Nova 2 Lite) │
        │  - Uses all context   │
        │  - Cites sources      │
        │  - Actionable advice  │
        └───────────────────────┘
```

### **RAG Configuration**

Edit `app/services/rag_service.py`:

```python
# Vector search parameters
top_k = 8                    # Number of chunks to retrieve
similarity_threshold = 0.25  # Minimum relevance score

# Web search parameters
top_k_web = 5               # Number of web results
search_depth = "basic"      # or "advanced"

# Hybrid search triggers
force_web = False           # Always use web
rag_threshold = 3           # Minimum RAG results before web search
```

---

## 🎨 Frontend Features

### **Chat Interface**
- Dark theme optimized for readability
- Markdown support (headings, lists, code blocks)
- **RAG source cards** showing knowledge origins
- **Web result cards** with clickable links
- Collapsible context data viewer
- Image preview before sending
- Real-time message updates

### **RAG Information Display**
```
🔍 Knowledge Sources
[RAG: 8] [Web: 3]

📚 ICAR Wheat Cultivation Guide
Crop: Wheat | Region: Bihar | Relevance: 87.3%
"Wheat varieties HD-2967 and HD-3086 are recommended for Bihar's alluvial soil..."

🌐 Latest Agricultural News - KVK Bihar
https://kvk-bihar.org/wheat-varieties-2024
"New high-yielding wheat varieties released for 2024-25 season..."
```

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

### **Adjust RAG Parameters**
Edit `app/services/rag_service.py`:
```python
similarity_threshold = 0.25  # Lower = more results, higher = more precise
top_k_rag = 8               # Number of knowledge chunks
top_k_web = 5               # Number of web results
```

### **Modify System Prompt**
Edit `SYSTEM_PROMPT` in `app/services/ai_service.py`:
```python
SYSTEM_PROMPT = """
You are FarmBot Nova — an agricultural assistant for Indian farming.
...
When citing knowledge, mention the source (e.g., "According to ICAR...").
"""
```

---

## 🐛 Troubleshooting

### **RAG Issues**

**"No RAG results found"**
- Run `python test_RAG.py` to diagnose
- Check if Qdrant collection has data: `qdrant.count("farmbot_knowledge")`
- Lower similarity threshold in `rag_service.py`
- Re-run knowledge ingestion pipeline

**"All searches returning low similarity"**
- Knowledge base may not cover the topic
- Try more specific queries
- Add more documents to knowledge base
- Check if embeddings were generated correctly

**"Tavily search failing"**
- Verify `TAVILY_API_KEY` in `.env`
- Check API quota (free tier: 1000/month)
- Ensure internet connectivity

### **General Issues**

**"No sensor data available"**
- Check Supabase connection
- Verify `sensor_data` table has records
- Or provide `lat` and `lon` in the request

**"All OpenRouter keys failed"**
- Verify API keys in `.env`
- Check OpenRouter account limits
- Ensure internet connectivity

**"Soil data not available"**
- Download raster files from OpenLandMap
- Update file paths in `soil_service.py`
- Or continue without soil data (RAG will compensate)

**CORS Errors**
- Check `allow_origins` in `main.py`
- In production, replace `["*"]` with your domain

---

## 📊 Knowledge Base Sources

Current sources (see `knowledge_pipeline/sources/links.txt`):

- **ICAR Publications** - Crop-specific cultivation guides
- **FAO Reports** - Food and agriculture statistics
- **NCERT Textbooks** - Agricultural science chapters
- **State Agricultural Departments** - Regional guidelines
- **Research Papers** - IJHSSI, JETIR, IOSR journals
- **Bihar-Specific** - KVK reports, state profiles

**Total chunks ingested: Check with `test_RAG.py`**

---

## 🚧 Roadmap

### **Core Features**
- [x] Multi-turn conversations with memory
- [x] Vision analysis (crop diseases, pests)
- [x] RAG-powered knowledge retrieval
- [x] Hybrid search (vector + web)
- [x] Source citation in responses
- [x] Indian soil classification

### **Planned Enhancements**
- [ ] Multi-language support (Hindi, Tamil, Telugu)
- [ ] Voice input/output
- [ ] Crop price prediction
- [ ] Disease outbreak alerts
- [ ] Mobile app (React Native)
- [ ] WhatsApp bot integration
- [ ] Offline mode with cached data
- [ ] Advanced RAG with re-ranking
- [ ] User feedback loop for knowledge base

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### **Contributing to Knowledge Base**

To add new agricultural knowledge:

1. Add PDF links to `knowledge_pipeline/sources/links.txt`
2. Run the ingestion notebook in `knowledge_pipeline/scripts/`
3. Test with `python test_RAG.py`
4. Submit PR with updated `links.txt`

---

## 🙏 Acknowledgments

- **OpenRouter** - AI API aggregation
- **Amazon Nova 2 Lite** - Vision-capable AI model
- **Qdrant** - Vector database for RAG
- **Tavily** - Web search API
- **Sentence Transformers** - Embedding models
- **Supabase** - Backend-as-a-Service
- **OpenWeather** - Weather data API
- **OpenLandMap** - Global soil property maps
- **OpenStreetMap** - Geocoding services
- **Wikipedia/Wikidata** - Location context
- **ICAR, FAO, NCERT** - Agricultural knowledge sources

---

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Email: support@farmbotnova.com
- Documentation: [docs.farmbotnova.com](https://docs.farmbotnova.com)

---

## 🔍 Keywords

Agricultural AI, RAG, Vector Search, Conversational AI, Crop Disease Detection, Soil Analysis, Indian Farming, Smart Agriculture, Knowledge Retrieval, Hybrid Search, Semantic Search

---

## ⭐ Star History

If you find FarmBot Nova useful, please consider giving it a star! ⭐

---

**Built with ❤️ for Indian farmers**

🌾 Happy Farming! 🚜