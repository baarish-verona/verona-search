# Verona AI Search - Architecture

## Overview

Verona AI Search is a profile ingestion and semantic search system for matrimonial matching. It stores user profiles in Qdrant vector database with multi-vector embeddings and generates AI-powered "vibe reports" using GPT-4o vision.

### System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           VERONA AI SEARCH                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────────┐    ┌─────────────────────────────┐  │
│  │  API Layer  │    │  Ingest Service │    │  Search Service             │  │
│  │             │    │                 │    │                             │  │
│  │ POST /ingest│───▶│ - Transform     │    │ - Query Parser (GPT-4o-mini)│  │
│  │ POST /search│    │ - Smart Update  │    │ - Multi-Vector Search       │  │
│  │ POST /parse │    │ - Vibe Generate │    │ - Filter Engine             │  │
│  └─────────────┘    └────────┬────────┘    └──────────────┬──────────────┘  │
│                              │                            │                 │
│                              ▼                            ▼                 │
│                     ┌─────────────────────────────────────────────┐         │
│                     │              Embedding Layer                │         │
│                     │                                             │         │
│                     │  ┌───────────────┐  ┌───────────────────┐   │         │
│                     │  │ OpenAI Small  │  │ BGE-M3 ColBERT    │   │         │
│                     │  │ (1536 dim)    │  │ (1024 × N tokens) │   │         │
│                     │  └───────────────┘  └───────────────────┘   │         │
│                     └─────────────────────────────────────────────┘         │
│                                        │                                    │
│                                        ▼                                    │
│                     ┌─────────────────────────────────────────────┐         │
│                     │           Qdrant Vector Store               │         │
│                     │                                             │         │
│                     │  Named Vectors: education, profession,      │         │
│                     │                 vibe_report, blurb_colbert  │         │
│                     └─────────────────────────────────────────────┘         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                     ┌─────────────────────────────────────────────┐
                     │              External Services              │
                     │                                             │
                     │  ┌───────────────┐  ┌───────────────────┐   │
                     │  │ OpenAI API    │  │ CloudFront CDN    │   │
                     │  │ (GPT-4o)      │  │ (Photo URLs)      │   │
                     │  └───────────────┘  └───────────────────┘   │
                     └─────────────────────────────────────────────┘
```

### Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI (Python 3.11+) |
| Vector DB | Qdrant |
| Embeddings | OpenAI text-embedding-3-small |
| ColBERT | BGE-M3 (late interaction) |
| Vibe Generation | OpenAI GPT-4o (vision) |
| Query Parser | OpenAI GPT-4o-mini |

---

## Directory Structure

```
verona-ai-search/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI application entry
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py            # Environment-based settings
│   │   └── embedding_specs.py     # Vector configuration
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py              # API endpoints
│   │   └── dependencies.py        # FastAPI dependencies
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── ingest.py              # IngestUserProfile (input)
│   │   ├── object.py              # User (Qdrant storage)
│   │   └── responses.py           # API response models
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── ingest_service.py      # Profile ingestion with smart updates
│   │   ├── vibe_service.py        # GPT-4o vibe report generation
│   │   ├── search_service.py      # Search orchestration
│   │   └── query_parser.py        # LLM query parsing
│   │
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── base.py                # Abstract provider interface
│   │   ├── openai_provider.py     # OpenAI embeddings
│   │   ├── bge_colbert.py         # BGE-M3 ColBERT
│   │   └── factory.py             # Provider factory
│   │
│   └── vector_store/
│       ├── __init__.py
│       ├── qdrant_client.py       # Qdrant operations
│       ├── query_builder.py       # Dynamic query building
│       └── filters.py             # Filter utilities
│
├── scripts/
│   ├── ingest_profiles.py         # Batch ingestion
│   └── populate_vectors.py        # Regenerate vectors
│
├── .env.example
├── requirements.txt
└── ARCHITECTURE.md
```

---

## Data Models

### IngestUserProfile (Input Model)

Input model for profile ingestion. Accepts camelCase via aliases for TypeScript compatibility.

```python
class IngestUserProfile(BaseModel):
    # Required identifier (MongoDB _id)
    id: str = Field(..., alias="_id")

    # Status flags (for is_circulateable computation)
    is_ql: bool = Field(False, alias="isQL")
    is_active: bool = Field(False, alias="isActive")
    is_verified: bool = Field(False, alias="isVerified")
    is_non_serviceable: bool = Field(False, alias="isNonServiceable")
    is_soft_deleted: bool = Field(False, alias="isSoftDeleted")
    pause_details: Optional[PauseDetails] = Field(None, alias="pauseDetails")
    onboarded_on: Optional[datetime] = Field(None, alias="onboardedOn")
    test_lead: bool = Field(False, alias="testLead")

    # Control flag
    force_update_vector_profile: bool = Field(False, alias="forceUpdateVectorProfile")

    # Demographics
    gender: str
    height: int
    dob: str
    current_location: str = Field(..., alias="currentLocation")
    annual_income: Optional[int] = Field(None, alias="annualIncome")

    # Filter fields
    religion: str
    caste: str
    fitness: str
    religiosity: str
    smoking: str
    drinking: str
    food_habits: str = Field(..., alias="foodHabits")
    intent: str
    open_to_children: str = Field(..., alias="openToChildren")
    family_type: Optional[str] = Field(None, alias="familyType")

    # App details
    app_version_details: AppVersionDetails = Field(..., alias="appVersionDetails")

    # Text fields for embedding
    education_details: Optional[List[EducationDetail]] = Field(None, alias="educationDetails")
    professional_journey_details: Optional[List[ProfessionalJourneyDetail]] = Field(None, alias="professionalJourneyDetails")
    highlighted_professional_detail_id: Optional[str] = Field(None, alias="highlightedProfessionalDetailId")
    similar_interests_v2: Optional[List[str]] = Field(None, alias="similarInterestsV2")
    blurb: Optional[str] = None

    # Photos
    photo_collection: Optional[List[PhotoDoc]] = Field(None, alias="photoCollection")
    show_case_profile_ids: Optional[List[str]] = Field(None, alias="showCaseProfileIds")
```

### User (Qdrant Storage Model)

Simplified model stored in Qdrant with computed/derived fields.

```python
class User(BaseModel):
    id: Optional[str] = None
    is_circulateable: bool = False      # Computed from status flags
    is_paused: bool = False
    last_active: Optional[datetime] = None

    # Demographics
    gender: str
    height: int
    dob: str
    current_location: str
    annual_income: Optional[int] = None

    # Filter fields
    religion: str
    caste: str
    fitness: str
    religiosity: str
    smoking: str
    drinking: str
    family_type: Optional[str] = None
    food_habits: str
    intent: str
    open_to_children: str

    # Derived text fields (with vectors)
    profession: Optional[str] = None          # "{designation} at {company}"
    profession_hash: Optional[str] = None     # MD5 for change detection
    education: Optional[str] = None           # "{degree} from {college}; ..."
    education_hash: Optional[str] = None
    vibe_report: Optional[str] = None         # Generated by GPT-4o
    vibe_report_hash: Optional[str] = None

    # Content fields
    blurb: Optional[str] = None
    profile_hook: Optional[str] = None        # Trump-Adams summary
    life_style_tags: List[str] = []
    interests: List[str] = []
    photo_collection: List[ProcessedPhoto] = []
```

### ProcessedPhoto

```python
class ProcessedPhoto(BaseModel):
    show_case_id: str
    url: str                    # CloudFront URL
    cropped_url: Optional[str] = None
```

---

## Computed Fields

### is_circulateable

Determines if a profile should appear in search results:

```python
is_circulateable = (
    is_ql
    AND is_active
    AND is_verified
    AND onboarded_on is not None
    AND NOT is_non_serviceable
    AND NOT is_soft_deleted
    AND NOT pause_details.is_paused
    AND NOT test_lead
)
```

### profession

Built from `professional_journey_details`:
- Uses `highlighted_professional_detail_id` if specified
- Otherwise uses the last element in the list
- Format: `"{designation} at {company}"`

### education

Built from `education_details`:
- Concatenates all entries
- Format: `"{degree} from {college}; {degree2} from {college2}"`

### photo_collection

Built from input `photo_collection` and `show_case_profile_ids`:
1. Filter out removed photos (`is_removed = true`)
2. Only include photos in `show_case_profile_ids`
3. Match by `show_case_id` or `media_id` (when `media_type = IMAGE_JPEG`)
4. Prepend CloudFront URL based on environment

---

## Profile Ingestion

### Ingest Flow

```
┌─────────────────────┐
│  IngestUserProfile  │  (camelCase JSON from API)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ User.from_ingest    │  Transform:
│    _profile()       │  - Compute is_circulateable
│                     │  - Build profession/education strings
│                     │  - Build photo_collection with CloudFront URLs
│                     │  - Compute MD5 hashes
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  IngestService      │  Fetch existing profile from Qdrant
│    .ingest()        │
└──────────┬──────────┘
           │
     ┌─────┴─────┐
     │           │
     ▼           ▼
┌─────────┐ ┌─────────────┐
│ New or  │ │  Existing   │
│ Forced  │ │  Profile    │
└────┬────┘ └──────┬──────┘
     │             │
     ▼             ▼
┌─────────┐ ┌─────────────┐
│  Full   │ │   Smart     │  Compare hashes,
│ Upsert  │ │   Update    │  update only changed
└────┬────┘ └──────┬──────┘
     │             │
     └──────┬──────┘
            │
            ▼
┌─────────────────────┐
│    VibeService      │  (if vibe content changed)
│ .generate_vibe_map  │  GPT-4o vision
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  EmbeddingProvider  │  OpenAI text-embedding-3-small
│     .embed()        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ QdrantVectorStore   │
│  .upsert_points()   │  Full insert
│  .set_payload()     │  Partial payload update
│  .update_vectors()  │  Vector-only update
└─────────────────────┘
```

### Smart Update Logic

Uses MD5 hashes to detect changes and minimize operations:

| Field | Hash Field | Action on Change |
|-------|------------|------------------|
| `education` | `education_hash` | Regenerate education embedding |
| `profession` | `profession_hash` | Regenerate profession embedding |
| `vibe_report` | `vibe_report_hash` | (Generated, not input) |

### Vibe Report Regeneration Triggers

Vibe report is regenerated when any of these change:
- `education_hash` changed
- `profession_hash` changed
- `blurb` changed
- `interests` changed
- `photo_collection` changed (different photo IDs)
- No existing `vibe_report`

### last_active Update Rules

To avoid excessive updates for minor timestamp changes:

| Condition | Action |
|-----------|--------|
| Other fields changed | Update `last_active` if changed |
| Only `last_active` changed | Update only if difference > 2 hours |

---

## Vibe Report Generation

Uses OpenAI GPT-4o with vision to analyze profile data and photos.

### Input

```json
{
  "education": "MBA from IIM Bangalore",
  "profession": "Director at Google",
  "photos": [
    {"id": "photo_1", "url": "https://cloudfront.../photo1.jpg"}
  ],
  "interests": ["Travel", "Music", "Photography"],
  "blurb": "Love exploring new places and meeting new people..."
}
```

### Output

```json
{
  "vibeReport": "A 2-3 paragraph character study synthesizing the user's professional shell with their cultural anchors...",
  "trumpAdamsSummary": "A total game-changer. Spectacular alignment of analytical rigor and creative soul...",
  "imageTags": {
    "photo_1": ["#LinenSeason", "#OldMoneyAesthetic", "#IntellectualInvestigator"]
  }
}
```

### Fields Stored

| Field | Description |
|-------|-------------|
| `vibe_report` | Full character study (2-3 paragraphs) |
| `vibe_report_hash` | MD5 hash for change detection |
| `profile_hook` | Trump-Adams summary (3-4 lines) |

---

## Vector Configuration

### Named Vectors in Qdrant

```python
VECTOR_CONFIG = {
    "education": {
        "provider": "openai-small",
        "dim": 1536,
        "distance": "Cosine"
    },
    "profession": {
        "provider": "openai-small",
        "dim": 1536,
        "distance": "Cosine"
    },
    "vibe_report": {
        "provider": "openai-small",
        "dim": 1536,
        "distance": "Cosine"
    },
    "blurb_colbert": {
        "provider": "bge-colbert",
        "dim": 1024,
        "type": "multivector",
        "comparator": "MaxSim"
    }
}
```

### Embedding Providers

| Provider | Model | Dimensions | Used For |
|----------|-------|------------|----------|
| `openai-small` | text-embedding-3-small | 1536 | education, profession, vibe_report |
| `bge-colbert` | BAAI/bge-m3 | 1024 × N | blurb (late interaction) |

---

## API Reference

### POST /api/ingest

Ingest a single user profile.

**Request:** `IngestUserProfile` (camelCase JSON)

**Response:** `"Ok"`

**Example:**

```bash
curl -X POST http://localhost:8000/api/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "_id": "68f0cca72aa8021bc6fb544f",
    "isQL": true,
    "isActive": true,
    "isVerified": true,
    "onboardedOn": "2024-01-01T00:00:00Z",
    "gender": "male",
    "height": 72,
    "dob": "1990-01-01",
    "currentLocation": "Mumbai",
    "religion": "HI",
    "caste": "XX",
    "fitness": "ER",
    "religiosity": "MO",
    "smoking": "NS",
    "drinking": "DD",
    "foodHabits": "VGT",
    "intent": "01",
    "openToChildren": "OC",
    "appVersionDetails": {"lastUpdatedOn": "2024-01-01T00:00:00Z"},
    "educationDetails": [
      {"id": "edu1", "degree": "B.Tech", "college": "IIT Delhi"}
    ],
    "professionalJourneyDetails": [
      {"id": "prof1", "designation": "Engineer", "company": "Google"}
    ],
    "blurb": "Love traveling and exploring new places"
  }'
```

### POST /api/search

Execute semantic search with filters.

**Request:**

```json
{
  "query": "Female engineer from IIT aged 25-30",
  "filters": {
    "min_age": 25,
    "max_age": 30
  },
  "limit": 50,
  "offset": 0
}
```

**Response:**

```json
{
  "query": "Female engineer from IIT aged 25-30",
  "parsed": {
    "education_query": "IIT",
    "profession_query": "engineer"
  },
  "results": [
    {
      "id": "user_123",
      "score": 0.85,
      "payload": {
        "id": "68f0cca72aa8021bc6fb544f",
        "education": "B.Tech from IIT Delhi",
        "profession": "Software Engineer at Google"
      }
    }
  ],
  "total_count": 42,
  "vectors_used": ["education(openai)", "profession(openai)"],
  "filters_applied": {"min_age": 25, "max_age": 30},
  "search_time_ms": 45.2
}
```

### POST /api/parse

Parse natural language query into structured format.

**Request:**

```json
{
  "query": "Female engineer aged 25-30 from IIT who loves traveling"
}
```

**Response:**

```json
{
  "original_query": "Female engineer aged 25-30 from IIT who loves traveling",
  "filters": {
    "genders": ["female"],
    "min_age": 25,
    "max_age": 30
  },
  "education_query": "IIT",
  "profession_query": "engineer",
  "interests_query": "loves traveling",
  "blurb_query": ""
}
```

### GET /api/health

Health check endpoint.

**Response:**

```json
{
  "status": "healthy",
  "timestamp": 1699123456.789
}
```

---

## Configuration

### Environment Variables

```bash
# Application
APP_ENV=development              # development, staging, production
APP_NAME=Verona AI Search
DEBUG=false

# Qdrant
QDRANT_HOST=localhost                        # Use qdrant.qdrant.svc.cluster.local for k8s
QDRANT_PORT=6333
QDRANT_COLLECTION=matrimonial_profiles

# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini         # For query parsing

# Embedding
EMBEDDING_DEVICE=cpu             # cpu, cuda, mps
COLBERT_BATCH_SIZE=20

# Search
DEFAULT_SEARCH_LIMIT=50
MAX_SEARCH_LIMIT=200
SCORE_THRESHOLD=0.0

# CloudFront (optional override - auto-selected based on APP_ENV)
# CLOUD_FRONT_URL=https://d34thlcszyehjn.cloudfront.net
```

### CloudFront URLs by Environment

CloudFront URL is **automatically selected** based on `APP_ENV`:

| APP_ENV | CloudFront URL |
|---------|----------------|
| `production` | `https://d1awhb6e7ezecy.cloudfront.net` |
| `staging` | `https://d34thlcszyehjn.cloudfront.net` |
| `development` | `https://d34thlcszyehjn.cloudfront.net` (uses staging) |

You can override by setting `CLOUD_FRONT_URL` explicitly in your environment.

---

## Qdrant Point Structure

Each user profile is stored as a Qdrant point:

```json
{
  "id": "uuid5(user_id)",
  "vectors": {
    "education": [1536 floats],
    "profession": [1536 floats],
    "vibe_report": [1536 floats],
    "blurb_colbert": [[1024 floats], ...]
  },
  "payload": {
    "id": "68f0cca72aa8021bc6fb544f",
    "is_circulateable": true,
    "is_paused": false,
    "last_active": "2024-01-15T10:30:00Z",

    "gender": "male",
    "height": 72,
    "dob": "1990-01-01",
    "current_location": "Mumbai",
    "annual_income": 2500000,

    "religion": "HI",
    "caste": "XX",
    "fitness": "ER",
    "religiosity": "MO",
    "smoking": "NS",
    "drinking": "DD",
    "food_habits": "VGT",
    "intent": "01",
    "open_to_children": "OC",

    "education": "B.Tech from IIT Delhi; MBA from IIM Bangalore",
    "education_hash": "a1b2c3d4...",
    "profession": "Director at Google",
    "profession_hash": "e5f6g7h8...",
    "vibe_report": "A compelling synthesis of analytical rigor...",
    "vibe_report_hash": "i9j0k1l2...",

    "blurb": "Love traveling and meeting new people...",
    "profile_hook": "A total game-changer. Spectacular alignment...",
    "interests": ["Travel", "Music", "Photography"],
    "photo_collection": [
      {
        "show_case_id": "photo_1",
        "url": "https://d1awhb6e7ezecy.cloudfront.net/photos/photo_1.jpg",
        "cropped_url": "https://d1awhb6e7ezecy.cloudfront.net/photos/photo_1_cropped.jpg"
      }
    ]
  }
}
```

---

## Running the Application

### Quick Start

```bash
# Setup (first time only)
./setup.sh

# Start server
./start.sh
```

### Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install "transformers>=4.36.0,<4.46.0"

# Copy environment file
cp .env.example .env.development

# Configure settings in .env.development
# - Set OPENAI_API_KEY

# Start server
export TOKENIZERS_PARALLELISM=false
APP_ENV=development uvicorn app.main:app --reload
```

### Verify Setup

```bash
# Check Qdrant collection
curl http://localhost:6333/collections/profiles_development

# Test ingest API
curl -X POST http://localhost:8000/api/ingest \
  -H "Content-Type: application/json" \
  -d '{"_id": "test123", "gender": "male", ...}'

# Check API docs
open http://localhost:8000/docs
```
