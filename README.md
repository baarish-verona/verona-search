# Verona AI Search

Profile search and ingestion service using Qdrant vector database with semantic search capabilities.

## Features

- Profile ingestion with smart updates (only updates changed fields)
- Vibe report generation using GPT-4o vision
- Semantic search using OpenAI embeddings and BGE-M3 ColBERT
- Multi-vector late interaction search

## Local Development

### Prerequisites

- Python 3.11+
- Qdrant running locally (`docker run -p 6333:6333 qdrant/qdrant`)
- OpenAI API key

### Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY=sk-proj-...

# Run server
APP_ENV=development uvicorn app.main:app --reload
```

## Docker build & deploy for production

### Verify Code (before building)

```shell
python -c "from app.main import app" && echo "Code OK"
```

### Build Image

```shell
docker build --platform linux/amd64 --build-arg APP_ENV=production -t 988602099059.dkr.ecr.ap-south-1.amazonaws.com/verona-ai-search-prod:latest .
```

### Push Image

```shell
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin 988602099059.dkr.ecr.ap-south-1.amazonaws.com
docker push 988602099059.dkr.ecr.ap-south-1.amazonaws.com/verona-ai-search-prod:latest
```

### Deploy Image

```shell
kubectl apply -f verona-ai-search-prod.yaml
```

## Redeploy if Image changed
```shell
kubectl rollout restart deployment/verona-ai-search -n prod
```

## Docker build & deploy for staging

### Verify Code (before building)

```shell
python -c "from app.main import app" && echo "Code OK"
```

### Build Image

```shell
docker build --platform linux/amd64 --build-arg APP_ENV=staging -t 988602099059.dkr.ecr.ap-south-1.amazonaws.com/verona-ai-search-staging:latest .
```

### Push Image

```shell
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin 988602099059.dkr.ecr.ap-south-1.amazonaws.com
docker push 988602099059.dkr.ecr.ap-south-1.amazonaws.com/verona-ai-search-staging:latest
```

### Deploy Image

```shell
kubectl apply -f verona-ai-search-staging.yaml
```

## Redeploy if Image changed

```shell
kubectl rollout restart deployment/verona-ai-search -n staging
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `APP_ENV` | Environment (development/staging/production) | `production` |
| `OPENAI_API_KEY` | OpenAI API key (required) | - |
| `QDRANT_HOST` | Qdrant server host | `localhost` |
| `QDRANT_PORT` | Qdrant server port | `6333` |
| `QDRANT_COLLECTION` | Qdrant collection name | `matrimonial_profiles` |
| `CLOUD_FRONT_URL` | CloudFront base URL for photos | - |

## API Endpoints

### Health Check
```
GET /api/health
```

### Ingest Profile
```
POST /api/ingest
Content-Type: application/json

{
  "_id": "user123",
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
  "blurb": "Profile description"
}
```

Returns the processed User object.

### Search
```
POST /api/search
Content-Type: application/json

{
  "query": "engineer who loves travel",
  "filters": {"genders": ["male"]},
  "limit": 10
}
```

### Collection Info
```
GET /api/collection/info
```

## Ingestion Logic

1. Profile is converted to internal User model
2. `is_circulateable` is computed from status flags
3. If not circulateable and profile doesn't exist → skip
4. If not circulateable and profile exists → update `is_circulateable` only
5. If circulateable → full upsert or smart update

### Smart Updates

- Compares MD5 hashes of education, profession, and vibe input
- Only regenerates embeddings for changed fields
- Vibe report regenerated only when input content changes
