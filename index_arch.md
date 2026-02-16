# Indexing Strategy Review for Matrimonial Profile Search

## How Indexing Currently Works

### 1. Collection Creation (`qdrant_client.py:45-94`)

When `create_collection()` is called:

```
1. Delete existing collection (if recreate=True)
2. Build vectors_config from VECTOR_CONFIG in embedding_specs.py
   - For each vector field: create VectorParams with size, distance metric, multivector config
3. Create collection with vectors_config
4. Call _create_payload_indexes() to create filterable field indexes
```

### 2. Vector Indexing (`embedding_specs.py`)

Three named vectors are configured:

| Vector Name | Provider | How It's Generated |
|-------------|----------|-------------------|
| `education` | OpenAI text-embedding-3-small | Concatenated "{degree} from {college}" strings |
| `profession` | OpenAI text-embedding-3-small | Concatenated "{designation} at {company}" strings |
| `vibe_report` | BGE-M3 ColBERT | AI-generated character summary from GPT-4o |

**Vector generation flow:**
```
Profile JSON → ingest_service.py → embedding provider → Qdrant upsert
```

### 3. Payload Indexing (`qdrant_client.py:96-117`)

**Integer indexes** (for range queries like min_age/max_age):
- `age`, `height`, `income`

**Keyword indexes** (for exact match/match_any):
- `id` ← **Already indexed!**
- `gender`, `religion`, `location`, `marital_status`
- `family_type`, `food_habits`, `smoking`, `drinking`
- `religiosity`, `fitness`, `intent`

### 4. Document Upsert Flow (`ingest_service.py`)

```
1. Receive IngestUserProfile (from API/batch script)
2. Convert to User model (compute derived fields)
3. Check if profile exists in Qdrant
4. If new or forceUpdate:
   - Generate ALL embeddings (education, profession, vibe_report)
   - Full upsert to Qdrant
5. If existing (smart update):
   - Compare MD5 hashes of education, profession, vibe_report
   - Only regenerate embeddings for changed fields
   - Partial vector update via update_vectors()
```

---

## Current State Summary

**Vector Database:** Qdrant with collection `matrimonial_profiles`

**Current Vectors:**
| Field | Provider | Dimensions | Type |
|-------|----------|------------|------|
| education | OpenAI text-embedding-3-small | 1536 | Dense |
| profession | OpenAI text-embedding-3-small | 1536 | Dense |
| vibe_report | BGE-M3 ColBERT | 1024 | Multivector |

**Current Payload Indexes:**
- Integer: `age`, `height`, `income`
- Keyword: `id` (already indexed!), `gender`, `religion`, `location`, `marital_status`, `family_type`, `food_habits`, `smoking`, `drinking`, `religiosity`, `fitness`, `intent`

---

## Latency Analysis: Dense vs ColBERT

| Approach | Operation | Latency Impact |
|----------|-----------|----------------|
| **Dense** | Single cosine similarity O(d) | **Faster** |
| **ColBERT** | MaxSim across all token pairs O(q×d) | **Slower** |

For vibe_report with ~250 tokens, ColBERT computes ~250x more operations than Dense per document.

---

## Observations & Recommendations (Review Only)

### Vector Indexing

| Current | Assessment |
|---------|------------|
| education (Dense) | Good - short structured text works well with dense |
| profession (Dense) | Good - short structured text works well with dense |
| vibe_report (ColBERT) | Consider Dense if latency is priority - 250x faster per doc |

### Payload Indexes

**Already indexed:**
- `id` ← Yes, already in keyword_fields list

**Potentially missing (not critical):**
- `is_circulateable` - useful for filtering non-searchable profiles
- `caste`, `open_to_children` - mentioned in data model but not indexed
- `last_active` - useful for recency sorting

### What's Working Well

- Smart update with MD5 hashing avoids unnecessary re-embedding
- Hybrid search with prefetch + DBSF fusion
- Integer indexes for range queries (age, height, income)
- Keyword indexes for all categorical filter fields

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `app/config/embedding_specs.py` | Vector configuration (providers, dimensions, types) |
| `app/vector_store/qdrant_client.py` | Collection creation, payload indexes, upsert/search |
| `app/vector_store/filters.py` | Filter building for range and match_any queries |
| `app/services/ingest_service.py` | Smart update logic, embedding generation |
| `app/vector_store/query_builder.py` | Dynamic query construction with prefetch |
