# Verona Profile Search -Architecture Documentation

A multi-vector profile search service

---

1. [Project Overview](#project-overview)

---

## Project Overview

![img.png](img.png)

### Capabilities

- **Profile-Search**: It will search users in the database for the given queries, and return the profiles with the number of profiles for the hard-filters specified
- **MultiVector Embedding Model**: Currently, we have User Model 
```python
class User(BaseModel):
    id: str
    height: Optional[str] = None
    income: Optional[str] = None
    religion: Optional[str] = None
    age: Optional[str] = None
    location: Optional[List[str]] = None
    
    
```