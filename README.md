# Sinhala Mind Map Generator API

A Python Flask backend API that generates graph-ready mind map data from Sinhala text with **AI-powered intelligent node and relationship creation**. This API can process Sinhala paragraphs directly or fetch cleaned text from external APIs.

## ✨ New: AI-Powered Intelligent Generation

**Now featuring lightweight AI models for enhanced mind map generation!**

- 🤖 **AI-Powered Entity Extraction**: Automatically identifies important concepts using NLP
- 🔗 **Intelligent Relationship Detection**: Discovers semantic connections between concepts
- 📊 **Semantic Similarity Analysis**: Groups related concepts using multilingual embeddings
- 🎯 **Importance Scoring**: Assigns relevance scores to nodes and relationships
- 🌐 **Concept Clustering**: Organizes similar ideas automatically

## Features

- ✨ Generate hierarchical mind map structures from Sinhala text
- 🤖 **NEW**: Intelligent mode with AI-powered entity and relationship extraction
- 🧠 **NEW**: Semantic analysis using lightweight multilingual models
- 🔄 Support for both direct text input and external API integration
- 📊 Graph-ready output with nodes and edges
- 🚀 Batch processing support
- 🌐 CORS enabled for frontend integration
- 🔍 RESTful API design
- 💾 Neo4j graph database integration

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```


4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env file with your settings if needed
```

## Usage

### Starting the API Server

```bash
python app.py
```

The API will start on `http://localhost:5000`

### API Endpoints

#### 1. Health Check
```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "Sinhala Mind Map Generator API"
}
```

#### 2. Generate Mind Map (Direct Text)
```
POST /api/mindmap/generate
```

**Request Body (Intelligent Mode - Default):**
```json
{
  "text": "ශ්‍රී ලංකාව දකුණු ආසියාවේ පිහිටි දිවයිනකි. එය සුන්දර වෙරළ තීරයන්, පුරාණ නටබුන් සහ පොහොසත් සංස්කෘතියකින් යුක්තය.",
  "intelligent": true,
  "max_nodes": 50,
  "semantic_clustering": true,
  "relationship_threshold": 0.4
}
```

**Request Parameters:**
- `text` (string, required): Sinhala text to process
- `essay_id` (string, optional): Identifier for tracking
- `intelligent` (boolean, default: true): Use AI-powered generation
- `max_nodes` (integer, default: 50): Maximum number of nodes
- `semantic_clustering` (boolean, default: true): Enable concept clustering
- `relationship_threshold` (float, default: 0.4): Minimum confidence for relationships

**Response (Intelligent Mode):**
```json
{
  "success": true,
  "data": {
    "nodes": [
      {
        "id": "abc123",
        "label": "ශ්‍රී ලංකාව",
        "level": 0,
        "type": "root",
        "size": 35,
        "importance": 1.0,
        "color": "#FF6B6B"
      },
      {
        "id": "def456",
        "label": "දකුණු ආසියාව",
        "level": 1,
        "type": "topic",
        "size": 25,
        "importance": 0.85,
        "color": "#4ECDC4"
      }
    ],
    "edges": [
      {
        "id": "edge123",
        "source": "abc123",
        "target": "def456",
        "type": "hierarchy",
        "weight": 3
      },
      {
        "id": "edge456",
        "source": "def456",
        "target": "ghi789",
        "type": "semantic",
        "weight": 2,
        "confidence": 0.75,
        "style": "dashed"
      }
    ],
    "metadata": {
      "total_nodes": 25,
      "total_edges": 32,
      "entities_found": 18,
      "relationships_found": 12,
      "clusters": 5,
      "text_length": 150,
      "intelligence_level": "advanced"
    }
  }
}
```

**Response (Basic Mode):**
Set `"intelligent": false` to use the original rule-based generation.

```json
{
  "success": true,
  "data": {
    "nodes": [...],
    "edges": [...],
    "metadata": {
      "total_nodes": 10,
      "total_edges": 9,
      "text_length": 150
    }
  }
}
```

#### 3. Generate Mind Map (External API)
```
POST /api/mindmap/generate
```

**Request Body:**
```json
{
  "external_api_url": "https://api.example.com/v1/text/cleaned",
  "api_key": "your_api_key_here"
}
```

The API will fetch data from the external URL and look for a `cleaned_text` field in the response.

#### 4. Batch Generate Mind Maps
```
POST /api/mindmap/batch
```

**Request Body:**
```json
{
  "texts": [
    {"essay_id": "e1", "text": "පරිගණකය යනු ඉලෙක්ට්‍රොනික උපකරණයකි."},
    {"essay_id": "e2", "text": "ශ්‍රී ලංකාව සංචාරක ගමනාන්ත සඳහා ප්‍රසිද්ධය."}
  ]
}
```

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "essay_id": "e1",
      "nodes": [...],
      "edges": [...],
      "metadata": {...}
    },
    {
      "essay_id": "e2",
      "nodes": [...],
      "edges": [...],
      "metadata": {...}
    }
  ]
}
```

#### 5. Retrieve Saved Mind Map by Essay ID
```
GET /api/mindmap/essay/<essay_id>
```

Fetches a previously saved mind map (nodes and edges) from the configured Neo4j database using the `essay_id` property.

**Response:**
```json
{
  "success": true,
  "data": {
    "nodes": [...],
    "edges": [...],
    "metadata": {
      "total_nodes": 10,
      "total_edges": 9,
      "text_length": null
    }
  }
}
```

Note: This endpoint requires Neo4j to be configured (`NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`). If Neo4j is not configured, the endpoint returns an error.

## Testing

### Using curl

```bash
# Health check
curl http://localhost:5000/health

# Generate mind map
curl -X POST http://localhost:5000/api/mindmap/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "ශ්‍රී ලංකාව දකුණු ආසියාවේ පිහිටි දිවයිනකි..."}'
```

### Using the test script

```bash
chmod +x test_api.sh
./test_api.sh
```



## Output Format

The API returns a graph-ready structure with:

### Nodes
Each node represents a concept with:
- `id`: Unique identifier
- `label`: Text content
- `level`: Hierarchy level (0=root, 1=topic, 2=subtopic, 3=detail)
- `type`: Node type (root, topic, subtopic, detail)
- `size`: Visual size for rendering

### Edges
Each edge represents a relationship with:
- `id`: Unique identifier
- `source`: Source node ID
- `target`: Target node ID
- `type`: Relationship type (hierarchy, detail)

## Configuration

Edit `.env` file to customize:

```env
DEBUG=True
HOST=0.0.0.0
PORT=5000
EXTERNAL_API_TIMEOUT=10
MAX_NODES=100
MAX_LEVELS=4
CORS_ORIGINS=*
```

## Project Structure

```
visual-mapping-model-training/
├── app.py                      # Main Flask application
├── mindmap_generator.py        # Mind map generation logic
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variables template
├── .gitignore                 # Git ignore rules
├── test_api.sh                # API test script
└── README.md                  # This file
```

## Dependencies

- Flask 3.0.0 - Web framework
- Flask-CORS 4.0.0 - CORS support
- requests 2.31.0 - HTTP library
- python-dotenv 1.0.0 - Environment variables
- spacy 3.7.2 - NLP library (for future enhancements)

## Future Enhancements

- [ ] Integration with Sinhala NLP models
- [ ] Named entity recognition
- [ ] Keyword extraction using TF-IDF
- [ ] Custom graph layouts
- [ ] Export to various formats (JSON, GraphML, etc.)
- [ ] Authentication and rate limiting
- [ ] Caching layer for improved performance

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.