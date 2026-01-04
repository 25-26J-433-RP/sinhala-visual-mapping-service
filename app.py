"""
Sinhala Mind Map Generator Backend API
This Flask application generates graph-ready mind map data from Sinhala text.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from intelligent_mindmap_generator import IntelligentMindMapGenerator
import logging
from config import Config

# Optional Neo4j
try:
    from neo4j import GraphDatabase
except Exception:
    GraphDatabase = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize intelligent mind map generator (AI-powered only)
intelligent_generator = IntelligentMindMapGenerator()
logger.info("Initialized intelligent mind map generator with AI capabilities")


def build_generation_options(payload: dict) -> dict:
    """Construct generation options using request overrides and Config defaults."""
    payload = payload or {}
    return {
        'max_nodes': payload.get('max_nodes', Config.MAX_NODES),
        'semantic_clustering': payload.get('semantic_clustering', Config.SEMANTIC_CLUSTERING),
        'relationship_threshold': payload.get('relationship_threshold', Config.MIN_RELATIONSHIP_CONFIDENCE)
    }

# Initialize Neo4j driver if configured
neo4j_driver = None
if GraphDatabase and Config.NEO4J_URI and Config.NEO4J_USER and Config.NEO4J_PASSWORD:
    try:
        neo4j_driver = GraphDatabase.driver(Config.NEO4J_URI, auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD))
        logger.info("Connected to Neo4j (driver initialized)")
    except Exception as e:
        logger.error(f"Failed to create Neo4j driver: {e}")


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'Sinhala Mind Map Generator API'
    }), 200


@app.route('/api/mindmap/generate', methods=['POST'])
def generate_mindmap():
    """
    Generate mind map from Sinhala text.
    
    Request Body:
    {
        "text": "Sinhala paragraph text"
    }
    OR
    {
        "external_api_url": "https://api.example.com/cleaned_text",
        "api_key": "optional_api_key"
    }
    
    Response:
    {
        "success": true,
        "data": {
            "nodes": [...],
            "edges": [...]
        }
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        # Get text either directly or from external API
        sinhala_text = None
        essay_id = data.get('essay_id') if isinstance(data, dict) else None
        
        if 'text' in data:
            sinhala_text = data['text']
            if not isinstance(sinhala_text, str):
                return jsonify({
                    'success': False,
                    'error': '"text" must be a string'
                }), 400
            logger.info("Received direct text input")
        elif 'external_api_url' in data:
            # Fetch from external API
            external_url = data['external_api_url']
            headers = {}
            
            if 'api_key' in data:
                headers['Authorization'] = f"Bearer {data['api_key']}"
            
            logger.info(f"Fetching data from external API: {external_url}")
            response = requests.get(external_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                external_data = response.json()
                # Try to get cleaned_text field
                sinhala_text = external_data.get('cleaned_text')
                # If external response includes an essay_id, prefer that
                if not essay_id:
                    essay_id = external_data.get('essay_id')

                if not sinhala_text:
                    # Fallback to other possible fields
                    sinhala_text = external_data.get('text') or external_data.get('content')
            else:
                return jsonify({
                    'success': False,
                    'error': f'Failed to fetch from external API: {response.status_code}'
                }), 500
        
        if not sinhala_text:
            return jsonify({
                'success': False,
                'error': 'No text provided. Please provide either "text" or "external_api_url"'
            }), 400
        
        # Generate mind map using intelligent mode (AI-powered)
        generation_options = build_generation_options(data)
        
        logger.info("Generating intelligent mind map with AI-powered node/relationship creation")
        mindmap_data = intelligent_generator.generate(sinhala_text, generation_options)
        # Save to Neo4j if available (best-effort)
        if neo4j_driver:
            try:
                def _merge_node(tx, props):
                    tx.run(
                        """
                        MERGE (n:Concept {id: $id})
                        SET n += $props
                        """,
                        id=props['id'],
                        props=props
                    )

                def _merge_edge(tx, params):
                    tx.run(
                        """
                        MATCH (s:Concept {id: $source}), (t:Concept {id: $target})
                        MERGE (s)-[r:REL {id: $id}]->(t)
                        SET r.rel_type = $rel_type
                        """,
                        id=params['id'],
                        source=params['source'],
                        target=params['target'],
                        rel_type=params.get('rel_type')
                    )

                with neo4j_driver.session(database=Config.NEO4J_DATABASE) as session:
                    nodes = mindmap_data.get('nodes', [])
                    edges = mindmap_data.get('edges', [])

                    for node in nodes:
                        props = {
                            'id': node.get('id'),
                            'label': node.get('label'),
                            'level': node.get('level'),
                            'type': node.get('type'),
                            'size': node.get('size')
                        }
                        if essay_id:
                            props['essay_id'] = essay_id
                        session.execute_write(_merge_node, props)

                    for edge in edges:
                        params = {
                            'id': edge.get('id'),
                            'source': edge.get('source'),
                            'target': edge.get('target'),
                            'rel_type': edge.get('type')
                        }
                        session.execute_write(_merge_edge, params)

            except Exception as e:
                logger.error(f"Error saving mindmap to Neo4j: {e}")
        # Attach essay_id to response if provided
        response_payload = {
            'success': True,
            'data': mindmap_data
        }

        if essay_id:
            response_payload['essay_id'] = essay_id

        return jsonify(response_payload), 200
        
    except requests.RequestException as e:
        logger.error(f"Error fetching from external API: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'External API error: {str(e)}'
        }), 500
    except Exception as e:
        logger.error(f"Error generating mind map: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/mindmap/batch', methods=['POST'])
def batch_generate_mindmap():
    """
    Generate multiple mind maps from a list of Sinhala texts.
    
    Request Body:
    {
        "texts": ["text1", "text2", ...]
    }
    
    Response:
    {
        "success": true,
        "data": [
            {"nodes": [...], "edges": [...]},
            {"nodes": [...], "edges": [...]}
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({
                'success': False,
                'error': 'No texts array provided'
            }), 400
        
        texts = data['texts']
        if not isinstance(texts, list):
            return jsonify({
                'success': False,
                'error': 'texts must be an array'
            }), 400
        
        results = []
        for item in texts:
            # Support either string items or objects with text and optional essay_id
            item_text = None
            item_essay_id = None
            item_options = {}

            if isinstance(item, dict):
                item_text = item.get('text')
                item_essay_id = item.get('essay_id')
                item_options = item
            else:
                item_text = item
                item_options = {}

            if not isinstance(item_text, str) or not item_text.strip():
                empty_graph = intelligent_generator._empty_graph()
                result_entry = {
                    **empty_graph,
                    'essay_id': item_essay_id,
                    'error': 'Invalid or empty text provided'
                }
                results.append(result_entry)
                continue

            generation_options = build_generation_options(item_options)
            mindmap_data = intelligent_generator.generate(item_text, generation_options)

            # Attach essay_id to each result when available
            result_entry = mindmap_data.copy()
            if item_essay_id:
                result_entry['essay_id'] = item_essay_id

            results.append(result_entry)
        
        return jsonify({
            'success': True,
            'data': results
        }), 200
        
    except Exception as e:
        logger.error(f"Error in batch generation: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/mindmap/essay/<essay_id>', methods=['GET'])
def get_mindmap_by_essay(essay_id):
    """Retrieve a previously saved mindmap from Neo4j using the essay_id.

    Returns the same graph-ready structure (nodes, edges, metadata) when available.
    """
    try:
        if not neo4j_driver:
            return jsonify({
                'success': False,
                'error': 'Neo4j is not configured on the server'
            }), 400

        nodes = []
        edges = []
        nodes_map = {}

        with neo4j_driver.session(database=Config.NEO4J_DATABASE) as session:
            # Fetch all nodes with the essay_id
            node_records = session.run(
                "MATCH (n:Concept {essay_id:$essay_id}) RETURN n",
                essay_id=essay_id
            )

            for rec in node_records:
                n = rec.get('n')
                if not n:
                    continue
                nid = n.get('id') if 'id' in n else None
                if nid in nodes_map:
                    continue
                node_obj = {
                    'id': nid,
                    'label': n.get('label'),
                    'level': n.get('level'),
                    'type': n.get('type'),
                    'size': n.get('size')
                }
                nodes_map[nid] = node_obj

            # Fetch edges among those nodes
            edge_records = session.run(
                "MATCH (s:Concept {essay_id:$essay_id})-[r]->(t:Concept {essay_id:$essay_id}) RETURN r,s,t",
                essay_id=essay_id
            )

            for rec in edge_records:
                r = rec.get('r')
                s = rec.get('s')
                t = rec.get('t')
                if not (r and s and t):
                    continue

                sid = s.get('id')
                tid = t.get('id')

                # ensure source/target nodes are included
                if sid not in nodes_map:
                    nodes_map[sid] = {
                        'id': sid,
                        'label': s.get('label'),
                        'level': s.get('level'),
                        'type': s.get('type'),
                        'size': s.get('size')
                    }
                if tid not in nodes_map:
                    nodes_map[tid] = {
                        'id': tid,
                        'label': t.get('label'),
                        'level': t.get('level'),
                        'type': t.get('type'),
                        'size': t.get('size')
                    }

                edge_obj = {
                    'id': r.get('id'),
                    'source': sid,
                    'target': tid,
                    'type': r.get('rel_type')
                }
                edges.append(edge_obj)

        # Build nodes list from nodes_map
        nodes = list(nodes_map.values())

        metadata = {
            'total_nodes': len(nodes),
            'total_edges': len(edges),
            'text_length': None
        }

        return jsonify({
            'success': True,
            'data': {
                'nodes': nodes,
                'edges': edges,
                'metadata': metadata
            }
        }), 200

    except Exception as e:
        logger.error(f"Error retrieving mindmap from Neo4j: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
