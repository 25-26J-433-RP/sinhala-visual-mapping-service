"""
Sinhala Mind Map Generator Backend API
This Flask application generates graph-ready mind map data from Sinhala text.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from mindmap_generator import SinhalaMindMapGenerator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize mind map generator
mindmap_generator = SinhalaMindMapGenerator()


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
        
        if 'text' in data:
            sinhala_text = data['text']
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
        
        # Generate mind map
        logger.info("Generating mind map from Sinhala text")
        mindmap_data = mindmap_generator.generate(sinhala_text)
        
        return jsonify({
            'success': True,
            'data': mindmap_data
        }), 200
        
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
        for text in texts:
            mindmap_data = mindmap_generator.generate(text)
            results.append(mindmap_data)
        
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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
