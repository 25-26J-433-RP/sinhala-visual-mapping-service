"""
Example usage of the Intelligent Mind Map Generator with AI-powered features.
Demonstrates how to use the enhanced API endpoints with semantic analysis.
"""

import requests
import json


def test_intelligent_generation():
    """Test the intelligent mind map generation."""
    
    # Sample Sinhala text for testing
    sinhala_text = """
    ශ්‍රී ලංකාව දකුණු ආසියාවේ පිහිටි දූපත් රටකි. එහි ජනගහනය මිලියන 22 කි. 
    කොළඹ ශ්‍රී ලංකාවේ වාණිජ අගනුවර වේ. ශ්‍රී ජයවර්ධනපුර කෝට්ටේ පරිපාලන අගනුවර වේ.
    
    ශ්‍රී ලංකාවේ ප්‍රධාන ආර්ථික කටයුතු වන්නේ තේ, කෝපි සහ කුළුබඩු වගාවයි. 
    මෙරට සංචාරක ව්‍යාපාරය ද වැදගත් වේ. බොහෝ ඓතිහාසික ස්ථාන සහ වෙරළ තීරයන් ඇත.
    
    සිංහල, දෙමළ සහ ඉංග්‍රීසි මෙරට භාෂා වේ. බුද්ධාගම ප්‍රධාන ආගමයි. 
    ශ්‍රී ලංකාවේ අධ්‍යාපන මට්ටම ඉහළයි. නොමිලේ අධ්‍යාපනය ලැබේ.
    """
    
    # API endpoint
    url = "http://localhost:5000/api/mindmap/generate"
    
    # Test 1: Basic intelligent generation
    print("Test 1: Basic Intelligent Generation")
    print("-" * 50)
    
    payload = {
        "text": sinhala_text,
        "intelligent": True,
        "max_nodes": 30,
        "semantic_clustering": True,
        "relationship_threshold": 0.4
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Success! Generated {result['data']['metadata']['total_nodes']} nodes")
        print(f"✓ Found {result['data']['metadata']['entities_found']} entities")
        print(f"✓ Detected {result['data']['metadata']['relationships_found']} relationships")
        print(f"✓ Intelligence level: {result['data']['metadata']['intelligence_level']}")
        print(f"\nSample Nodes:")
        for node in result['data']['nodes'][:5]:
            print(f"  - {node['label']} (Level {node['level']}, Type: {node['type']}, Importance: {node.get('importance', 'N/A')})")
        
        print(f"\nSample Edges:")
        for edge in result['data']['edges'][:5]:
            print(f"  - Type: {edge['type']}, Weight: {edge.get('weight', 1)}")
    else:
        print(f"✗ Error: {response.status_code}")
        print(response.text)
    
    print("\n" + "=" * 50 + "\n")
    
    # Test 2: Compare with basic generation
    print("Test 2: Basic Generation (for comparison)")
    print("-" * 50)
    
    payload_basic = {
        "text": sinhala_text,
        "intelligent": False
    }
    
    response = requests.post(url, json=payload_basic)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Basic generation: {result['data']['metadata']['total_nodes']} nodes")
        print(f"✓ Intelligence level: {result['data']['metadata'].get('intelligence_level', 'basic')}")
    else:
        print(f"✗ Error: {response.status_code}")
    
    print("\n" + "=" * 50 + "\n")
    
    # Test 3: High semantic clustering
    print("Test 3: High Semantic Clustering")
    print("-" * 50)
    
    payload_clustering = {
        "text": sinhala_text,
        "intelligent": True,
        "max_nodes": 50,
        "semantic_clustering": True,
        "relationship_threshold": 0.6  # Higher threshold for stronger relationships
    }
    
    response = requests.post(url, json=payload_clustering)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Generated {result['data']['metadata']['total_nodes']} nodes")
        print(f"✓ Formed {result['data']['metadata']['clusters']} concept clusters")
        print(f"✓ High-confidence relationships: {result['data']['metadata']['relationships_found']}")
        
        # Show relationship types
        edge_types = {}
        for edge in result['data']['edges']:
            edge_type = edge.get('type', 'unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        
        print(f"\nRelationship types distribution:")
        for rel_type, count in edge_types.items():
            print(f"  - {rel_type}: {count}")
    else:
        print(f"✗ Error: {response.status_code}")
    
    print("\n" + "=" * 50 + "\n")
    
    # Test 4: Save full graph data to file
    print("Test 4: Save Graph Data")
    print("-" * 50)
    
    payload = {
        "text": sinhala_text,
        "intelligent": True,
        "max_nodes": 40,
        "semantic_clustering": True
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        
        # Save to file
        with open('intelligent_mindmap_output.json', 'w', encoding='utf-8') as f:
            json.dump(result['data'], f, ensure_ascii=False, indent=2)
        
        print("✓ Graph data saved to 'intelligent_mindmap_output.json'")
        print(f"✓ Total nodes: {result['data']['metadata']['total_nodes']}")
        print(f"✓ Total edges: {result['data']['metadata']['total_edges']}")
        
        # Calculate statistics
        node_levels = {}
        for node in result['data']['nodes']:
            level = node['level']
            node_levels[level] = node_levels.get(level, 0) + 1
        
        print(f"\nNodes by level:")
        for level in sorted(node_levels.keys()):
            print(f"  - Level {level}: {node_levels[level]} nodes")
    else:
        print(f"✗ Error: {response.status_code}")
    
    print("\n" + "=" * 50 + "\n")


def test_batch_processing():
    """Test batch processing of multiple texts."""
    print("Test 5: Batch Processing")
    print("-" * 50)
    
    texts = [
        "ශ්‍රී ලංකාව අපගේ මාතෘභූමියයි. එය ඉතා සුන්දර දූපතකි.",
        "ගණිතය විද්‍යාවේ ඉතා වැදගත් අංශයකි. එය තාර්කික චින්තනයට උපකාර වේ.",
        "පරිසරය ආරක්ෂා කිරීම අප සැමගේ යුතුකමකි. ගස් වැඩීම වැදගත් වේ."
    ]
    
    url = "http://localhost:5000/api/mindmap/generate"
    
    for i, text in enumerate(texts, 1):
        print(f"\nProcessing text {i}...")
        response = requests.post(url, json={
            "text": text,
            "intelligent": True,
            "max_nodes": 20
        })
        
        if response.status_code == 200:
            result = response.json()
            nodes = result['data']['metadata']['total_nodes']
            entities = result['data']['metadata']['entities_found']
            print(f"  ✓ Generated {nodes} nodes, found {entities} entities")
        else:
            print(f"  ✗ Error: {response.status_code}")
    
    print("\n" + "=" * 50 + "\n")


def print_usage_info():
    """Print usage information."""
    print("\n" + "=" * 70)
    print("Intelligent Mind Map Generator - API Usage")
    print("=" * 70)
    print("\nEndpoint: POST /api/mindmap/generate")
    print("\nRequest Parameters:")
    print("  - text (string): Sinhala text to process [REQUIRED]")
    print("  - intelligent (boolean): Use AI-powered generation [default: true]")
    print("  - max_nodes (int): Maximum number of nodes [default: 50]")
    print("  - semantic_clustering (boolean): Group similar concepts [default: true]")
    print("  - relationship_threshold (float): Min confidence for relationships [default: 0.4]")
    print("\nFeatures:")
    print("  ✓ AI-powered entity extraction")
    print("  ✓ Intelligent relationship detection")
    print("  ✓ Semantic similarity analysis")
    print("  ✓ Automatic concept clustering")
    print("  ✓ Importance scoring for nodes")
    print("  ✓ Multi-level graph hierarchy")
    print("\nModels Used:")
    print("  - Sentence Transformers (paraphrase-multilingual-MiniLM-L12-v2)")
    print("  - Custom Sinhala NLP Engine")
    print("  - Semantic embeddings for similarity")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    import sys
    
    print_usage_info()
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:5000/health", timeout=2)
        if response.status_code == 200:
            print("✓ Server is running\n")
        else:
            print("✗ Server responded with error\n")
            sys.exit(1)
    except requests.ConnectionError:
        print("✗ Error: Server is not running!")
        print("Please start the server with: python app.py")
        print("or: ./run.sh\n")
        sys.exit(1)
    
    # Run tests
    try:
        test_intelligent_generation()
        test_batch_processing()
        
        print("\n" + "=" * 70)
        print("All tests completed!")
        print("Check 'intelligent_mindmap_output.json' for detailed output")
        print("=" * 70 + "\n")
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
