#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sinhala Mind Map API Tester
Tests the API with Sinhala text and displays results in a readable format.
"""

import requests
import json
import sys

# Ensure UTF-8 encoding for terminal output
if sys.stdout.encoding != 'UTF-8':
    sys.stdout.reconfigure(encoding='utf-8')

def print_separator(char="=", length=80):
    """Print a separator line."""
    print(char * length)

def print_section(title):
    """Print a section header."""
    print()
    print_separator()
    print(f"  {title}")
    print_separator()
    print()

def run_mindmap_api(text, api_url='http://localhost:5000'):
    """
    Test the mind map API with given Sinhala text.
    
    Args:
        text: Sinhala text to process
        api_url: API base URL
    """
    print_section("🧪 Sinhala Mind Map API Test")
    
    # Display input
    print("📝 Input Text:")
    print("-" * 80)
    print(text)
    print()
    
    # Call API
    print("🔄 Calling API...")
    try:
        response = requests.post(
            f'{api_url}/api/mindmap/generate',
            json={'text': text},
            timeout=10
        )
        
        if response.status_code != 200:
            print(f"❌ HTTP Error: {response.status_code}")
            return
        
        data = response.json()
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Cannot connect to API. Is the server running?")
        print(f"   Try: python app.py")
        return
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    if not data.get('success'):
        print(f"❌ API Error: {data.get('error', 'Unknown error')}")
        return
    
    print("✅ Success!\n")
    
    # Display metadata
    metadata = data['data']['metadata']
    print("📊 Statistics:")
    print("-" * 80)
    print(f"  • Total Nodes: {metadata['total_nodes']}")
    print(f"  • Total Edges: {metadata['total_edges']}")
    print(f"  • Text Length: {metadata['text_length']} characters")
    print()
    
    # Display nodes hierarchically
    nodes = data['data']['nodes']
    print("🔵 Mind Map Structure (Hierarchical):")
    print("-" * 80)
    
    # Group by level
    levels = {}
    for node in nodes:
        level = node['level']
        if level not in levels:
            levels[level] = []
        levels[level].append(node)
    
    # Icons and colors for different node types
    type_icons = {
        'root': '🌳',
        'topic': '📌',
        'subtopic': '📎',
        'detail': '💠'
    }
    
    type_labels = {
        'root': 'ROOT',
        'topic': 'TOPIC',
        'subtopic': 'SUBTOPIC',
        'detail': 'DETAIL'
    }
    
    # Display by hierarchy
    for level in sorted(levels.keys()):
        for node in levels[level]:
            icon = type_icons.get(node['type'], '•')
            label = type_labels.get(node['type'], node['type'].upper())
            indent = '  ' * level
            
            print(f"{indent}{icon} [{label}] {node['label']}")
            print(f"{indent}   ID: {node['id']} | Level: {node['level']} | Size: {node['size']}")
    
    print()
    
    # Display edges
    edges = data['data']['edges']
    print("🔗 Relationships (Edges):")
    print("-" * 80)
    
    # Create node lookup
    node_lookup = {n['id']: n['label'] for n in nodes}
    
    for i, edge in enumerate(edges, 1):
        source_label = node_lookup.get(edge['source'], 'Unknown')
        target_label = node_lookup.get(edge['target'], 'Unknown')
        
        # Truncate long labels
        if len(source_label) > 60:
            source_label = source_label[:57] + '...'
        if len(target_label) > 60:
            target_label = target_label[:57] + '...'
        
        edge_type_icon = '━━▶' if edge['type'] == 'hierarchy' else '┄┄▶'
        
        print(f"  {i}. {edge_type_icon} [{edge['type'].upper()}]")
        print(f"     From: {source_label}")
        print(f"     To:   {target_label}")
    
    print()
    
    # Graph visualization hint
    print("💡 Graph Visualization:")
    print("-" * 80)
    print("  This structure is ready for visualization with:")
    print("  • D3.js (Force-directed graph)")
    print("  • Cytoscape.js (Network graph)")
    print("  • Vis.js (Network diagram)")
    print("  • React Flow (React-based graphs)")
    print()
    print(f"  See examples/visualization.html for a web demo")
    print()
    
    # Optional: Full JSON output
    print("📄 Full JSON Response:")
    print("-" * 80)
    print(json.dumps(data, ensure_ascii=False, indent=2))
    
    print_section("✅ Test Complete!")


def main():
    """Main function."""
    # Default sample text
    sample_text = """ශ්‍රී ලංකාව දකුණු ආසියාවේ පිහිටි දිවයිනකි. 
එය සුන්දර වෙරළ තීරයන්, පුරාණ නටබුන් සහ පොහොසත් සංස්කෘතියකින් යුක්තය. 
ශ්‍රී ලංකාවේ ජනගහනය මිලියන 22 කි. 
රට බෞද්ධ ආගමික උරුමයන්ගෙන් පොහොසත්ය. 
කොළඹ වාණිජ අගනුවර වන අතර ශ්‍රී ජයවර්ධනපුර කෝට්ටේ පරිපාලන අගනුවරයි."""
    
    # Check if custom text provided via command line
    if len(sys.argv) > 1:
        sample_text = ' '.join(sys.argv[1:])
    
    # Run test
    run_mindmap_api(sample_text)


if __name__ == '__main__':
    main()
