#!/bin/bash

# Configure UTF-8 for Sinhala text display
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║           Sinhala Mind Map API - Complete Feature Test                    ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

API_URL="http://localhost:5000"

# Test 1: Health Check
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 1: 🏥 Health Check"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
curl -s $API_URL/health | python3 -m json.tool
echo ""
echo ""

# Test 2: Simple Sinhala Text
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 2: 📝 Generate Mind Map - Simple Text"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 test_sinhala.py "සූර්යයා අපගේ සෞරග්‍රහ මණ්ඩලයේ කේන්ද්‍රයයි. පෘථිවිය සූර්යයා වටා කරකැවේ."
echo ""
echo ""

# Test 3: Batch Processing
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 3: 📦 Batch Processing"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
curl -s -X POST $API_URL/api/mindmap/batch \
  -H "Content-Type: application/json" \
  -d @examples/batch_request.json | python3 -c "
import sys, json
data = json.load(sys.stdin)
print('✅ Success!' if data['success'] else '❌ Failed')
print(f'📊 Generated {len(data[\"data\"])} mind maps')
for i, mindmap in enumerate(data['data'], 1):
    print(f'  Map {i}: {mindmap[\"metadata\"][\"total_nodes\"]} nodes, {mindmap[\"metadata\"][\"total_edges\"]} edges')
"
echo ""
echo ""

# Test 4: Statistics
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 4: 📊 Detailed Statistics"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
curl -s -X POST $API_URL/api/mindmap/generate \
  -H "Content-Type: application/json" \
  -d @examples/sample_request.json | python3 -c "
import sys, json
data = json.load(sys.stdin)
if data['success']:
    metadata = data['data']['metadata']
    nodes = data['data']['nodes']
    edges = data['data']['edges']
    
    print('📈 Mind Map Metrics:')
    print('─' * 80)
    print(f'  Total Nodes:     {metadata[\"total_nodes\"]}')
    print(f'  Total Edges:     {metadata[\"total_edges\"]}')
    print(f'  Text Length:     {metadata[\"text_length\"]} characters')
    print()
    
    # Count by type
    from collections import Counter
    node_types = Counter(n['type'] for n in nodes)
    edge_types = Counter(e['type'] for e in edges)
    
    print('📊 Node Distribution:')
    for ntype, count in node_types.items():
        print(f'  {ntype.capitalize():12} {count}')
    print()
    
    print('🔗 Edge Types:')
    for etype, count in edge_types.items():
        print(f'  {etype.capitalize():12} {count}')
"
echo ""
echo ""

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                          ✅ All Tests Complete!                            ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "📚 For more examples, see:"
echo "   • examples/visualization.html - Web visualization demo"
echo "   • example_client.py - Python client usage"
echo "   • API_DOCUMENTATION.md - Full API reference"
echo ""
