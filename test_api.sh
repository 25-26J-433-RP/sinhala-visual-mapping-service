#!/bin/bash

# Test the Sinhala Mind Map API

API_URL="http://localhost:5000"

echo "================================"
echo "Sinhala Mind Map API Test Script"
echo "================================"
echo ""

# Test 1: Health Check
echo "Test 1: Health Check"
echo "--------------------"
curl -X GET "${API_URL}/health"
echo -e "\n\n"

# Test 2: Generate Mind Map with Direct Text
echo "Test 2: Generate Mind Map with Direct Text"
echo "-------------------------------------------"
curl -X POST "${API_URL}/api/mindmap/generate" \
  -H "Content-Type: application/json" \
  -d @examples/sample_request.json
echo -e "\n\n"

# Test 3: Batch Generation
echo "Test 3: Batch Generation"
echo "------------------------"
curl -X POST "${API_URL}/api/mindmap/batch" \
  -H "Content-Type: application/json" \
  -d @examples/batch_request.json
echo -e "\n\n"

echo "================================"
echo "Tests completed!"
echo "================================"
