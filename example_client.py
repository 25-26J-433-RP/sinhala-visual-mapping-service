"""
Example Python client for Sinhala Mind Map API
"""

import requests
import json


class SinhalaMindMapClient:
    """Client for interacting with Sinhala Mind Map API."""
    
    def __init__(self, base_url='http://localhost:5000'):
        """
        Initialize the client.
        
        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url.rstrip('/')
    
    def health_check(self):
        """Check if the API is healthy."""
        response = requests.get(f'{self.base_url}/health')
        return response.json()
    
    def generate_mindmap(self, text):
        """
        Generate mind map from Sinhala text.
        
        Args:
            text: Sinhala text to process
            
        Returns:
            Dictionary with mind map data
        """
        response = requests.post(
            f'{self.base_url}/api/mindmap/generate',
            json={'text': text}
        )
        return response.json()
    
    def generate_from_external_api(self, api_url, api_key=None):
        """
        Generate mind map from external API.
        
        Args:
            api_url: URL of external API
            api_key: Optional API key for authentication
            
        Returns:
            Dictionary with mind map data
        """
        payload = {'external_api_url': api_url}
        if api_key:
            payload['api_key'] = api_key
        
        response = requests.post(
            f'{self.base_url}/api/mindmap/generate',
            json=payload
        )
        return response.json()
    
    def batch_generate(self, texts):
        """
        Generate multiple mind maps.
        
        Args:
            texts: List of Sinhala texts
            
        Returns:
            Dictionary with list of mind map data
        """
        response = requests.post(
            f'{self.base_url}/api/mindmap/batch',
            json={'texts': texts}
        )
        return response.json()


# Example usage
if __name__ == '__main__':
    # Initialize client
    client = SinhalaMindMapClient()
    
    # Example 1: Health check
    print("=" * 50)
    print("Health Check")
    print("=" * 50)
    health = client.health_check()
    print(json.dumps(health, indent=2, ensure_ascii=False))
    print()
    
    # Example 2: Generate mind map from text
    print("=" * 50)
    print("Generate Mind Map from Text")
    print("=" * 50)
    text = """ශ්‍රී ලංකාව දකුණු ආසියාවේ පිහිටි දිවයිනකි. 
    එය සුන්දර වෙරළ තීරයන්, පුරාණ නටබුන් සහ පොහොසත් සංස්කෘතියකින් යුක්තය. 
    ශ්‍රී ලංකාවේ ජනගහනය මිලියන 22 කි. 
    රට බෞද්ධ ආගමික උරුමයන්ගෙන් පොහොසත්ය. 
    කොළඹ වාණිජ අගනුවර වන අතර ශ්‍රී ජයවර්ධනපුර කෝට්ටේ පරිපාලන අගනුවරයි."""
    
    result = client.generate_mindmap(text)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print()
    
    # Example 3: Batch generation
    print("=" * 50)
    print("Batch Generate Mind Maps")
    print("=" * 50)
    texts = [
        "පරිගණකය යනු ඉලෙක්ට්‍රොනික උපකරණයකි. එය දත්ත සකසයි සහ ගබඩා කරයි.",
        "ශ්‍රී ලංකාව සංචාරක ගමනාන්ත සඳහා ප්‍රසිද්ධය. අපට සුන්දර වෙරළ තීරයන් සහ කඳු ඇත."
    ]
    
    batch_result = client.batch_generate(texts)
    print(json.dumps(batch_result, indent=2, ensure_ascii=False))
    print()
    
    # Print summary
    if result.get('success'):
        metadata = result['data']['metadata']
        print("=" * 50)
        print("Summary")
        print("=" * 50)
        print(f"Total nodes: {metadata['total_nodes']}")
        print(f"Total edges: {metadata['total_edges']}")
        print(f"Text length: {metadata['text_length']}")
