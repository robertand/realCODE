import requests
from bs4 import BeautifulSoup
import json
import re
from urllib.parse import urlparse, quote


class WebTool:
    name = "web"
    description = "Fetch and read content from web pages. Can search the internet or fetch specific URLs."

    schema = {
        "description": "Access internet to fetch web page content or search",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to fetch (optional if using search)"
                },
                "search": {
                    "type": "string",
                    "description": "Search query for web search (optional if using url)"
                },
                "max_length": {
                    "type": "integer",
                    "description": "Maximum characters to return (default: 10000)",
                    "default": 10000
                }
            }
        }
    }

    def execute(self, url: str = None, search: str = None, max_length: int = 10000) -> str:
        if search:
            return self._search_web(search, max_length)
        elif url:
            return self._fetch_url(url, max_length)
        else:
            return "[error: either url or search parameter is required]"

    def _search_web(self, query: str, max_length: int) -> str:
        """Search the web using DuckDuckGo (no API key required)"""
        try:
            # Use DuckDuckGo HTML search
            encoded_query = quote(query)
            url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = soup.find_all('a', class_='result__a')
            urls = soup.find_all('a', class_='result__url')
            
            output = [f"Search results for: {query}\n"]
            for i, (title, link) in enumerate(zip(results[:10], urls[:10]), 1):
                title_text = title.get_text(strip=True)
                url_text = link.get('href', '') if link else ''
                if url_text.startswith('/'):
                    url_text = 'https://duckduckgo.com' + url_text
                output.append(f"{i}. {title_text}")
                output.append(f"   URL: {url_text}\n")
            
            result_text = "\n".join(output)
            if len(result_text) > max_length:
                result_text = result_text[:max_length] + "\n... (truncated)"
            
            return result_text
            
        except Exception as e:
            return f"[error searching web: {e}]"

    def _fetch_url(self, url: str, max_length: int) -> str:
        """Fetch and extract text content from a URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=20, allow_redirects=True)
            response.raise_for_status()
            
            # Try to parse HTML
            content_type = response.headers.get('content-type', '').lower()
            
            if 'text/html' in content_type:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style", "nav", "footer", "header"]):
                    script.decompose()
                
                # Get text
                text = soup.get_text()
                
                # Clean up whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                # Also get title
                title = soup.find('title')
                if title:
                    text = f"Title: {title.get_text()}\n\n{text}"
                    
            elif 'application/json' in content_type:
                try:
                    data = response.json()
                    text = json.dumps(data, indent=2)
                except:
                    text = response.text[:max_length]
            else:
                text = response.text
            
            if len(text) > max_length:
                text = text[:max_length] + "\n... (truncated)"
            
            return text
            
        except requests.exceptions.Timeout:
            return "[error: request timed out]"
        except requests.exceptions.RequestException as e:
            return f"[error fetching url: {e}]"
        except Exception as e:
            return f"[error: {e}]"