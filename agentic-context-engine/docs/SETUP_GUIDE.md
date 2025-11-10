# ⚙️ ACE Framework Setup Guide

Comprehensive setup and configuration guide for ACE Framework.

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [API Keys Configuration](#api-keys-configuration)
- [Provider Setup](#provider-setup)
- [Advanced Configuration](#advanced-configuration)
- [Troubleshooting](#troubleshooting)

## Requirements

### System Requirements
- Python 3.9 or higher
- pip (Python package manager)
- 4GB RAM minimum (8GB recommended for local models)
- Internet connection for API-based models

### Python Version Check
```bash
python --version  # Should show 3.9 or higher
```

If you need to upgrade Python:
- **macOS**: `brew install python@3.11`
- **Ubuntu/Debian**: `sudo apt update && sudo apt install python3.11`
- **Windows**: Download from [python.org](https://python.org)

## Installation

### Basic Installation
```bash
pip install ace-framework
```

### Installation with Extras

#### For LangChain Support
```bash
pip install ace-framework[langchain]
```

#### For All Features
```bash
pip install ace-framework[all]
```

#### For Development
```bash
pip install ace-framework[dev]
```

### Installing from Source
```bash
git clone https://github.com/Kayba-ai/agentic-context-engine.git
cd agentic-context-engine
pip install -e .
```

### Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv ace-env

# Activate it
# On macOS/Linux:
source ace-env/bin/activate
# On Windows:
ace-env\Scripts\activate

# Install ACE
pip install ace-framework
```

## API Keys Configuration

### Method 1: Environment Variables

Create a `.env` file in your project root:

```bash
# OpenAI
OPENAI_API_KEY=sk-...

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Google
GOOGLE_API_KEY=AIza...

# Cohere
COHERE_API_KEY=...

# Hugging Face
HUGGINGFACE_API_KEY=hf_...

# AWS Bedrock
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
```

Load in Python:
```python
from dotenv import load_dotenv
load_dotenv()
```

### Method 2: Direct Configuration

```python
import os

os.environ["OPENAI_API_KEY"] = "your-key-here"
```

### Method 3: Client Configuration

```python
from ace import LiteLLMClient

client = LiteLLMClient(
    model="gpt-4",
    api_key="your-key-here"
)
```

## Provider Setup

### OpenAI

1. Get API key from [platform.openai.com](https://platform.openai.com)
2. Set environment variable:
```bash
export OPENAI_API_KEY="sk-..."
```
3. Test connection:
```python
from ace import LiteLLMClient
client = LiteLLMClient(model="gpt-3.5-turbo")
response = client.complete("Hello!")
print(response.text)
```

### Anthropic Claude

1. Get API key from [console.anthropic.com](https://console.anthropic.com)
2. Set environment variable:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```
3. Test connection:
```python
client = LiteLLMClient(model="claude-3-haiku-20240307")
```

### Google Gemini

1. Get API key from [makersuite.google.com](https://makersuite.google.com/app/apikey)
2. Set environment variable:
```bash
export GOOGLE_API_KEY="AIza..."
```
3. Test connection:
```python
client = LiteLLMClient(model="gemini-pro")
```

### Azure OpenAI

```python
client = LiteLLMClient(
    model="azure/gpt-4",
    api_base="https://your-resource.openai.azure.com",
    api_version="2024-02-15-preview",
    api_key="your-azure-key"
)
```

### AWS Bedrock

```bash
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_REGION="us-east-1"
```

```python
client = LiteLLMClient(model="bedrock/claude-3-haiku")
```

### Local Models (Ollama)

1. Install Ollama:
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh
```

2. Pull a model:
```bash
ollama pull llama2
```

3. Use with ACE:
```python
client = LiteLLMClient(
    model="ollama/llama2",
    api_base="http://localhost:11434"
)
```

### OpenRouter (100+ Models)

1. Get API key from [openrouter.ai](https://openrouter.ai)
2. Configure:
```python
client = LiteLLMClient(
    model="openrouter/openai/gpt-4",
    api_key="your-openrouter-key",
    api_base="https://openrouter.ai/api/v1"
)
```

## Advanced Configuration

### Proxy Configuration

For corporate environments:
```python
import os
os.environ["HTTP_PROXY"] = "http://proxy.company.com:8080"
os.environ["HTTPS_PROXY"] = "http://proxy.company.com:8080"
```

### Custom Endpoints

```python
client = LiteLLMClient(
    model="gpt-4",
    api_base="https://your-custom-endpoint.com/v1"
)
```

### Rate Limiting

```python
from ace import LiteLLMClient
import time

class RateLimitedClient(LiteLLMClient):
    def complete(self, prompt, **kwargs):
        time.sleep(0.5)  # 500ms delay
        return super().complete(prompt, **kwargs)
```

### Retry Configuration

```python
client = LiteLLMClient(
    model="gpt-4",
    max_retries=5,
    timeout=60
)
```

### Cost Tracking

```python
from ace.llm_providers import LiteLLMConfig

config = LiteLLMConfig(
    model="gpt-4",
    track_cost=True,
    max_budget=10.0  # Stop after $10
)

client = LiteLLMClient(config=config)
```

### Logging Setup

```python
import logging

# Enable detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# ACE-specific logging
logger = logging.getLogger('ace')
logger.setLevel(logging.DEBUG)
```

## Docker Setup

### Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir ace-framework

COPY . .

CMD ["python", "main.py"]
```

### docker-compose.yml
```yaml
version: '3.8'

services:
  ace:
    build: .
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    volumes:
      - ./playbooks:/app/playbooks
```

## Production Deployment

### Environment Variables

```bash
# Required
ACE_LLM_MODEL=gpt-4
ACE_API_KEY=your-key

# Optional
ACE_TEMPERATURE=0.7
ACE_MAX_TOKENS=1000
ACE_FALLBACK_MODELS=claude-3-haiku,gpt-3.5-turbo
ACE_LOG_LEVEL=INFO
ACE_PLAYBOOK_PATH=/data/playbooks
```

### Systemd Service

```ini
[Unit]
Description=ACE Framework Service
After=network.target

[Service]
Type=simple
User=ace
WorkingDirectory=/opt/ace
EnvironmentFile=/etc/ace/ace.env
ExecStart=/usr/bin/python3 /opt/ace/main.py
Restart=always

[Install]
WantedBy=multi-user.target
```

### Health Check

```python
from ace import LiteLLMClient

def health_check():
    try:
        client = LiteLLMClient(model="gpt-3.5-turbo")
        response = client.complete("ping", max_tokens=5)
        return {"status": "healthy", "model": "gpt-3.5-turbo"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

## Troubleshooting

### Common Issues

#### ImportError: No module named 'ace'
```bash
pip install --upgrade ace-framework
```

#### API Key Not Found
```python
import os
print(os.environ.get("OPENAI_API_KEY"))  # Check if set
```

#### SSL Certificate Error
```python
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```

#### Rate Limiting
```python
import time

def retry_with_backoff(func, max_retries=3):
    for i in range(max_retries):
        try:
            return func()
        except Exception as e:
            if "rate_limit" in str(e).lower():
                time.sleep(2 ** i)  # Exponential backoff
            else:
                raise
```

#### Memory Issues
```python
# Use smaller models
client = LiteLLMClient(model="gpt-3.5-turbo")

# Process in batches
def process_in_batches(items, batch_size=10):
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        yield process_batch(batch)
```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# LiteLLM verbose mode
import litellm
litellm.set_verbose = True
```

### Performance Optimization

```python
# Use caching
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_complete(prompt):
    return client.complete(prompt)

# Use async operations
import asyncio

async def process_many(prompts):
    tasks = [client.acomplete(p) for p in prompts]
    return await asyncio.gather(*tasks)
```

## Next Steps

- Read the [Quick Start Guide](QUICK_START.md)
- Explore [API Reference](API_REFERENCE.md)
- Try the [examples](../examples/)
- Join our [Discord community](#)

## Support

- GitHub Issues: [github.com/Kayba-ai/agentic-context-engine/issues](https://github.com/Kayba-ai/agentic-context-engine/issues)
- Documentation: [ace-framework.readthedocs.io](https://ace-framework.readthedocs.io)
- Email: support@kayba.ai