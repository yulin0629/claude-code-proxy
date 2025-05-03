# Anthropic API Proxy for Gemini, OpenAI & OpenRouter Models 🔄

**Use Anthropic clients (like Claude Code) with Gemini, OpenAI, or OpenRouter backends.** 🤝

A proxy server that lets you use Anthropic clients with Gemini, OpenAI, or OpenRouter models via LiteLLM. 🌉


![Anthropic API Proxy](pic.png)

## Quick Start ⚡

### Prerequisites

- OpenAI API key 🔑
- Google AI Studio (Gemini) API key (if using Google provider) 🔑
- OpenRouter API key (if using OpenRouter provider) 🔑
- [uv](https://github.com/astral-sh/uv) installed.

### Setup 🛠️

1. **Clone this repository**:
   ```bash
   git clone https://github.com/1rgs/claude-code-openai.git
   cd claude-code-openai
   ```

2. **Install uv** (if you haven't already):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   *(`uv` will handle dependencies based on `pyproject.toml` when you run the server)*

3. **Configure Environment Variables**:
   Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and fill in your API keys and model configurations:

   *   `ANTHROPIC_API_KEY`: (Optional) Needed only if proxying *to* Anthropic models.
   *   `OPENAI_API_KEY`: Your OpenAI API key (Required if using the default OpenAI preference or as fallback).
   *   `GEMINI_API_KEY`: Your Google AI Studio (Gemini) API key (Required if PREFERRED_PROVIDER=google).
   *   `OPENROUTER_API_KEY`: Your OpenRouter API key (Required if PREFERRED_PROVIDER=openrouter).
   *   `OPENROUTER_API_BASE`: Your OpenRouter endpoint (Optional, defaults to "https://openrouter.ai/api/v1").
   *   `PREFERRED_PROVIDER` (Optional): Set to `openai` (default), `google`, or `openrouter`. This determines the primary backend for mapping `haiku`/`sonnet`.
   *   `BIG_MODEL` (Optional): The model to map `sonnet` requests to. 
   *   `SMALL_MODEL` (Optional): The model to map `haiku` requests to.

   **Default Model Mappings:**
   - OpenAI: `gpt-4.1` (BIG) and `gpt-4.1-mini` (SMALL)
   - Google: `gemini-2.5-pro-preview-03-25` (BIG) and `gemini-2.0-flash` (SMALL)
   - OpenRouter: `anthropic/claude-3-opus` (BIG) and `anthropic/claude-3-haiku` (SMALL)

   **Custom API Endpoints:**
   - You can also specify custom API endpoints for each provider if needed:
     - `OPENAI_API_BASE`
     - `ANTHROPIC_API_BASE`
     - `GEMINI_API_BASE`

4. **Run the server**:
   ```bash
   uv run uvicorn server:app --host 0.0.0.0 --port 8082 --reload
   ```
   *(`--reload` is optional, for development)*

### Using with Claude Code 🎮

1. **Install Claude Code** (if you haven't already):
   ```bash
   npm install -g @anthropic-ai/claude-code
   ```

2. **Connect to your proxy**:
   ```bash
   ANTHROPIC_BASE_URL=http://localhost:8082 claude
   ```

3. **That's it!** Your Claude Code client will now use the configured backend models through the proxy. 🎯

### Claude Code Guidelines

This repository includes a `CLAUDE.md` file that provides guidance for using Claude Code with this project. It covers:

- Build & run commands
- Test commands with options
- Code style guidelines
- Development best practices

## Model Mapping 🗺️

The proxy automatically maps Claude models to either OpenAI, Gemini, or OpenRouter models based on your configuration:

| Claude Model | OpenAI Default | Gemini Default | OpenRouter Default |
|--------------|---------------|----------------|-------------------|
| haiku | openai/gpt-4.1-mini | gemini/gemini-2.0-flash | openrouter/anthropic/claude-3-haiku |
| sonnet | openai/gpt-4.1 | gemini/gemini-2.5-pro-preview-03-25 | openrouter/anthropic/claude-3-opus |

### Supported Models

#### OpenAI Models
The following OpenAI models are supported with automatic `openai/` prefix handling:
- o3-mini
- o1
- o1-mini
- o1-pro
- gpt-4.5-preview
- gpt-4o
- gpt-4o-audio-preview
- chatgpt-4o-latest
- gpt-4o-mini
- gpt-4o-mini-audio-preview
- gpt-4.1
- gpt-4.1-mini

#### Gemini Models
The following Gemini models are supported with automatic `gemini/` prefix handling:
- gemini-2.5-pro-preview-03-25
- gemini-2.0-flash

#### OpenRouter Models
OpenRouter models are supported with automatic `openrouter/` prefix handling.

### Model Prefix Handling
The proxy automatically adds the appropriate prefix to model names:
- OpenAI models get the `openai/` prefix 
- Gemini models get the `gemini/` prefix
- OpenRouter models get the `openrouter/` prefix
- The BIG_MODEL and SMALL_MODEL will get the appropriate prefix based on the provider

For example:
- `gpt-4o` becomes `openai/gpt-4o`
- `gemini-2.5-pro-preview-03-25` becomes `gemini/gemini-2.5-pro-preview-03-25`
- `anthropic/claude-3-opus` becomes `openrouter/anthropic/claude-3-opus` when using OpenRouter

### Customizing Model Mapping

Control the mapping using environment variables in your `.env` file or directly:

**Example 1: Default (Use OpenAI)**
No changes needed in `.env` beyond API keys, or ensure:
```dotenv
OPENAI_API_KEY="your-openai-key"
GEMINI_API_KEY="your-google-key" # Needed if PREFERRED_PROVIDER=google
# PREFERRED_PROVIDER="openai" # Optional, it's the default
# BIG_MODEL="gpt-4.1" # Optional, it's the default
# SMALL_MODEL="gpt-4.1-mini" # Optional, it's the default
```

**Example 2: Prefer Google**
```dotenv
GEMINI_API_KEY="your-google-key"
OPENAI_API_KEY="your-openai-key" # Needed for fallback
PREFERRED_PROVIDER="google"
# BIG_MODEL="gemini-2.5-pro-preview-03-25" # Optional, it's the default for Google pref
# SMALL_MODEL="gemini-2.0-flash" # Optional, it's the default for Google pref
```

**Example 3: Use OpenRouter**
```dotenv
OPENROUTER_API_KEY="your-openrouter-key"
OPENAI_API_KEY="your-openai-key" # Needed for fallback
PREFERRED_PROVIDER="openrouter"
# BIG_MODEL="anthropic/claude-3-opus" # Optional, it's the default for OpenRouter pref
# SMALL_MODEL="anthropic/claude-3-haiku" # Optional, it's the default for OpenRouter pref
```

## Recent Updates

### OpenRouter Provider Support (fa2a9b7)
- Added OpenRouter as a supported provider with automatic model mapping
- Updated LiteLLM dependency to version 1.67.4
- Enhanced error handling and logging for better troubleshooting
- Added OpenRouter configuration in .env.example and model mapping logic

### Model Mapping Logic Improvement (c01ded7)
- Updated PREFERRED_PROVIDER default to empty string for more intuitive behavior
- Adjusted model mapping logic to better handle provider selection
- Improved error messages to include model name for easier debugging

### Testing Framework Enhancements (1c77b3b, cb0fc87)
- Removed deprecated streaming tests and adjusted server behavior
- Updated test scenarios to use latest Claude models
- Added test for claude-small model with the calculator tool
- Added filtering option to test specific scenarios
- Better validation of test results with detailed statistics

### Claude Code Integration (cb0fc87)
- Added CLAUDE.md with guidance for Claude Code users
- Includes build & run commands, test options, and code style guidelines
- Formatted for easy use with claude.ai/code

## Running Tests

The project includes a comprehensive test suite to verify functionality:

```bash
# Run all tests
python tests.py

# Run only simple tests (no tools)
python tests.py --simple

# Skip streaming tests
python tests.py --no-streaming

# Only run tests with a specific name
python tests.py --filter "calculator"
```

## Contributing

Contributions are welcome! Please follow the code style guidelines in CLAUDE.md when submitting changes.
