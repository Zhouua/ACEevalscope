# Release Notes

## v0.4.0 - 2024-10-18

### 🆕 **New Features**
- **Modern Dependency Management**: Implemented pip-tools for automatic dependency locking
  - Added `requirements.in` for high-level dependencies
  - Auto-generated `requirements.txt` with 154 locked packages
  - Reproducible builds across all environments
  - Prevention of version mismatch bugs

- **Enhanced Retry Functionality**: Added tenacity dependency for robust error handling
  - Automatic retries on API failures
  - Configurable retry strategies for LiteLLM

### 🐛 **Bug Fixes**
- **Fixed Missing Dependency**: Added `tenacity>=8.0.0` to resolve LiteLLM retry import errors
- **Claude Parameter Compatibility**: Resolved top_p/temperature conflicts with Anthropic models
- **Accurate Documentation**: Fixed README code examples to use correct API imports

### 🔧 **Infrastructure Improvements**
- **Professional Contributor Workflow**:
  - Clear development setup instructions
  - pip-tools workflow for dependency management
  - Clean environment setup with pip-sync
- **Updated Documentation**:
  - Added dependency management section for contributors
  - Fixed API examples (`Generator, Reflector, Curator` vs non-existent `ACE` class)
  - Added pip-tools badge showing modern engineering practices

### 📦 **Dependencies Updated**
- `litellm`: 1.78.0 → 1.78.3 (latest stable with bug fixes)
- `pydantic`: 2.8.2 → 2.12.3 (major version update with improvements)
- `rich`: 13.7.1 → 14.2.0 (enhanced terminal output)
- `tenacity`: Added 9.1.2 (new dependency for retry functionality)

### ✅ **Verified Compatibility**
- All example scripts tested and working
- Kayba Test demo functional with updated dependencies
- LiteLLM integration with 100+ providers verified
- Clean environment setup tested

### 🚀 **Breaking Changes**
None - this release is fully backward compatible.

### 📚 **For Developers**
New pip-tools workflow:
```bash
# Update dependencies
pip-compile requirements.in

# Sync environment
pip-sync requirements.txt
```

This release significantly improves the reliability, maintainability, and developer experience of the ACE Framework while maintaining full backward compatibility.