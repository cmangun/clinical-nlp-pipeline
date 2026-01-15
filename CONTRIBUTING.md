# Contributing to clinical-nlp-pipeline

Thank you for your interest in contributing! This project aims to make healthcare AI more accessible and compliant.

## Getting Started

### Prerequisites
- Python 3.9+
- pip or poetry
- Git

### Setup

```bash
# Clone the repo
git clone https://github.com/cmangun/healthcare-rag-platform.git
cd healthcare-rag-platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## How to Contribute

### Reporting Bugs
- Use GitHub Issues
- Include Python version, OS, and steps to reproduce
- Attach relevant logs or error messages

### Suggesting Features
- Open an issue with the `enhancement` label
- Describe the use case and expected behavior
- Bonus: include pseudocode or examples

### Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass: `pytest`
6. Run linting: `black . && ruff check .`
7. Commit with clear messages: `git commit -m "feat: add PHI redaction for SSN"`
8. Push to your fork: `git push origin feature/your-feature`
9. Open a Pull Request

### Commit Message Format
We follow [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation only
- `test:` - Adding tests
- `refactor:` - Code change that neither fixes a bug nor adds a feature

## Code Style

- **Formatter:** Black (line length 88)
- **Linter:** Ruff
- **Type hints:** Required for all public functions
- **Docstrings:** Google style

```python
def detect_phi(text: str, identifiers: list[str] | None = None) -> PHIResult:
    """Detect Protected Health Information in text.
    
    Args:
        text: Clinical text to analyze.
        identifiers: Optional list of PHI types to detect.
    
    Returns:
        PHIResult containing detected entities and risk level.
    
    Raises:
        ValueError: If text is empty.
    """
```

## Areas We Need Help

- [ ] Additional PHI detection patterns (international formats)
- [ ] Performance optimization for large documents
- [ ] Integration tests with real LLM providers
- [ ] Documentation improvements
- [ ] Example notebooks

## Questions?

- Open an issue with the `question` label
- Connect on [LinkedIn](https://linkedin.com/in/christophermangun)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
