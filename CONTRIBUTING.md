# Contributing to CiteRAG

Thank you for your interest in contributing to CiteRAG! We welcome contributions from the community.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/citerag.git`
   - Replace `YOUR_USERNAME` with your GitHub username
   - Original repo: `https://github.com/melbinjp/citerag.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Test your changes thoroughly
6. Commit your changes: `git commit -m "Add: brief description of changes"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Open a Pull Request

## Development Setup

### Backend

```bash
cd backend
pip install -r requirements.txt
python -m pytest tests/ -v
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Docker

```bash
docker compose up --build
```

## Code Standards

### Python (Backend)

- Follow PEP 8 style guidelines
- Add type hints to function signatures
- Write docstrings for modules, classes, and functions
- Keep functions focused and single-purpose
- Add tests for new functionality

### JavaScript/React (Frontend)

- Use functional components with hooks
- Follow the existing component structure
- Keep components small and reusable
- Add proper prop validation
- Ensure accessibility (ARIA labels, keyboard navigation)

## Testing

- All new features should include tests
- Run the test suite before submitting: `pytest tests/ -v`
- Ensure all tests pass
- Add integration tests for end-to-end features

## Pull Request Guidelines

- **Title**: Use a clear, descriptive title
- **Description**: Explain what changes you made and why
- **Testing**: Describe how you tested your changes
- **Screenshots**: Include screenshots for UI changes
- **Breaking Changes**: Clearly mark any breaking changes

## Reporting Issues

When reporting issues, please include:

- A clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, browser, etc.)
- Error messages and logs
- Screenshots if applicable

## Feature Requests

We welcome feature requests! Please:

- Check if the feature already exists or is planned
- Clearly describe the feature and its use case
- Explain why it would be valuable
- Consider proposing an implementation approach

## Code Review Process

- All PRs require review before merging
- Address reviewer feedback promptly
- Keep PRs focused and reasonably sized
- Squash commits if requested

## Questions?

- Open an [issue](https://github.com/melbinjp/citerag/issues) for questions or discussions
- For security concerns, see [SECURITY.md](SECURITY.md)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
