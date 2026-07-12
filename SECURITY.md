# Security Policy

## Supported Versions

CiteRAG does not currently publish versioned releases or a support matrix.
The `main` branch is the maintained line.

## Reporting a Vulnerability

We take the security of CiteRAG seriously. If you believe you have found a security vulnerability, please report it to us responsibly.

**Please do NOT report security vulnerabilities through public GitHub issues.**

### How to Report

Please report security vulnerabilities using GitHub's private vulnerability reporting:

1. Go to the [Security tab](https://github.com/melbinjp/citerag/security/advisories)
2. Click "Report a vulnerability"
3. Fill out the form with details

**Or email**: melbinjpaulose@gmail.com with subject "SECURITY: CiteRAG Vulnerability"

### What to Include

- Type of vulnerability
- Full paths of source file(s) related to the vulnerability
- Location of the affected source code (tag/branch/commit or direct URL)
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

### What to Expect

- We will review the report and follow up when we can validate the issue.
- We will coordinate a fix or mitigation with the reporter when practical.
- Public acknowledgment is optional and should be agreed with the reporter.

## Security Best Practices

When deploying CiteRAG:

1. **Environment Variables**: Never commit `.env` files with real API keys to version control
2. **API Keys**: Rotate API keys regularly and use keys with minimal required permissions
3. **Network Security**: Run behind a reverse proxy (nginx, Caddy) in production
4. **Updates**: Keep dependencies up to date to patch known vulnerabilities
5. **Access Control**: Implement authentication/authorization if exposing publicly
6. **Rate Limiting**: Add rate limiting to prevent abuse
7. **Input Validation**: The app validates inputs, but review upload restrictions for your use case
8. **HTTPS**: Always use HTTPS in production
9. **CORS**: Configure CORS appropriately for your deployment environment
10. **Monitoring**: Monitor logs for suspicious activity

## Known Security Considerations

- **API Keys**: The application requires Google API keys. Protect these credentials.
- **File Uploads**: PDF uploads are processed server-side. Only accept uploads from trusted users.
- **Vector Store**: Qdrant access should be restricted in production deployments.
- **LLM Prompts**: User queries are sent to the LLM provider. Consider content filtering for public deployments.

## Dependencies

Dependency vulnerabilities should be checked with the package-manager tooling
before a public deployment, including `pip-audit` for Python dependencies and
`npm audit` for JavaScript dependencies.

## Updates

Security changes will be documented in the repository history and release notes
when versioned releases are introduced.
