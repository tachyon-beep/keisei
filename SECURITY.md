# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.2.x   | Yes                |
| < 0.2   | No                 |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it privately by emailing
the repository owner rather than opening a public issue.

Include:

- Description of the vulnerability
- Steps to reproduce
- Potential impact

You should receive an acknowledgement within 72 hours.

## Scope

Keisei is a research project for training Shogi-playing neural networks. The
primary attack surface is:

- The FastAPI web server (`keisei-serve`), which serves a spectator WebUI
- File I/O (checkpoint loading, config parsing, SFEN/CSA parsing)

The training loop and model code are not exposed to untrusted input in normal
operation.
