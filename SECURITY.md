# Security

## Reporting a vulnerability

Open a private GitHub security advisory on this repository, or email
the maintainer listed on the GitHub org. Do not file a public issue for
an unfixed vulnerability.

## What this project stores

Riftcount is a local CLI. It reads photos and videos you pass on the
command line, downloads cardvision weights into
`~/.cache/riftcount/weights`, and prints inventory JSON. It does not
upload media or call a hosted identification API.

## What not to commit

- Secrets, tokens, `.env` files, or private keys
- Extra personal photos beyond the checked-in examples
- cardvision weight files (fetch them with `riftcount update`)
