# Local Configuration Sanitization

Before copying a local configuration or helper into `configs/examples/`,
review it for values that should remain private or machine-specific.

## Remove or replace

- API keys, access tokens, passwords, cookies, and authentication headers.
- Private hostnames, non-local network addresses, VPN names, and internal URLs.
- Absolute paths containing usernames, home directories, shared-drive names,
  or private repository locations.
- Personal identifiers, account names, email addresses, and workstation names.
- Private model repositories, signed download URLs, or credentials embedded in
  model paths.
- Benchmark prompts, test data, generated output, or evaluator inputs that are
  not intended for publication.
- Local logs and result files. Keep generated output under `logs/`, which is
  ignored by Git.

## Keep when useful

Concrete operating-system, hardware, engine, model, and sampler details are
fine when they help explain a working setup and are intentionally public. Add
model provenance such as a public source URL, revision, artifact filename, and
checksum when reproducibility matters.

## Before committing

1. Search the diff for secrets, usernames, absolute paths, private URLs, and
   unexpected generated output.
2. Confirm that every command and relative path works from the repository root.
3. Confirm that example files are clearly labeled as reference-only and do not
   look like universal defaults.
4. Check that copied examples do not silently change the benchmark case set or
   claim results that were produced with different settings.