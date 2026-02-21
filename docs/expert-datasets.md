# Expert Datasets (Linux + Tool Calling)

Use a mix of permissive/public sources and your own synthetic data.
Always verify license terms before redistribution.

## Linux Expert Dataset Sources

1. **The Linux Documentation Project (TLDP)**
   - https://tldp.org/
   - Classic admin/how-to content.

2. **Arch Wiki (for learning, verify reuse policy)**
   - https://wiki.archlinux.org/
   - High quality troubleshooting and command docs.

3. **Debian/Ubuntu/Fedora official docs/manpages**
   - https://www.debian.org/doc/
   - https://help.ubuntu.com/
   - https://docs.fedoraproject.org/

4. **GNU / man-pages corpus on local machine**
   - Use `man` page exports as supervised reference snippets.

5. **Your own terminal traces/runbooks**
   - Best high-signal data for your environment.

## Tool-Calling Expert Dataset Sources

1. **OpenAPI specs from your tools/services**
   - Convert endpoint schemas into tool-call examples.

2. **JSON Schema corpora + function calling examples**
   - Build synthetic calls with valid/invalid argument pairs.

3. **Your OpenClaw tool traces (anonymized)**
   - Real calls for router/expert behavior.

4. **Synthetic generation from a teacher model**
   - Prompt teacher to produce:
     - user intent
     - correct tool name
     - exact JSON args
     - repair examples for malformed calls

## Recommended Dataset Format (jsonl)

```json
{"text":"Restart nginx safely on Ubuntu.","expert":"linux"}
{"text":"Call tool: browser.open with URL https://example.com","expert":"toolcalling"}
```

For richer supervision later:
```json
{"prompt":"...","response":"...","expert":"linux","metadata":{"source":"runbook"}}
```

## Minimum v1 volume targets
- Linux: 20k-50k high-quality instruction/response pairs
- Tool-calling: 15k-40k structured examples (include 20-30% negative/repair samples)

## Data quality rules
- Deduplicate aggressively
- Remove contradictory or unsafe command sequences
- Keep command explanations paired with command output expectations
- Include failure-and-recovery examples
