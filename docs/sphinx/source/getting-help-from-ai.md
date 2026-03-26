# Getting help from AI

All pyvespa documentation is available in plain Markdown for easy consumption by LLMs and AI coding assistants.

## LLM-optimized documentation

Following the [llms.txt](https://llmstxt.org/) standard, we publish:

- **[llms.txt](https://vespa-engine.github.io/pyvespa/llms.txt)** -- Concise index of all docs with descriptions
- **[llms-full.txt](https://vespa-engine.github.io/pyvespa/llms-full.txt)** -- Complete documentation in a single file (~2 MB)

Every documentation page also has a Markdown version available by replacing `.html` with `.md.txt` in the URL.
Look for the Markdown icon button (top-right of each page) to view it.

## Instruct your AI assistant

Add the following to your project's `CLAUDE.md`, `AGENTS.md`, `.cursorrules`, or equivalent file to help your AI coding assistant use pyvespa docs effectively:

```markdown
## pyvespa

When looking up pyvespa documentation, prefer fetching the Markdown versions
instead of scraping HTML pages:

- Documentation index: https://vespa-engine.github.io/pyvespa/llms.txt
- Full documentation: https://vespa-engine.github.io/pyvespa/llms-full.txt
- Per-page markdown: replace `.html` with `.md.txt` in any docs URL

Examples:
- https://vespa-engine.github.io/pyvespa/reads-writes.md.txt
- https://vespa-engine.github.io/pyvespa/api/vespa/application.md.txt
```

!!! note "Compatibility"

    The snippet above works with [Claude Code](https://claude.ai/code) (`CLAUDE.md`),
    [GitHub Copilot](https://github.com/features/copilot) (`.github/copilot-instructions.md`),
    [Cursor](https://cursor.com) (`.cursor/rules/*.mdc`),
    [OpenAI Codex](https://openai.com/codex/) and most other AI coding tools that support
    [AGENTS.md](https://agents.md/).
    All of these read plain Markdown instructions.
