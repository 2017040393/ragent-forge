# Project Positioning

RAGentForge is a local-first TUI application for building inspectable Agentic
RAG workflows over personal knowledge bases. The first users are developers who
want to work with their own Markdown and TXT notes, project documents, learning
materials, and interview preparation documents from the terminal.

RAGentForge is not a no-code platform, enterprise knowledge base, desktop app,
cloud service, or autonomous agent framework. Early releases should stay small,
legible, and useful for learning how RAG systems behave.

Local-first matters because personal knowledge often lives on a developer's
machine and may include private notes, drafts, and project context. The default
workflow should not require a hosted service, account, or hidden network call.

TUI-first keeps the application close to developer workflows. A terminal
interface is fast to iterate on, easy to run locally, and well matched to
inspectable text-heavy workflows.

Inspectability comes before autonomy. Users should be able to see what was
loaded, how text was chunked, which sources were retrieved, and what trace led
to an answer before the project attempts more autonomous behavior.

Python is the main implementation language for early versions because the AI and
RAG ecosystem moves quickly, and Python keeps experimentation lightweight.

Rust is intentionally deferred. It may become useful for performance-critical
modules later, but the project should wait for measured bottlenecks before
adding native extensions or a mixed-language build.
