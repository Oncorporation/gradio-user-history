---
name: file-discovery
description: "Use when: discover media files and extract metadata from a folder for the MediaGallery pipeline."
---

# File Discovery Agent

Name: File Discovery Agent
Purpose: Walk a user-specified folder, discover supported media files, extract metadata, and return a `files_info` list compatible with the existing MediaGallery pipeline.

Skills required:
- python-pro
- mcp-developer
- gradio-expert (for UI wiring guidance)

Triggers:
- User provides a folder path in the UI
- CLI / automated import job

Entrypoint function:
- `discover_folder_files(folder_path: str) -> list[dict]`

Expected outputs:
- A list of file info dicts, each containing: `filename`, `filepath`, `type` ("image"/"video"/"audio"), `width`, `height`, `duration_sec` (video/audio), `mime`, `created_at`
- Compatible with `normalize_files()` and `get_files_infos()` in repo

Implementation notes:
- Use `pathlib.Path.rglob()` to walk the directory and filter by `allowed_medias` from `app.py`.
- Use `Pillow` (`PIL.Image`) for image dimensions, `moviepy` for video/audio duration and dimensions, and `python-magic` or `mimetypes` fallback for mime type detection.
- Respect file size and duration limits described in README (file size limit, max duration).
- Support stable ordering (by filename or file created time) to make deterministic plans.

Testing:
- Provide pytest unit tests under `tests/test_file_discovery.py` mocking a temporary directory with sample files.

Deployment:
- Place implementation stub in `utils.py` (function name: `discover_folder_files`).

Security:
- Do not follow symlinks outside folder root unless explicitly allowed.
- Validate path input to avoid path-traversal.
- When running as part of an MCP server, prefer using the MCP file-system service (FSS) to access files in approved locations rather than direct disk access.
  - Configure and honor an `allowed_paths` / whitelist (examples: `C:\Users\CharlesFettinger\.github\agents`, project media folders) so the agent only reads from approved roots.
  - Reject or sanitize user-supplied paths that reference locations outside the configured allowed paths.
  - Do not enable recursive traversal of system roots (e.g., `C:\` or `/`) from untrusted inputs.
  - Log and audit all FSS file accesses for traceability.

Example prompt for the agent (if exposed to an LLM-backed subagent):
"Given a folder path `C:/Users/Me/Pictures/trip`, find all supported media files, extract dimensions and duration, and return a JSON array of file-info objects compatible with the project's `files_info` schema."
