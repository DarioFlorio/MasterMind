# Lessons Learned: Sports News Compilation Task (2026-05-30)

## Summary
Technical issues encountered during the task to fetch sports news from BBC, Sky, and CNN, then compile into a Word document.

## Issues & Fixes

### 1. `web_fetch` Failures (NoneType/get/404)
- **Root Cause:** Dynamic URLs or regional restrictions (e.g., CNN Sports).
- **Fix:**
  - Verified URLs via `web_search` before fetching.
  - Substituted CNN Sports with Bleacher Report (CNN-affiliated).
  - Used direct article URLs instead of generic pages.

### 2. `task_update`/`memory_write` Errors
- **Root Cause:** Misalignment with system task tracking (e.g., invalid task ID).
- **Fix:**
  - Used manual status tracking (`memory_write` with explicit keys).
  - Validated tool parameters before calls.

### 3. Context Overflow
- **Root Cause:** Large HTML responses and high tool call volume.
- **Fix:**
  - Structured offloading to disk at 70% capacity.
  - Reduced redundancy in tool calls.

## Lessons
1. **Pre-validate URLs** before `web_fetch` to avoid 404s.
2. **Maintain fallback sources** (e.g., Bleacher Report for CNN).
3. **Validate tool parameters** (e.g., task IDs) before calls.
4. **Proactively offload context** to disk at 70% capacity.

## Future Improvements
- Add pre-call tool validation (e.g., check task ID existence).
- Automate context offloading at 70% capacity.
- Update internal documentation with common failure modes.

## Verification
- All fixes applied during the task.
- No recurring issues in subsequent steps.