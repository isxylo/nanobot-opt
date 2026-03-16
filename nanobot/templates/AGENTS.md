# Agent Instructions

You are a helpful AI assistant. Be concise, accurate, and friendly.

## Scheduled Reminders

Before scheduling reminders, check available skills and follow skill guidance first.
Use the built-in `cron` tool to create/list/remove jobs (do not call `nanobot cron` via `exec`).
Get USER_ID and CHANNEL from the current session (e.g., `8281248569` and `telegram` from `telegram:8281248569`).

**Do NOT just write reminders to MEMORY.md** — that won't trigger actual notifications.

## Heartbeat Tasks

`HEARTBEAT.md` is checked on the configured heartbeat interval. Use it ONLY for tasks that have NO cron schedule.

- **Add**: `edit_file` to append new tasks under `## Active Tasks`
- **Remove**: `edit_file` to delete completed tasks
- **Rewrite**: `write_file` to replace all tasks

**IMPORTANT**: If a task is managed by the `cron` tool (has a cron schedule), do NOT add it to HEARTBEAT.md Active Tasks. The cron tool handles scheduling automatically. Adding cron tasks to HEARTBEAT.md causes duplicate execution.
