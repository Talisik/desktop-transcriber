# Build Specs & GitHub Actions Workflows

## Spec Files Inventory

| Spec File | Status | Used In Workflow |
|---|---|---|
| `huey_worker_windows.spec` | **Active** | `build-windows-huey-worker.yml`, `build-windows.yml` |
| `model_profile_windows.spec` | **Active** | `build-windows-model-profile.yml` |
| `transcriber_huey_windows.spec` | Commented out | `build-windows-huey-worker.yml` |
| `resource_tracker_windows.spec` | Commented out | `build-windows-huey-worker.yml` |
| `model_manager_windows.spec` | Commented out | `build-windows-huey-worker.yml` |
| `queue_simulator_windows.spec` | Commented out | `build-windows-huey-worker.yml`, `build-windows.yml` |
| `huey_worker.spec` | Local only | Not in any workflow |
| `transcriber_huey.spec` | Local only | Not in any workflow |
| `queue_simulator.spec` | Local only | Not in any workflow |
| `resource_tracker.spec` | Local only | Not in any workflow |
| `main.spec` | Local only | Not in any workflow |

## GitHub Actions Workflows

### `build-windows-huey-worker.yml`
- **Triggers:** Push to `main`, `master`, `staging`, `face-recog/queue` | PRs to `main`/`master` | Manual
- **Builds:** `huey_worker_windows.spec`
- **Artifact:** `windows-huey-worker` (retention: 1 day)
- **Timeout:** 60 minutes

### `build-windows-model-profile.yml`
- **Triggers:** Push to `main`, `master`, `face-recog/queue` | PRs to `main`/`master` | Manual
- **Builds:** `model_profile_windows.spec`
- **Artifact:** `windows-model-profile` (retention: 7 days)
- **Timeout:** 30 minutes

### `build-windows.yml`
- **Triggers:** Manual only (`workflow_dispatch`)
- **Builds:** `huey_worker_windows.spec`
- **Artifact:** `windows-huey-worker` (retention: 1 day)
- **Note:** Effectively superseded by `build-windows-huey-worker.yml`. All automated triggers are disabled.

## Notes

- **Commented-out specs in `build-windows-huey-worker.yml`:** `transcriber_huey_windows`, `queue_simulator_windows`, `resource_tracker_windows`, `model_profile_windows`, `model_manager_windows` — these were previously built in a single workflow but are now either moved to dedicated workflows (model_profile) or disabled.
- **Local-only specs** (`huey_worker.spec`, `transcriber_huey.spec`, `queue_simulator.spec`, `resource_tracker.spec`, `main.spec`) are not referenced in any GitHub Actions workflow and are used for local development builds.
- **`build-windows.yml`** is a legacy workflow replaced by `build-windows-huey-worker.yml` which has better dependency handling (CUDA PyTorch, torchcodec, munchkin-chunker).
