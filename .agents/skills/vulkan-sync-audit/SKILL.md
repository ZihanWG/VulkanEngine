---
name: vulkan-sync-audit
description: Audit VulkanEngine synchronization and resource lifetimes. Use for barrier, image-layout, queue-ownership, semaphore, fence, frame-overlap, readback, indirect-buffer, Render Graph declaration, or use-after-free investigations and reviews.
---

# Vulkan Synchronization Audit

Reconstruct the actual execution and memory dependency chain before judging individual barriers.

## Scope and Safety

1. Resolve the repository root with `git rev-parse --show-toplevel` and work from there.
2. Read `AGENTS.md` completely and inspect `git status --short`.
3. Treat audit, review, explain, and diagnose requests as read-only. Edit only when the user explicitly asks for a fix.
4. Follow live code from `Renderer::drawFrame()` through command recording, queue submissions, presentation, and recreation. Use docs as supporting context.

## Build the Execution Model

Record the relevant sequence using these boundaries:

1. CPU wait and frame-slot reuse.
2. Resource update and command recording.
3. Graphics/compute submissions and semaphore waits/signals.
4. Presentation and deferred CPU reads.

For each involved resource, make a hazard table with producer, consumer, queue, stage, access, layout, frame index, and the mechanism that makes the edge safe.

## Audit Checklist

### Frame lifetime

- A frame-indexed buffer/image is not reused before its fence completes.
- Readback occurs only after the producing submission is complete.
- Swapchain recreation does not leave stale views, descriptors, framebuffers, or history validity.
- Partial initialization and feature-disable paths do not leak or destroy twice.

### Queue synchronization

- Binary/timeline semaphore waits cover the earliest consuming stage.
- Signals occur after the actual producer submission.
- Queue-family ownership or concurrent sharing is correct when families differ.
- CPU fences are not being used as GPU memory barriers.

### Memory and layouts

- Every write-to-read, write-to-write, or layout transition has a dependency with correct source/destination stage and access.
- Indirect and count buffers use indirect-command reads, not generic shader reads.
- Transfer, host read, and mapped-memory visibility are covered.
- No barrier relies on an invalid old layout or an overly narrow source stage.

### Render Graph boundary

- Declared reads/writes match commands recorded inside each pass.
- `beginDeclaredPass()` occurs before the Vulkan operations that require its inferred barriers.
- Manual barriers remain for intra-pass sequencing and resources outside graph ownership.
- Async compute is analyzed through renderer submissions; do not assume the graph schedules queues.

## Validate Findings

Search all writers, readers, descriptor bindings, copy operations, and recreation paths before reporting a hazard. Distinguish a correctness bug from a conservative/performance-only barrier.

When a fix is requested, make the smallest change that repairs the precise dependency and run `tools/dev/verify_renderer.sh full`. GPU validation-layer execution remains a separate check.

## Report Format

Lead with findings ordered by severity. For each finding include the exact file/line, failing producer-to-consumer edge, why existing synchronization is insufficient, and a concrete correction. Then list verified-safe edges, assumptions, and missing GPU evidence. Say explicitly when no correctness issue was found.
