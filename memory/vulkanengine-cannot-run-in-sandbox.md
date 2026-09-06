---
name: vulkanengine-cannot-run-in-sandbox
description: The Vulkan engine CAN launch from a session on the user's Mac (log-verified); visual correctness still needs the user's eyes
metadata: 
  node_type: memory
  type: project
  originSessionId: 23f06752-0feb-47dc-97c2-9bdca3b36d09
  modified: 2026-08-01T08:55:44.751Z
---

Update (2026-07-06): the engine DOES launch from a Claude Code session on the user's Mac (Apple M3, MoltenVK, validation layers active). Verified by running `./build/debug/VulkanEngine` in the background for ~20s, killing it, and inspecting the captured stdout log — full startup, GPU frame timings, and validation output all appear. An earlier session (2026-06) could not launch it; that limitation no longer holds, at least for log-based verification.

**How to apply:** For runtime/validation-layer verification, run the binary briefly in the background redirecting output to the scratchpad, then grep the log for "Validation Error"/VUID. Note: `timeout` is not available in zsh here, and the Bash tool blocks foreground `sleep` — use perl instead: `perl -e 'alarm(12); exec @ARGV' ./build/debug/VulkanEngine > log 2>&1`. Visual correctness (does it *look* right) still needs the user to check on screen. Builds and unit tests (`ctest --preset ci-debug`) run headless as always.

Update (2026-08-01): **SIGTERM gives a graceful shutdown**, not a hard kill — SDL turns it into SDL_EVENT_QUIT, the run loop exits, and destructors run. That is the only way from a session to exercise teardown-only code paths (pipeline cache save, resource cleanup logs). Use `perl -e '$p=fork; if($p==0){open(STDOUT,">","log");open(STDERR,">&",STDOUT);exec @ARGV} sleep 12; kill "TERM",$p; sleep 4; kill "KILL",$p; waitpid($p,0)' ./build/debug/VulkanEngine` and confirm the teardown line appears in the log. SIGALRM/SIGKILL skip teardown entirely.

Update (2026-08-19): **`--capture-frame` does NOT include ImGui.** A capture taken
with the culling panel enabled (`config/runtime_settings.json` with
`debugUi.advancedMode` + `showCullingStats` true, both persisted so they can be
scripted) contains the rendered scene only — no debug UI anywhere in the 2560x1440
PNG. So the capture path could verify *rendering* only.

**Superseded 2026-08-23: `--capture-include-ui` takes the copy AFTER the ImGui
pass, so debug-UI readouts ARE verifiable from a session now** — the amber
frame-capacity warning was read straight out of a scripted capture (see
[[frame-capacity-overflow]]). Two things still hold: it is off by default so the
normal capture is unchanged, and a log line is still the better home for anything
that needs to be machine-*checked* rather than looked at, because grepping a
number beats reading it off an image. Use the flag when the question is about the
UI itself — a colour, a layout, whether a line appears at all.

The trap it hides: after the overlay the swapchain image is past the graph's
present transition, so the copy has to restore PRESENT_SRC rather than the
colour-attachment layout the pre-overlay copy uses.
