---
name: instrument-before-guessing-runtime-bugs
description: "For runtime/interaction bugs I can't reproduce in-sandbox, add on-screen/console diagnostics early instead of iterating blind fixes"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 41ac6ee5-9d22-4dc4-b6c3-87fe38f5087d
---

When a runtime or input/interaction bug can only be observed on the user's machine
(GPU app, OS-specific behavior), add instrumentation that surfaces the actual state
BEFORE proposing fixes — don't ship a sequence of plausible-but-blind guesses.

**Why:** On the VulkanEngine "gizmo stuck/delayed on mouse release" bug I shipped four
guesses across four round-trips (windowID rewrite → SDL_GetMouseState reconcile →
SDL_GetGlobalMouseState + force-release → ImGui capture-disable), each requiring a
rebuild + user test, and each was wrong or partial. A small temporary on-screen readout
(ImGuizmo::IsUsing, io.MouseDown[0]+held time, SDL_GetGlobalMouseState, WantCaptureMouse)
pinpointed the true cause in ONE round: all signals — including the OS-level one — stayed
"down" until the cursor moved, which is SDL auto-capture deferring the button-up on macOS
(SDL_HINT_MOUSE_AUTO_CAPTURE). The user's terse repeated "还是有延迟" was the cost signal.

**How to apply:** Since I [[vulkanengine-cannot-run-in-sandbox]], for any bug I can't
reproduce headlessly, after one quick hypothesis-test, switch to adding a cheap diagnostic
(debug-UI readout or stderr log of the suspect state each frame) and ask the user for that
one observation. One instrumented round beats N blind rebuild/test cycles. Strip the
diagnostics once the cause is found.
