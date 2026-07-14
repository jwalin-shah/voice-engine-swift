# WisprFlow Full UX Teardown

*Research for voice-engine-swift wayfinder #10. Compiled 2026-07-13 from web search, vendor docs, forensic analyses, and community sources.*

**Tagging convention:** `[Marketing]` = vendor claim, unverified. `[Doc]` = vendor documentation. `[Independent]` = third-party analysis or verified claim. `[Unknown]` = no public data exists.

---

## 1. Perceived Latency Profile

### Vendor claims

- **[Marketing]** "4x faster than typing" — quantified as 220 wpm (Flow) vs 45 wpm (keyboard). This is **throughput**, not stop-recording-to-first-text latency.
- **[Independent]** Baseten case study: **<700ms p99 end-to-end** — from audio ingestion through ASR + Llama-based transcript cleanup to final text. This is the best available hard number. (Source: https://www.baseten.co/customers/wispr-flow/)
- **[Independent]** Llama post-processing alone: **100+ tokens in <250ms p50**. The CTO states they optimize for p99, not p50: *"We measure latency on a p90 or p99 basis for each user; we don't care at all about p50."*
- **[Independent]** Single logged measurement: `basetenCommitAckTimeMsecs: 96` for the Baseten inference acknowledgement.

### What's missing

- **No stop-recording-to-first-text number.** The 700ms p99 covers the full pipeline but the recording boundary isn't clear — is it from end-of-speech (VAD-triggered) or from manual stop? Unclear.
- **No streaming-partial vs final-commit split.** Wispr markets "AI Auto Edits" that rewrite while you talk (implying streaming partials), but the Llama cleanup is a post-process. The split between first-partial-appears and final-text-committed latency is unpublished.
- **Cloud round-trip floor is unquantified.** Audio → Baseten (gRPC) → fine-tuned Llama → back. Network jitter is a latency floor on-device solutions don't have. No benchmark quantifies it.
- **[Independent]** Competitor Willow Voice claims Wispr "creates a slight delay that breaks your rhythm" — unverified counter-claim, useful only as evidence some users perceive lag.

### What this means for voice-engine

WisprFlow's latency is **not magic** — it's ~700ms p99 for a cloud pipeline. That's a beatable target for a local CoreML Moonshine-tiny pipeline *if* our model's encode+decode fits within that window. The 700ms number sets a concrete upper bound: our post-stop E2E should land under 700ms to claim "matches WisprFlow." But we need a working bench (#12) to measure against this.

---

## 2. Model + Architecture

### Architecture (confirmed cloud-only)

**[Doc]** WisprFlow's official Data Controls page: *"Transcription always occurs on the cloud. This is the best way for us to provide accurate, low latency transcription."* No offline mode at any price tier.

**Pipeline (confirmed via subprocessor docs + forensic analysis):**

| Stage | Technology | Where |
|---|---|---|
| Audio capture | OPUS-encoded chunks (`opusChunks` JSON field) | Local |
| Screen context | Accessibility tree traversal (up to 214 elements, 9 levels deep) | Local |
| Speech recognition | Cloud ASR on **Baseten** (gRPC endpoint: `model-v31pl413.grpc.api.baseten.co`) | Cloud |
| Transcript cleanup | **Fine-tuned Llama models** on TensorRT-LLM | Cloud (Baseten) |
| Secondary AI | **OpenAI, Anthropic, Cerebras** for Command Mode/Polish | Cloud |
| Storage | **AWS S3 us-east-1** | Cloud |

**[Doc]** 11+ subprocessors. The official FAQ claims *"Is this just a wrapper around Whisper? No! … we actually predate OpenAI's Whisper"* — but the Baseten subprocessor strongly implies a Whisper-family ASR component.

### Model details

- **[Unknown]** ASR model size, parameter count, and architecture are **not disclosed**. Baseten offers Whisper models — plausible but unconfirmed.
- **[Independent]** Llama model for cleanup: fine-tuned, runs on TensorRT-LLM. **100+ tokens in <250ms p50.**
- **[Unknown]** Streaming incremental decode vs batch decode is undocumented. The gRPC streaming architecture implies server-side streaming, but the client-side protocol is opaque.
- **[Independent]** Ensemble model investigated but **not enabled** (`use-ensemble-model: false` in forensic analysis).
- **[Independent]** No CoreML. No ANE usage. Everything runs on cloud GPUs.

### What this means for voice-engine

WisprFlow's speed advantage comes from **cloud GPU inference**, not a fundamentally faster model architecture. The ASR model (likely Whisper-family on Baseten GPUs) + Llama cleanup runs on server-class hardware. Moonshine-tiny on ANE/CPU is a different performance envelope — we need to measure whether it fits in the ~700ms window.

**Streaming is a UX trick, not the latency driver.** WisprFlow streams partials to create the *perception* of instant response, but the final committed text still goes through a ~700ms pipeline. Since we can't stream partials (Moonshine-tiny architecture), we need our total post-stop latency to be low enough that the user doesn't *miss* the streaming trick. Whether that means <700ms or <500ms or <300ms is a UX question the bench will help answer.

**Key insight:** WisprFlow's latency is *not* primarily from streaming — it's from fast cloud inference. The streaming just masks the wait. Our challenge is making the total wait short enough that masking isn't needed.

---

## 3. Paste / Injection Mechanism

### Primary mechanism: Accessibility (AX) insertion

**[Doc]** WisprFlow requires **Accessibility permission** on macOS (System Settings → Privacy & Security → Accessibility). Text is injected via AX API.

**Fallback path:** When AX insertion fails, it falls back to **clipboard + simulated paste**: `Cmd+Ctrl+V` (Mac) or app-menu "Copy last transcript."

### Keyboard interception (CGEventTap)

**[Independent]** Forensic analysis (Wensen Wu, April 2026) revealed WisprFlow installs a **CGEventTap** that intercepts **every keystroke system-wide** — not just during dictation. Key findings:
- Two-process architecture: Swift helper (`swift-helper-app-dist`) runs the CGEventTap + keyboard state; Electron main process handles UI/cloud
- 12 dispatch queues including `keyEventQueue`, `sendQueue`, `runQueue`
- Events are buffered in `_keyEventBuffer` and processed asynchronously
- **16 buffer flushes in ~30 hours** — keyboard stalls roughly every 2 hours
- **145 spacebar presses suppressed in 10 minutes** due to a persistent race condition
- Stale key recovery mechanism exists but has blind spots
- Source: https://wensenwu.com/thoughts/wispr-flow-investigation

### App compatibility

**[Doc]** Terminal support:
- **Works:** Terminal.app, iTerm2, Warp, Ghostty, Hyper, Alacritty, Kitty, WezTerm
- **Does NOT work (requires clipboard fallback):** WSL, SSH, tmux, screen, Termius
- Windows uses `Shift+Insert` instead of `Ctrl+V` in IDE terminals

**[Doc]** Known issues:
- Fails to detect text fields or inserts incorrectly on **non-QWERTY keyboard layouts**
- **Missing first words** in transcriptions
- "Connection lost" on VPNs
- iOS missing audio chunks on unstable internet
- **[Independent]** Competitor reports Wispr "freezes target apps like VS Code" on Windows — unverified

**[Doc]** Password fields: Standard macOS password fields are excluded from Context Awareness reading, but **custom/web-based password fields "may be read like normal text fields."** Audio is still transcribed into them normally.

### What this means for voice-engine

WisprFlow's injection is **invasive and buggy**. They need a CGEventTap + AX combo because they're a cloud service trying to act like a local IME. We're local — we can use the same three-tier fallback in `Paster.swift` (AX insert → osascript/CGEvent paste → Cmd+V clipboard) without the keyboard interception overhead.

Their reliability issues (missing first words, non-QWERTY breakage, buffer stalls) are **availability problems caused by complexity** — the CGEventTap + cloud round-trip + async buffer architecture creates race conditions. Our local pipeline avoids the worst of these.

**The clipboard-fallback pattern is worth adopting explicitly.** WisprFlow falls back when AX fails; our `Paster.swift` already does this. Their documentation of *which* apps fail AX insertion (terminals via SSH/tmux, non-QWERTY layouts) is a useful test matrix for us.

---

## 4. Accuracy + Command Grammar

### Accuracy claims

- **[Marketing]** "90% zero-edit accuracy" — self-published "Wispr Zero-Edit Rate Benchmark": Flow 90%, OpenAI 71%, ElevenLabs 63%, Siri 52%. **Not WER, not independently reproduced, methodology undisclosed.** (Source: https://wisprflow.ai/why-flow)
- **[Marketing]** Accent/noise handling claimed via proprietary retraining on user feedback. No WER numbers for accented speech.
- **[Independent]** Blog post: *"When transcription quality seems to drop suddenly, it's rarely the model itself. Much more often, it's a change in the audio."* — suggests their pipeline is audio-quality-sensitive despite marketing claims.

### Voice commands

**[Doc]** **Command Mode** (Pro/trial only, Settings → Experimental):
- Highlight text + shortcut (`Fn+Ctrl` Mac), speak a command: "Make this more concise," "Translate to Polish," etc.
- Replaces selection with LLM-processed result

**[Doc]** Built-in literal commands:
- **"press enter"** — strips the phrase, sends Enter keystroke (for chat apps)
- Punctuation is automatic (commas, question marks, periods from pauses/tone)
- **Self-correction handling:** "let's meet at 5… actually 6 pm" → "Let's meet at 6pm." This is their core differentiator vs built-in dictation.
- Undo via `Cmd+Z` (not a voice command). No dedicated "select/delete word" verbs beyond Command Mode's free-form LLM rewrite.

**[Doc]** Developer features:
- Syntax awareness: preserves camelCase, snake_case, file names
- File tagging: speak file names, Flow tags them in Cursor/Windsurf
- Code formatting preservation

**[Marketing]** 100+ languages with automatic detection.

### What this means for voice-engine

The "90% zero-edit" benchmark is marketing — ignore it as a target. Their real accuracy advantage comes from the **Llama cleanup layer**, not the ASR model itself. Raw Whisper/Moonshine transcription will always look worse than WisprFlow's polished output because they post-process through an LLM.

**We should NOT chase this.** The cleanup layer is a separate product decision (do we want an LLM post-pass?). For the core dictation engine, the target is raw transcription accuracy — WER against a reference dataset. The cleanup layer can be a later ticket if raw accuracy is competitive.

Their self-correction handling ("actually 6 pm") is a UX feature, not an accuracy feature — it's the LLM understanding conversational corrections. Out of scope for us unless we add an LLM pass.

---

## 5. Competitive Landscape

| | WisprFlow | MacWhisper | Superwhisper | macOS Built-in | **voice-engine (target)** |
|---|---|---|---|---|---|
| Architecture | Cloud-only | On-device | On-device (cloud opt-in) | On-device | On-device |
| Model | Cloud ASR + fine-tuned Llama | CoreML Whisper | CoreML Whisper | Apple proprietary | Moonshine-tiny CoreML |
| Price | $15/mo, $144/yr | ~$39-99 lifetime | $8.49/mo, $250 lifetime | Free | Free (OSS) |
| Offline | No | Yes | Yes | Yes | Yes |
| Polish/auto-edit | Built-in | Add-on | Per-mode, BYOK | None | None |
| Platforms | Mac/Win/iOS/Android | Mac | Mac/Win/iOS | Mac | Mac |
| Latency | ~700ms p99 (cloud) | ~500-1500ms (local) | ~500-1500ms (local) | ~200-500ms | TBD |
| Injection | AX + CGEventTap | AX API | AX API | System IME | AX + osascript + Cmd+V |
| ~800MB RAM idle | Reported (Win) | Lower | Lower | Minimal | Lower |

### WisprFlow's real advantages
- **Zero-config, cross-platform** — genuinely the only multi-OS option
- **Polish/auto-edit out of the box** — the LLM cleanup layer is their strongest feature
- **Team/enterprise features** — SSO, HIPAA BAA, SOC 2, MDM
- **Self-correction handling** — "actually 6 pm" → "6pm" is genuinely good UX

### WisprFlow's real weaknesses
- **No offline** — architectural, never happening
- **Privacy surface** — audio + screen context through 11+ subprocessors
- **Invasive** — CGEventTap intercepts all keystrokes system-wide, even when not dictating
- **Trustpilot 2.7/5** — reliability complaints post-trial
- **Privacy controversy** — 2025 screenshot-upload + user-ban incident; 2026 Delve fake-audit scandal
- **Reliability bugs** — keyboard buffer stalls, suppressed keystrokes, missing first words

---

## 6. Community Sentiment

### Praise
- Accessibility users (Parkinson's, mobility) call it life-changing
- Developer productivity: 170-179 wpm reported by some users
- "Easiest system-wide dictation option" — jamesm.blog review

### Complaints
- **Privacy scandal (2025):** User found Wispr uploading periodic screenshots; the user was banned from their community. CTO later apologized, Context Awareness changed from default-on to opt-in. (Source: https://embertype.com/blog/the-day-wispr-flow-banned-a-user/)
- **Delve audit scandal (2026):** Prior SOC 2/ISO certs via Delve (alleged ~99.8% boilerplate audit reports). Wispr migrated to Drata + A-LIGN, certs under reverification.
- **Trustpilot 2.7/5:** Recurring complaints about reliability degrading after 14-day trial, unpaid referral rewards, concerning ToS.
- **Open-source backlash:** Multiple OSS alternatives emerged explicitly targeting WisprFlow: FreeFlow (277 HN points), Yap, Muesli, SpeechOS, QSpeak, Voquill.

### Key sources
- Forensic investigation: https://wensenwu.com/thoughts/wispr-flow-investigation
- Baseten case study (latency + architecture): https://www.baseten.co/customers/wispr-flow/
- Mac dictation tools comparison: https://jamesm.blog/ai/mac-dictation-tools-comparison/
- WisprFlow vs Superwhisper vs MacWhisper: https://spokenly.app/blog/wispr-flow-vs-superwhisper-vs-macwhisper
- Security/privacy analysis: https://www.getvoibe.com/resources/is-wispr-flow-safe/
- Developer workflow review: https://zackproser.com/blog/wisprflow-high-leverage-workflow

---

## 7. Synthesis: What's Reachable for voice-engine?

### (a) Streaming-only advantages — out of reach
- **Streaming partials during recording.** Moonshine-tiny architecture can't do this. We close the gap by making total post-stop latency short enough that the user doesn't miss partials.
- **LLM-based auto-edit/cleanup.** This is their strongest feature. We can't match it without adding an LLM post-pass — a separate product decision for a later ticket.

### (b) Faster/different model — a real branch to consider
- WisprFlow's ASR is cloud Whisper-family on GPUs. Moonshine-tiny on ANE/CPU is a different performance tier.
- **Decision gate:** Measure our E2E latency first (#12 fixes the bench). If Moonshine-tiny fits under ~700ms, we're competitive. If not, we need to evaluate alternatives (e.g., Whisper-tiny CoreML, or a larger Moonshine variant with better accuracy).
- **The teardown confirms this is worth measuring before committing to Moonshine-tiny as the final model.** Their ~700ms p99 is a concrete target.

### (c) Injection mechanics — directly applicable
- Their AX-primary + clipboard-fallback pattern matches our `Paster.swift` three-tier approach.
- Their documented failure modes (non-QWERTY, SSH/tmux, missing first words) are our test matrix.
- **We avoid their worst problems** (CGEventTap race conditions, buffer stalls) by being a local tool without keyboard interception.
- Their clipboard-fallback + notification pattern ("copied to clipboard, couldn't insert") is worth adopting.

### (d) VAD/segmentation strategy — directly applicable
- WisprFlow streams OPUS chunks continuously; VAD triggers the ASR pipeline. The exact VAD strategy isn't public.
- We already have mic capture + VAD segmentation. Their approach of streaming chunks and letting cloud-side VAD segment them is a cloud luxury — we need client-side VAD, which we already have.
- **Their architecture confirms that VAD-driven segmentation (vs fixed-duration chunks) is the right approach.** We're on the right track.

### Bottom line

**WisprFlow is beatable on latency for raw transcription.** Their ~700ms p99 includes a network round-trip. A local CoreML pipeline should match or beat this. The question is whether Moonshine-tiny specifically is fast enough — measure first (#12).

**WisprFlow is not beatable on output quality without an LLM pass.** Their fine-tuned Llama cleanup layer is their real moat. We should separate "raw transcription accuracy" from "polished output quality" and target the former first.

**WisprFlow's injection reliability is worse than ours can be.** Their CGEventTap architecture creates bugs we don't have. Our simpler AX→osascript→Cmd+V fallback is already more robust — just needs testing across their documented failure matrix.

**The #1 action from this teardown:** Fix the bench (#12), measure Moonshine-tiny's E2E latency, and decide if it fits under the ~700ms target. If yes, proceed with efficiency tuning. If no, evaluate model alternatives before sinking more effort into the Moonshine pipeline.
