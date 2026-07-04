You are writing adversarial/breaking tests for the voice-engine-swift project at ~/projects/voice-engine-swift.

Goal: add tests that would have caught past bugs AND that probe edge cases likely to break in future. 
Add all tests to Tests/Runner/main.swift in the existing TestRunner.runAll() pattern.

Existing test suites: CommandParser.parse, CommandParser.extractCommand, CommandParser.Equatable, VocabularyService, CleanupService, MoonshineEngine.chunkRanges, VAD.isSpeech (just added).

Focus on these adversarial areas:

1. **VAD boundary attacks**
   - Single sample exactly at threshold (floating point equality)
   - windowSize=1 (degenerate), minActiveRatio=0 (always speech), minActiveRatio=1.0 (needs all active)
   - Buffer length exactly = windowSize, exactly = windowSize-1
   - All samples at Float.max (overflow RMS), all at -Float.max, NaN samples
   - Alternating +loud/-loud (RMS should cancel to near-zero)

2. **CommandParser mutation attacks**
   - Unicode homoglyphs: "undо" (Cyrillic о, not Latin o) should NOT parse as .undo
   - Mixed case partial: "UNDO" "Undo" "uNdO" 
   - Embedded commands: "please undo this" (not a pure undo), "can you new line" 
   - extractCommand: what if suffix is ambiguous — "hello select" (no target), "delete that delete that"
   - Very long input (10k chars) with a suffix command — should not hang/crash
   - Command at position 0 vs only-word vs multi-word

3. **MoonshineEngine.chunkRanges edge cases**
   - sampleCount=1, sampleCount=160001, sampleCount=320000 (exactly 2x)
   - Verify no chunk ever exceeds 160000 samples
   - Verify no overlap gap is negative (start of chunk2 < end of chunk1 - 32000)

4. **VocabularyService mutation attacks**  
   - Trigger with regex special chars: "hello.world", "^test$", "(parens)"
   - Trigger that is a substring of another trigger (ensure no partial match)
   - Empty trigger, empty replacement
   - Replacement longer than 1000 chars

After adding tests, run the test binary:
  cd ~/projects/voice-engine-swift && swift build --product voice-tests && .build/debug/voice-tests

Fix any compilation errors. Do not modify production code — only add tests. Report final pass/fail count.
