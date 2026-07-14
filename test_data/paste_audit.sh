#!/bin/bash
# Paste reliability audit — CGEvent Cmd+V across target app matrix
# Ticket #15: HITL test harness
#
# Usage: ./test_data/paste_audit.sh
#
# For each app, this script:
#   1. Prompts you to focus the app
#   2. Runs `swift run voice --paste <test_string>`
#   3. Asks you to verify the text appeared
#
# The test string includes a UUID to verify each paste is fresh (not a stale clipboard).
# Press Enter to advance. Type 'skip' to skip an app. Type 'quit' to exit.

set -euo pipefail

BOLD="\033[1m"
GREEN="\033[32m"
RED="\033[31m"
YELLOW="\033[33m"
RESET="\033[0m"

RESULTS_FILE="$(dirname "$0")/paste_audit_results.txt"
PASS=0
FAIL=0
SKIP=0

# Ensure clean results file
echo "# Paste Audit Results — $(date)" > "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# Build once
echo -e "${BOLD}Building voice-engine...${RESET}"
(cd "$(dirname "$0")/.." && swift build 2>&1 | tail -1)

test_one() {
    local category="$1"
    local app="$2"
    local notes="${3:-}"
    local uuid
    uuid="$(uuidgen | tr '[:upper:]' '[:lower:]')"
    local test_text="paste-audit-${uuid}"

    echo ""
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
    echo -e "${BOLD}[${category}] ${app}${RESET}"
    if [ -n "$notes" ]; then
        echo -e "  ${YELLOW}Note: ${notes}${RESET}"
    fi
    echo ""
    echo -e "  ${YELLOW}1. Focus/click into ${app}${RESET}"
    echo -e "  ${YELLOW}2. Place cursor where text should appear${RESET}"
    echo -n "  Press Enter when ready (skip/quit): "
    read -r answer
    case "$answer" in
        skip) echo "  Skipped."; echo "$category | $app | SKIP | —" >> "$RESULTS_FILE"; SKIP=$((SKIP + 1)); return ;;
        quit) echo "  Quitting."; exit 0 ;;
    esac

    # Run paste
    (cd "$(dirname "$0")/.." && swift run voice-engine --paste "$test_text" 2>/dev/null)

    echo ""
    echo -n "  Did \"${test_text}\" appear? (y/n/skip/quit): "
    read -r answer
    case "$answer" in
        y|Y)
            echo -e "  ${GREEN}PASS${RESET}"
            echo "$category | $app | PASS | $test_text" >> "$RESULTS_FILE"
            PASS=$((PASS + 1))
            ;;
        n|N)
            echo -e "  ${RED}FAIL${RESET}"
            echo "$category | $app | FAIL | $test_text" >> "$RESULTS_FILE"
            FAIL=$((FAIL + 1))
            ;;
        skip)
            echo "  Skipped."
            echo "$category | $app | SKIP | —" >> "$RESULTS_FILE"
            SKIP=$((SKIP + 1))
            ;;
        quit) echo "  Quitting."; exit 0 ;;
        *) echo "  Unrecognized, marking as FAIL."; echo "$category | $app | FAIL | $test_text" >> "$RESULTS_FILE"; FAIL=$((FAIL + 1)) ;;
    esac
}

echo ""
echo -e "${BOLD}Paste Reliability Audit — Ticket #15${RESET}"
echo "Each test: sets clipboard → posts CGEvent Cmd+V → you verify text landed."
echo "Each paste contains a unique UUID — stale clipboard is automatically detected."
echo ""

# ── Native text ──
test_one "Native text" "TextEdit" "New document, cursor in body"
test_one "Native text" "Notes" "New note, cursor in body"
test_one "Native text" "Pages" "New document, cursor in body"
test_one "Native text" "Xcode" "Open any .swift file, cursor in editor"

# ── Browsers ──
test_one "Browser" "Safari" "Any text field (address bar or web form)"
test_one "Browser" "Chrome" "Any text field"
test_one "Browser" "Firefox" "Any text field"
test_one "Browser" "Arc" "Any text field"

# ── Electron ──
test_one "Electron" "VS Code" "Open any file, cursor in editor"
test_one "Electron" "Slack" "Message input field"
test_one "Electron" "Discord" "Message input field"
test_one "Electron" "Cursor" "Open any file, cursor in editor"

# ── Terminals ──
test_one "Terminal" "Terminal.app" "At shell prompt (may need to be in insert mode)"
test_one "Terminal" "iTerm2" "At shell prompt"
test_one "Terminal" "Warp" "At shell prompt"
test_one "Terminal" "Ghostty" "At shell prompt"

# ── System ──
test_one "System" "Spotlight" "Open Spotlight (Cmd+Space), paste should fill search field"
test_one "System" "Finder rename" "Select a file, press Enter to rename, cursor in name field"
test_one "System" "Password field" "Any password/secure text field (e.g. System Settings login, 1Password unlock)"

# ── Full-screen ──
test_one "Full-screen" "Any app (full-screen)" "Put any app from above into full-screen mode and re-test"

# ── Non-QWERTY (only if you have the layout available) ──
test_one "Layout" "Dvorak/Colemak" "Switch to Dvorak or Colemak layout if available; test in TextEdit"

# ── Summary ──
echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${BOLD}Results: ${GREEN}${PASS} passed${RESET}, ${RED}${FAIL} failed${RESET}, ${YELLOW}${SKIP} skipped${RESET}"
echo ""
echo "Results saved to: $RESULTS_FILE"
echo ""
if [ "$FAIL" -gt 0 ]; then
    echo -e "${RED}Failures:${RESET}"
    grep "FAIL" "$RESULTS_FILE"
fi
