// voice-typer.c — Tiny CGEvent text injector.
// Compiled once, signed ad-hoc, never rebuilt.
// This is the ONLY binary that needs Accessibility permission.
// VoiceEngine calls it via subprocess — no Swift runtime needed.
//
// Build:
//   cc -o ~/local/bin/voice-typer voice-typer.c -framework ApplicationServices -framework Carbon
//   codesign --force --sign - voice-typer
//
// Usage:
//   voice-typer "text to type"
//   cat output.txt | voice-typer

#include <ApplicationServices/ApplicationServices.h>
#include <Carbon/Carbon.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static void type_string(const char *str) {
    CGEventSourceRef src = CGEventSourceCreate(kCGEventSourceStatePrivate);
    // Create UCKeyTranslate keymap for proper character mapping
    TISInputSourceRef is = TISCopyCurrentKeyboardInputSource();
    CFDataRef layout = TISGetInputSourceProperty(is, kTISPropertyUnicodeKeyLayoutData);
    const UCKeyboardLayout *keyboardLayout = (const UCKeyboardLayout *)CFDataGetBytePtr(layout);

    for (const char *p = str; *p; p++) {
        UniChar c = *p;
        // Handle uppercase via shift
        bool shift = (c >= 'A' && c <= 'Z');
        if (shift) c = c - 'A' + 'a';

        // Map to virtual key code
        UInt32 deadKeyState = 0;
        UInt16 keys[4];
        UniCharCount len;
        UCKeyTranslate(keyboardLayout, c, kUCKeyActionDown,
                       shift ? shiftKey : 0, LMGetKbdType(), 0,
                       &deadKeyState, 4, &len, keys);

        if (len > 0) {
            CGEventRef down = CGEventCreateKeyboardEvent(src, keys[0], true);
            if (shift) CGEventSetFlags(down, kCGEventFlagMaskShift);
            CGEventPost(kCGHIDEventTap, down);
            CFRelease(down);

            CGEventRef up = CGEventCreateKeyboardEvent(src, keys[0], false);
            if (shift) CGEventSetFlags(up, kCGEventFlagMaskShift);
            CGEventPost(kCGHIDEventTap, up);
            CFRelease(up);
        }
    }
    CFRelease(is);
    CFRelease(src);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        // Read from stdin
        char buf[65536];
        ssize_t n = read(0, buf, sizeof(buf) - 1);
        if (n > 0) {
            buf[n] = 0;
            type_string(buf);
        }
    } else {
        type_string(argv[1]);
    }
    return 0;
}
