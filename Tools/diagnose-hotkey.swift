import CoreGraphics
import Foundation

// Diagnostic: print every flagsChanged event with keycode and flags.
// Run with Accessibility permission granted.

final class Box {
    var lastFire: CFAbsoluteTime = 0
}

let box = Box()
let mask: CGEventMask = (1 << CGEventType.flagsChanged.rawValue)
let ctx = Unmanaged.passRetained(box).toOpaque()

let callback: CGEventTapCallBack = { _, type, event, refcon in
    guard type == .flagsChanged else { return Unmanaged.passUnretained(event) }
    let keycode = event.getIntegerValueField(.keyboardEventKeycode)
    let flags = event.flags.rawValue
    let name: String
    switch keycode {
    case 57: name = "CapsLock"
    case 56: name = "LeftShift"
    case 60: name = "RightShift"
    case 55: name = "LeftCmd"
    case 54: name = "RightCmd"
    case 58: name = "LeftOption"
    case 61: name = "RightOption"
    case 59: name = "LeftCtrl"
    case 62: name = "RightCtrl"
    default: name = "Key\(keycode)"
    }
    print("[\(Date())] \(name) keycode=\(keycode) flags=0x\(String(flags, radix: 16))")
    // Suppress Right Shift so we know the tap is working.
    if keycode == 60 {
        let now = CFAbsoluteTimeGetCurrent()
        if now - box.lastFire > 0.25 {
            box.lastFire = now
            print("  -> Right Shift TRIGGER")
        } else {
            print("  -> Right Shift debounced")
        }
        return nil
    }
    return Unmanaged.passUnretained(event)
}

guard let tap = CGEvent.tapCreate(
    tap: .cgSessionEventTap,
    place: .headInsertEventTap,
    options: .defaultTap,
    eventsOfInterest: mask,
    callback: callback,
    userInfo: ctx
) else {
    print("FAILED to create event tap. Grant Accessibility permission.")
    exit(1)
}

let source = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, tap, 0)
CFRunLoopAddSource(CFRunLoopGetMain(), source, .commonModes)
CGEvent.tapEnable(tap: tap, enable: true)
print("Event tap active. Press Right Shift (and other keys). Ctrl-C to stop.")
CFRunLoopRun()
