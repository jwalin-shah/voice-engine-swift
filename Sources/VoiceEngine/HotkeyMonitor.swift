import Carbon.HIToolbox
import CoreGraphics
import Foundation

/// Caps Lock (keycode 57) toggles recording.
/// Uses time-based debounce so a single press toggles once.
/// If Accessibility is denied, grant it in System Settings > Privacy & Security > Accessibility.
public final class HotkeyMonitor {
    public typealias Handler = () -> Void

    private var tap: CFMachPort?
    private let box: ContextBox

    public init(handler: @escaping Handler) {
        self.box = ContextBox(handler: handler)
    }

    deinit { stop() }

    public func start() throws {
        let mask: CGEventMask = (1 << CGEventType.flagsChanged.rawValue)
        let ctx = Unmanaged.passRetained(box).toOpaque()
        let callback: CGEventTapCallBack = { _, type, event, refcon in
            guard type == .flagsChanged else {
                return Unmanaged.passUnretained(event)
            }
            guard let refcon else { return Unmanaged.passUnretained(event) }
            let box = Unmanaged<ContextBox>.fromOpaque(refcon).takeUnretainedValue()
            // Caps Lock keycode = 57.
            let keycode = event.getIntegerValueField(.keyboardEventKeycode)
            let flags = event.flags.rawValue
            Foundation.NSLog("[HotkeyMonitor] flagsChanged keycode=%ld flags=0x%lx", keycode, flags)
            guard keycode == 57 else { return Unmanaged.passUnretained(event) }

            let now = CFAbsoluteTimeGetCurrent()
            let debounce: TimeInterval = 0.25
            if now - box.lastFireTime > debounce {
                box.lastFireTime = now
                Foundation.NSLog("[HotkeyMonitor] Caps Lock TRIGGER")
                DispatchQueue.main.async(execute: box.handler)
            } else {
                Foundation.NSLog("[HotkeyMonitor] Caps Lock debounced")
            }
            return nil
        }
        guard let tap = CGEvent.tapCreate(
            tap: .cgSessionEventTap,
            place: .headInsertEventTap,
            options: .defaultTap,
            eventsOfInterest: mask,
            callback: callback,
            userInfo: ctx
        ) else {
            throw NSError(domain: "HotkeyMonitor", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "CGEvent tap failed — grant Accessibility in System Settings > Privacy & Security > Accessibility"
            ])
        }
        self.tap = tap
        let source = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, tap, 0)
        CFRunLoopAddSource(CFRunLoopGetMain(), source, .commonModes)
        CGEvent.tapEnable(tap: tap, enable: true)
    }

    public func stop() {
        guard let tap else { return }
        CGEvent.tapEnable(tap: tap, enable: false)
        CFMachPortInvalidate(tap)
        self.tap = nil
    }
}

private final class ContextBox {
    var lastFireTime: TimeInterval = 0
    let handler: HotkeyMonitor.Handler
    init(handler: @escaping HotkeyMonitor.Handler) {
        self.handler = handler
    }
}
