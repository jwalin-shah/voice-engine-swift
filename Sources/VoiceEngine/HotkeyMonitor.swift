import Carbon.HIToolbox
import CoreGraphics
import Foundation

/// Push-to-talk via CGEvent tap.
/// If Accessibility is denied, grant it in System Settings > Privacy & Security > Accessibility.
public final class HotkeyMonitor {
    public typealias Handler = () -> Void

    private var tap: CFMachPort?
    private let box: ContextBox
    private var capsLockWasOn = false

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
            if event.getIntegerValueField(.keyboardEventKeycode) == 57 {
                let flags = event.flags.rawValue
                let isOn = (flags & 65536) != 0
                // Fire once per press: when state transitions (ON->OFF or OFF->ON).
                if isOn != box.capsLockWasOn {
                    DispatchQueue.main.async(execute: box.handler)
                }
                box.capsLockWasOn = isOn
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
    var capsLockWasOn = false
    let handler: HotkeyMonitor.Handler
    init(handler: @escaping HotkeyMonitor.Handler) {
        self.handler = handler
    }
}
