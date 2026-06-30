import AppKit

/// Floating HUD showing transcribed text for 1.5 s, then fades out.
@MainActor
public final class HUDWindow: @unchecked Sendable {
    private var window: NSWindow?
    private var label: NSTextField?
    private var fadeTask: Task<Void, Never>?

    public init() {}

    public nonisolated func show(_ text: String) {
        Task { @MainActor in showOnMain(text) }
    }

    public nonisolated func showPersistent(_ text: String) {
        Task { @MainActor in showPersistentOnMain(text) }
    }

    public nonisolated func hide() {
        Task { @MainActor in
            fadeTask?.cancel()
            window?.orderOut(nil)
        }
    }

    private func ensureWindow() {
        guard window == nil else { return }
        let w = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 600, height: 80),
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )
        w.isOpaque = false
        w.backgroundColor = .clear
        w.level = .floating
        w.collectionBehavior = [.canJoinAllSpaces, .stationary]
        w.ignoresMouseEvents = true
        w.hasShadow = true

        let box = NSBox(frame: w.contentView!.bounds)
        box.boxType = .custom
        box.cornerRadius = 14
        box.fillColor = NSColor.black.withAlphaComponent(0.75)
        box.borderWidth = 0
        box.autoresizingMask = [.width, .height]

        let tf = NSTextField(frame: box.bounds.insetBy(dx: 20, dy: 10))
        tf.isEditable = false
        tf.isBordered = false
        tf.drawsBackground = false
        tf.textColor = .white
        tf.font = .systemFont(ofSize: 20, weight: .medium)
        tf.alignment = .center
        tf.lineBreakMode = .byWordWrapping
        tf.autoresizingMask = [.width, .height]

        box.addSubview(tf)
        w.contentView?.addSubview(box)
        label = tf
        window = w
    }

    private func positionWindow() {
        guard let window, let screen = NSScreen.main else { return }
        let screenFrame = screen.visibleFrame
        let winFrame = window.frame
        let x = screenFrame.midX - winFrame.width / 2
        let y = screenFrame.minY + 120
        window.setFrameOrigin(NSPoint(x: x, y: y))
    }

    /// Show text that fades after 1.5s (for final transcription).
    private func showOnMain(_ text: String) {
        fadeTask?.cancel()
        ensureWindow()
        label?.stringValue = text
        positionWindow()
        window?.alphaValue = 0.92
        window?.orderFrontRegardless()

        fadeTask = Task { @MainActor [weak self] in
            try? await Task.sleep(for: .seconds(1.5))
            guard let self, let w = self.window else { return }
            NSAnimationContext.runAnimationGroup { ctx in
                ctx.duration = 0.3
                w.animator().alphaValue = 0
            } completionHandler: {
                w.orderOut(nil)
            }
        }
    }

    /// Show text that stays until replaced or hidden (for streaming).
    private func showPersistentOnMain(_ text: String) {
        fadeTask?.cancel()
        ensureWindow()
        label?.stringValue = text
        positionWindow()
        window?.alphaValue = 0.92
        window?.orderFrontRegardless()
        // No fade — stays visible until next showPersistent or hide()
    }
}