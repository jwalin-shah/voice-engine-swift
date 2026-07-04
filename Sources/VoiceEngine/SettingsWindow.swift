import AppKit
import SwiftUI

@MainActor
public final class SettingsWindow {
    private var window: NSWindow?
    public init() {}
    public func show() {
        if let w = window { w.makeKeyAndOrderFront(nil); NSApp.activate(ignoringOtherApps: true); return }
        let hostingView = NSHostingView(rootView: SettingsView(onClose: { [weak self] in self?.close() }))
        let w = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 500, height: 420),
            styleMask: [.titled, .closable, .miniaturizable], backing: .buffered, defer: false)
        w.title = "VoiceEngine Settings"
        w.contentView = hostingView
        w.center()
        w.isReleasedWhenClosed = false
        self.window = w
        w.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }
    public func close() { window?.close() }
}

struct SettingsView: View {
    let onClose: () -> Void
    @State private var selectedTab = 0
    @StateObject private var vocabModel = VocabViewModel()
    @State private var showingAddVocab = false
    @AppStorage("cleanupMode") private var cleanupMode: String = CleanupService.CleanupMode.full.rawValue

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 0) {
                TabButton(title: "Vocabulary", index: 0, selected: $selectedTab)
                TabButton(title: "App Commands", index: 1, selected: $selectedTab)
                TabButton(title: "Cleanup", index: 2, selected: $selectedTab)
                TabButton(title: "About", index: 3, selected: $selectedTab)
            }.padding(.top, 8)
            Divider()
            switch selectedTab {
            case 0: vocabView
            case 1: appCommandsView
            case 2: cleanupView
            case 3: aboutView
            default: EmptyView()
            }
        }.frame(minWidth: 480, minHeight: 380)
    }

    private var vocabView: some View {
        VStack(spacing: 12) {
            HStack {
                Text("Custom Vocabulary").font(.headline)
                Spacer()
                Button("+ Add") { showingAddVocab = true }.buttonStyle(.borderedProminent)
            }.padding(.horizontal).padding(.top, 12)
            if vocabModel.entries.isEmpty {
                VStack(spacing: 6) {
                    Text("No custom vocabulary yet").foregroundColor(.secondary)
                    Text("Add words the model frequently gets wrong")
                        .font(.caption).foregroundColor(.secondary)
                }.frame(maxHeight: .infinity)
            } else {
                List { ForEach(vocabModel.entries.indices, id: \.self) { i in
                    VocabRow(entry: vocabModel.entries[i],
                        onToggle: { vocabModel.entries[i].isActive.toggle(); vocabModel.save() },
                        onDelete: { vocabModel.entries.remove(at: i); vocabModel.save() })
                } }
            }
            if showingAddVocab { AddVocabSheet(model: vocabModel, dismiss: { showingAddVocab = false }) }
        }
    }

    private var appCommandsView: some View {
        VStack(spacing: 12) {
            HStack {
                Text("App-Specific Commands").font(.headline)
                Spacer()
                Text("Coming soon commands will work per-app")
                    .font(.caption).foregroundColor(.secondary)
            }.padding(.horizontal).padding(.top, 12)
            Spacer()
        }
    }

    private var cleanupView: some View {
        VStack(spacing: 16) {
            Text("Text Cleanup").font(.headline)
            Text("Lightweight filler-word removal, pure Swift, no external process.")
                .font(.subheadline).foregroundColor(.secondary)
            Picker("Cleanup mode", selection: $cleanupMode) {
                Text("Disabled").tag(CleanupService.CleanupMode.disabled.rawValue)
                Text("Filler only").tag(CleanupService.CleanupMode.fillerOnly.rawValue)
                Text("Full cleanup").tag(CleanupService.CleanupMode.full.rawValue)
            }.pickerStyle(.radioGroup).padding(.horizontal)
            Text("Filler only removes um, uh, like, you know, etc.")
                .font(.caption).foregroundColor(.secondary)
            Text("Full cleanup currently performs the same filler removal.")
                .font(.caption).foregroundColor(.secondary)
            Spacer()
        }.padding()
    }

    private var aboutView: some View {
        VStack(spacing: 16) {
            Image(systemName: "mic.fill").font(.system(size: 48)).foregroundColor(.accentColor)
            Text("VoiceEngine").font(.title).bold()
            Text("Moonshine CoreML ONNX ANE").font(.caption).foregroundColor(.secondary)
            Text("Model: UsefulSensors/moonshine-tiny").font(.caption).foregroundColor(.secondary)
            Text("~0.004 RTF on Apple Silicon").font(.caption).foregroundColor(.secondary)
            Spacer()
            Button("Close") { onClose() }.keyboardShortcut(.escape)
        }.padding()
    }
}

struct TabButton: View {
    let title: String
    let index: Int
    @Binding var selected: Int
    var body: some View {
        Button(action: { selected = index }) {
            Text(title).font(.subheadline)
                .padding(.horizontal, 16).padding(.vertical, 6)
                .background(selected == index ? Color.accentColor.opacity(0.15) : Color.clear)
                .cornerRadius(6)
        }.buttonStyle(.plain)
    }
}

struct VocabRow: View {
    var entry: VocabularyService.VocabEntry
    var onToggle: () -> Void
    var onDelete: () -> Void
    var body: some View {
        HStack {
            Toggle("", isOn: Binding(get: { entry.isActive }, set: { _ in onToggle() })).toggleStyle(.switch)
            Text("\"\(entry.trigger)\"").fontWeight(.medium)
            Image(systemName: "arrow.right").foregroundColor(.secondary)
            Text("\"\(entry.replacement)\"").foregroundColor(.accentColor)
            Spacer()
            Button(action: onDelete) { Image(systemName: "trash").foregroundColor(.red) }.buttonStyle(.plain)
        }.padding(.vertical, 2)
    }
}

struct AddVocabSheet: View {
    @ObservedObject var model: VocabViewModel
    let dismiss: () -> Void
    @State private var trigger = ""
    @State private var replacement = ""
    var body: some View {
        VStack(spacing: 8) {
            HStack {
                TextField("Word model hears...", text: $trigger).textFieldStyle(.roundedBorder)
                Image(systemName: "arrow.right").foregroundColor(.secondary)
                TextField("Replacement...", text: $replacement).textFieldStyle(.roundedBorder)
                Button("Add") {
                    guard !trigger.isEmpty, !replacement.isEmpty else { return }
                    model.entries.append(VocabularyService.VocabEntry(trigger: trigger, replacement: replacement))
                    model.save()
                    dismiss()
                }.buttonStyle(.borderedProminent)
                Button("Cancel") { dismiss() }.buttonStyle(.plain)
            }
        }.padding().background(Color(NSColor.controlBackgroundColor)).cornerRadius(8).padding(.horizontal)
    }
}

@MainActor
class VocabViewModel: ObservableObject {
    @Published var entries: [VocabularyService.VocabEntry] = []
    init() { load() }
    func load() { entries = VocabularyService.shared.vocabulary }
    func save() { VocabularyService.shared.vocabulary = entries }
}
