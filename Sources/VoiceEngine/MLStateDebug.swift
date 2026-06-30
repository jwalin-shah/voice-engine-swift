import CoreML
import Foundation

/// Utility to inspect MLState methods at runtime.
/// Used to find the correct selector for setting state values.
public enum MLStateDebug {
    public static func logMethods() {
        guard let stateClass = NSClassFromString("MLState") as? NSObject.Type else {
            NSLog("[MLStateDebug] MLState class not found")
            return
        }

        var methodCount: UInt32 = 0
        guard let methods = class_copyMethodList(stateClass, &methodCount) else {
            NSLog("[MLStateDebug] no methods found")
            return
        }
        defer { free(methods) }

        for i in 0..<Int(methodCount) {
            let sel = method_getName(methods[i])
            let name = sel_getName(sel)
            let type = method_getTypeEncoding(methods[i])
            NSLog("[MLStateDebug] method: %s, type: %s", name, type ?? "?")
        }
    }

    /// Try to call write_state via Obj-C runtime
    @discardableResult
    public static func writeState(_ state: MLState, key: String, value: MLMultiArray) -> Bool {
        let sel = Selector(("setStateValue:forKey:"))
        guard state.responds(to: sel) else {
            NSLog("[MLStateDebug] setStateValue:forKey: not found, trying alternatives...")
            // Try common alternatives
            let alternatives = [
                "setStateValue:forKey:",
                "writeState:forKey:",
                "_setStateValue:forKey:",
                "setValue:forStateKey:",
            ]
            for alt in alternatives {
                let altSel = Selector(alt)
                if state.responds(to: altSel) {
                    NSLog("[MLStateDebug] found: %@", alt)
                    return true
                }
            }
            return false
        }
        state.perform(sel, with: MLFeatureValue(multiArray: value), with: key)
        return true
    }
}
