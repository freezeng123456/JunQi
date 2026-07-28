import AppKit
import CoreGraphics
import Foundation
import ImageIO

let root = "/Users/zenghang/Documents/Codex/2026-07-24/https-git-woa-com-freezeng-junqi-2/JunQi"
let sourceURL = URL(fileURLWithPath: root).appendingPathComponent("legacy_gui/res/orange.bmp")
let sampleURL = URL(fileURLWithPath: "/Users/zenghang/Documents/Codex/2026-07-24/https-git-woa-com-freezeng-junqi-2/outputs/orange_siling_text_hardened_sample.png")
let comparisonURL = URL(fileURLWithPath: "/Users/zenghang/Documents/Codex/2026-07-24/https-git-woa-com-freezeng-junqi-2/outputs/orange_siling_text_hardened_comparison.png")

func load(_ url: URL) -> CGImage {
    let source = CGImageSourceCreateWithURL(url as CFURL, nil)!
    return CGImageSourceCreateImageAtIndex(source, 0, nil)!
}

func makeRGBAContext(width: Int, height: Int) -> CGContext {
    CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
              bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
}

func normalize(_ tile: CGImage) -> (CGImage, [UInt8]) {
    let c = makeRGBAContext(width: 36, height: 27)
    c.interpolationQuality = .none
    c.draw(tile, in: CGRect(x: 0, y: 0, width: 36, height: 27))
    return (c.makeImage()!, Array(UnsafeBufferPointer(start: c.data!.assumingMemoryBound(to: UInt8.self), count: 36 * 27 * 4)))
}

func makeImage(_ bytes: [UInt8]) -> CGImage {
    let data = Data(bytes)
    let provider = CGDataProvider(data: data as CFData)!
    return CGImage(width: 36, height: 27, bitsPerComponent: 8, bitsPerPixel: 32,
                   bytesPerRow: 36 * 4, space: CGColorSpaceCreateDeviceRGB(),
                   bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                   provider: provider, decode: nil, shouldInterpolate: false,
                   intent: .defaultIntent)!
}

func save(_ image: CGImage, to url: URL) {
    let d = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil)!
    CGImageDestinationAddImage(d, image, nil)
    precondition(CGImageDestinationFinalize(d))
}

let sheet = load(sourceURL)
let tile = sheet.cropping(to: CGRect(x: 360, y: 0, width: 36, height: 27))!
let (base, original) = normalize(tile)
var hardened = original
// Preserve the existing glyph geometry and color. Only turn the semi-transparent
// white antialias pixels inside the text band into solid white.
for y in 5..<24 {
    for x in 3..<33 {
        let p = (y * 36 + x) * 4
        let r = Int(original[p]), g = Int(original[p + 1]), b = Int(original[p + 2])
        if b > 40 && r > 165 && g > 130 && r - g < 130 {
            hardened[p] = 255; hardened[p + 1] = 255; hardened[p + 2] = 255; hardened[p + 3] = 255
        }
    }
}
let sample = makeImage(hardened)
save(sample, to: sampleURL)

let c = makeRGBAContext(width: 780, height: 340)
c.setFillColor(NSColor(calibratedWhite: 0.13, alpha: 1).cgColor)
c.fill(CGRect(x: 0, y: 0, width: 780, height: 340))
c.interpolationQuality = .high
c.draw(base, in: CGRect(x: 20, y: 45, width: 360, height: 270))
c.draw(sample, in: CGRect(x: 400, y: 45, width: 360, height: 270))
let font = NSFont(name: "Helvetica", size: 22)!
for (label, x) in [("当前", 200.0), ("文字锐化", 580.0)] {
    let attrs: [NSAttributedString.Key: Any] = [.font: font, .foregroundColor: NSColor.white]
    let line = NSAttributedString(string: label, attributes: attrs)
    let size = line.size()
    line.draw(at: CGPoint(x: x - size.width / 2.0, y: 10))
}
// The labels above are only for the preview; the chess tile itself is untouched
// except for solidifying the existing white glyph antialias pixels.
save(c.makeImage()!, to: comparisonURL)
