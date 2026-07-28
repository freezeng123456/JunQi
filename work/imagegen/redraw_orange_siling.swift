import AppKit
import CoreGraphics
import CoreText
import Foundation
import ImageIO

let root = "/Users/zenghang/Documents/Codex/2026-07-24/https-git-woa-com-freezeng-junqi-2/JunQi"
let sourceURL = URL(fileURLWithPath: root).appendingPathComponent("legacy_gui/res/orange.bmp")
let sampleURL = URL(fileURLWithPath: "/Users/zenghang/Documents/Codex/2026-07-24/https-git-woa-com-freezeng-junqi-2/outputs/orange_siling_text_redraw_sample.png")
let comparisonURL = URL(fileURLWithPath: "/Users/zenghang/Documents/Codex/2026-07-24/https-git-woa-com-freezeng-junqi-2/outputs/orange_siling_text_redraw_comparison.png")

let tileWidth = 36
let tileHeight = 27
let scale = 1
let colorSpace = CGColorSpaceCreateDeviceRGB()

func load(_ url: URL) -> CGImage {
    let source = CGImageSourceCreateWithURL(url as CFURL, nil)!
    return CGImageSourceCreateImageAtIndex(source, 0, nil)!
}

func makeRGBAContext(width: Int, height: Int, clear: Bool = false) -> CGContext {
    let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
                            bytesPerRow: width * 4, space: colorSpace,
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    if clear {
        context.clear(CGRect(x: 0, y: 0, width: width, height: height))
    }
    return context
}

func normalizedTile(from sheet: CGImage) -> (CGImage, [UInt8]) {
    let tile = sheet.cropping(to: CGRect(x: 360, y: 0, width: tileWidth, height: tileHeight))!
    let context = makeRGBAContext(width: tileWidth, height: tileHeight)
    context.interpolationQuality = .none
    context.draw(tile, in: CGRect(x: 0, y: 0, width: tileWidth, height: tileHeight))
    let bytes = Array(UnsafeBufferPointer(start: context.data!.assumingMemoryBound(to: UInt8.self),
                                           count: tileWidth * tileHeight * 4))
    return (context.makeImage()!, bytes)
}

func image(from bytes: [UInt8], width: Int, height: Int) -> CGImage {
    let data = Data(bytes)
    let provider = CGDataProvider(data: data as CFData)!
    return CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                   bytesPerRow: width * 4, space: colorSpace,
                   bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                   provider: provider, decode: nil, shouldInterpolate: false,
                   intent: .defaultIntent)!
}

func eraseOldGlyph(from input: [UInt8]) -> CGImage {
    var pixels = input
    var mask = Array(repeating: false, count: tileWidth * tileHeight)
    for y in 4..<25 {
        for x in 1..<35 {
            let p = (y * tileWidth + x) * 4
            let r = Int(input[p]), g = Int(input[p + 1]), b = Int(input[p + 2])
            let brightest = max(r, max(g, b))
            let darkest = min(r, min(g, b))
            // The glyph core is nearly neutral white; the orange background is
            // deliberately excluded by its high chroma.
            if brightest > 160 && brightest - darkest < 150 && r > 165 && g > 135 && b > 55 {
                mask[y * tileWidth + x] = true
            }
        }
    }
    // Include a one-pixel antialiased fringe around each detected glyph core.
    var expanded = mask
    for y in 4..<25 {
        for x in 1..<35 where mask[y * tileWidth + x] {
            for dy in -1...1 where y + dy >= 4 && y + dy < 25 {
                for dx in -1...1 where x + dx >= 1 && x + dx < 35 {
                    expanded[(y + dy) * tileWidth + (x + dx)] = true
                }
            }
        }
    }
    mask = expanded
    for y in 4..<25 {
        for x in 1..<35 where mask[y * tileWidth + x] {
            // Reconstruct the local orange gradient from the median of the
            // unmasked pixels on this scanline. This avoids sampling a dark
            // border when a glyph stroke spans most of the row.
            var row: [[Int]] = []
            for xx in 1..<35 where !mask[y * tileWidth + xx] {
                let q = (y * tileWidth + xx) * 4
                row.append([Int(input[q]), Int(input[q + 1]), Int(input[q + 2])])
            }
            if !row.isEmpty {
                let sample = (0..<3).map { channel in
                    row.map { $0[channel] }.sorted()[row.count / 2]
                }
                let p = (y * tileWidth + x) * 4
                pixels[p] = UInt8(sample[0]); pixels[p + 1] = UInt8(sample[1]); pixels[p + 2] = UInt8(sample[2])
                pixels[p + 3] = 255
            }
        }
    }
    return image(from: pixels, width: tileWidth, height: tileHeight)
}

func textLayer() -> CGImage {
    let context = makeRGBAContext(width: tileWidth * scale, height: tileHeight * scale, clear: true)
    let font = CTFontCreateWithName("STHeitiSC-Medium" as CFString, 16.5 * CGFloat(scale), nil)
    let attrs: [NSAttributedString.Key: Any] = [.font: font, .foregroundColor: NSColor.white]
    let line = CTLineCreateWithAttributedString(NSAttributedString(string: "司令", attributes: attrs))
    var ascent: CGFloat = 0, descent: CGFloat = 0, leading: CGFloat = 0
    let lineWidth = CGFloat(CTLineGetTypographicBounds(line, &ascent, &descent, &leading))
    let baseline = (CGFloat(tileHeight * scale) - (ascent + descent)) / 2.0 + descent
    context.textPosition = CGPoint(x: (CGFloat(tileWidth * scale) - lineWidth) / 2.0, y: baseline)
    CTLineDraw(line, context)
    return context.makeImage()!
}

func compose(base: CGImage) -> CGImage {
    let layer = textLayer()
    let lowContext = makeRGBAContext(width: tileWidth, height: tileHeight, clear: true)
    lowContext.interpolationQuality = .high
    lowContext.draw(layer, in: CGRect(x: 0, y: 0, width: tileWidth, height: tileHeight))
    let outputContext = makeRGBAContext(width: tileWidth, height: tileHeight)
    outputContext.interpolationQuality = .none
    outputContext.draw(base, in: CGRect(x: 0, y: 0, width: tileWidth, height: tileHeight))
    outputContext.draw(lowContext.makeImage()!, in: CGRect(x: 0, y: 0, width: tileWidth, height: tileHeight))
    return outputContext.makeImage()!
}

func save(_ image: CGImage, to url: URL) {
    let destination = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil)!
    CGImageDestinationAddImage(destination, image, nil)
    precondition(CGImageDestinationFinalize(destination))
}

let (baseTile, bytes) = normalizedTile(from: load(sourceURL))
let cleanBase = eraseOldGlyph(from: bytes)
let sample = compose(base: cleanBase)
save(sample, to: sampleURL)

let comparison = makeRGBAContext(width: 780, height: 340)
comparison.setFillColor(NSColor(calibratedWhite: 0.13, alpha: 1).cgColor)
comparison.fill(CGRect(x: 0, y: 0, width: 780, height: 340))
comparison.interpolationQuality = .high
comparison.draw(baseTile, in: CGRect(x: 20, y: 45, width: 360, height: 270))
comparison.draw(sample, in: CGRect(x: 400, y: 45, width: 360, height: 270))
let labelFont = CTFontCreateWithName("Heiti SC" as CFString, 22, nil)
let labelAttrs: [NSAttributedString.Key: Any] = [.font: labelFont, .foregroundColor: NSColor.white]
for (label, x) in [("当前", 200.0), ("重绘文字", 580.0)] {
    let line = CTLineCreateWithAttributedString(NSAttributedString(string: label, attributes: labelAttrs))
    let width = CGFloat(CTLineGetTypographicBounds(line, nil, nil, nil))
    comparison.textPosition = CGPoint(x: x - width / 2.0, y: 12)
    CTLineDraw(line, comparison)
}
save(comparison.makeImage()!, to: comparisonURL)
