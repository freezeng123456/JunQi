import AppKit
import CoreGraphics
import CoreText
import Foundation
import ImageIO

let root = "/Users/zenghang/Documents/Codex/2026-07-24/https-git-woa-com-freezeng-junqi-2/JunQi"
let originalURL = URL(fileURLWithPath: root).appendingPathComponent("work/imagegen/original-bmp/orange.bmp")
let sharpenedURL = URL(fileURLWithPath: root).appendingPathComponent("legacy_gui/res/orange.bmp")
let outputURL = URL(fileURLWithPath: "/Users/zenghang/Documents/Codex/2026-07-24/https-git-woa-com-freezeng-junqi-2/outputs/orange_siling_sharpen_comparison.png")

func load(_ url: URL) -> CGImage {
    let source = CGImageSourceCreateWithURL(url as CFURL, nil)!
    return CGImageSourceCreateImageAtIndex(source, 0, nil)!
}

func cropSiling(_ image: CGImage) -> CGImage {
    image.cropping(to: CGRect(x: 360, y: 0, width: 36, height: 27))!
}

func drawLabel(_ text: String, context: CGContext, centerX: CGFloat) {
    let font = CTFontCreateWithName("Heiti SC" as CFString, 22, nil)
    let attrs: [NSAttributedString.Key: Any] = [.font: font, .foregroundColor: NSColor.white]
    let line = CTLineCreateWithAttributedString(NSAttributedString(string: text, attributes: attrs))
    let bounds = CTLineGetBoundsWithOptions(line, [])
    context.textPosition = CGPoint(x: centerX - bounds.width / 2.0, y: 12)
    CTLineDraw(line, context)
}

let width = 780
let height = 340
let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
                        bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
context.setFillColor(NSColor(calibratedWhite: 0.13, alpha: 1).cgColor)
context.fill(CGRect(x: 0, y: 0, width: width, height: height))
context.interpolationQuality = .high
context.draw(cropSiling(load(originalURL)), in: CGRect(x: 20, y: 45, width: 360, height: 270))
context.draw(cropSiling(load(sharpenedURL)), in: CGRect(x: 400, y: 45, width: 360, height: 270))
drawLabel("原图", context: context, centerX: 200)
drawLabel("锐化", context: context, centerX: 580)
let destination = CGImageDestinationCreateWithURL(outputURL as CFURL, "public.png" as CFString, 1, nil)!
CGImageDestinationAddImage(destination, context.makeImage()!, nil)
precondition(CGImageDestinationFinalize(destination))
