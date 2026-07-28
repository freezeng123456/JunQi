import AppKit
import CoreGraphics
import CoreText
import Foundation
import ImageIO

let generatedURL = URL(fileURLWithPath: "/Users/zenghang/.codex/generated_images/019f9039-ba32-7bd0-9786-b7f9fc26d1b2/exec-e3858f54-28ab-4a96-a31d-b63b166445bd.png")
let existingURL = URL(fileURLWithPath: "/Users/zenghang/Documents/Codex/2026-07-24/https-git-woa-com-freezeng-junqi-2/JunQi/work/imagegen/orange-siling-existing-large.png")
let candidateURL = URL(fileURLWithPath: "/Users/zenghang/Documents/Codex/2026-07-24/https-git-woa-com-freezeng-junqi-2/outputs/orange_siling_generated.png")
let compareURL = URL(fileURLWithPath: "/Users/zenghang/Documents/Codex/2026-07-24/https-git-woa-com-freezeng-junqi-2/outputs/orange_siling_comparison.png")

func load(_ url: URL) -> CGImage {
    let source = CGImageSourceCreateWithURL(url as CFURL, nil)!
    return CGImageSourceCreateImageAtIndex(source, 0, nil)!
}

func save(_ image: CGImage, to url: URL) {
    let destination = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil)!
    CGImageDestinationAddImage(destination, image, nil)
    precondition(CGImageDestinationFinalize(destination))
}

func drawLabel(_ text: String, in context: CGContext, centerX: CGFloat, y: CGFloat, fontSize: CGFloat) {
    let font = CTFontCreateWithName("Heiti SC" as CFString, fontSize, nil)
    let shadow = NSShadow()
    shadow.shadowColor = NSColor.black.withAlphaComponent(0.45)
    shadow.shadowOffset = NSSize(width: 2, height: -2)
    shadow.shadowBlurRadius = 1
    let attributes: [NSAttributedString.Key: Any] = [
        .font: font,
        .foregroundColor: NSColor.white,
        .shadow: shadow
    ]
    let line = CTLineCreateWithAttributedString(NSAttributedString(string: text, attributes: attributes))
    let bounds = CTLineGetBoundsWithOptions(line, [])
    context.textPosition = CGPoint(x: centerX - bounds.width / 2.0, y: y - bounds.height / 2.0)
    CTLineDraw(line, context)
}

func makeCandidate() -> CGImage {
    let source = load(generatedURL)
    // The generated card is centered with a neutral margin. These bounds contain
    // only the straight-edged tile and keep its 4:3-ish proportions.
    let crop = source.cropping(to: CGRect(x: 90, y: 90, width: 1268, height: 904))!
    let width = 360
    let height = 270
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
                            bytesPerRow: width * 4, space: colorSpace,
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    context.interpolationQuality = .high
    context.draw(crop, in: CGRect(x: 0, y: 0, width: width, height: height))
    drawLabel("司令", in: context, centerX: CGFloat(width) / 2.0, y: CGFloat(height) / 2.0, fontSize: 112)
    return context.makeImage()!
}

func makeComparison(existing: CGImage, candidate: CGImage) -> CGImage {
    let width = 780
    let height = 340
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
                            bytesPerRow: width * 4, space: colorSpace,
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    context.setFillColor(NSColor(calibratedWhite: 0.13, alpha: 1).cgColor)
    context.fill(CGRect(x: 0, y: 0, width: width, height: height))
    context.interpolationQuality = .high
    context.draw(existing, in: CGRect(x: 20, y: 45, width: 360, height: 270))
    context.draw(candidate, in: CGRect(x: 400, y: 45, width: 360, height: 270))
    drawLabel("当前", in: context, centerX: 200, y: 22, fontSize: 22)
    drawLabel("生图候选", in: context, centerX: 580, y: 22, fontSize: 22)
    return context.makeImage()!
}

let candidate = makeCandidate()
save(candidate, to: candidateURL)
save(makeComparison(existing: load(existingURL), candidate: candidate), to: compareURL)
