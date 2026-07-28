import CoreGraphics
import CoreImage
import Foundation
import ImageIO

let root = "/Users/zenghang/Documents/Codex/2026-07-24/https-git-woa-com-freezeng-junqi-2/JunQi"
let outputDir = URL(fileURLWithPath: root).appendingPathComponent("work/imagegen/sharpened")
try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)

let colors = ["orange", "purple", "green", "blue"]
let tileWidth = 36
let tileHeight = 27
let context = CIContext(options: [.useSoftwareRenderer: false])

func loadImage(_ url: URL) -> CGImage {
    let source = CGImageSourceCreateWithURL(url as CFURL, nil)!
    return CGImageSourceCreateImageAtIndex(source, 0, nil)!
}

func savePNG(_ image: CGImage, to url: URL) {
    let destination = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil)!
    CGImageDestinationAddImage(destination, image, nil)
    precondition(CGImageDestinationFinalize(destination))
}

func sharpen(_ tile: CGImage) -> CGImage {
    let input = CIImage(cgImage: tile)
    // Sharpen luminance only.  CIUnsharpMask can overshoot the RGB channels on
    // the white glyphs and create red/blue halos on these tiny sprites.
    let filter = CIFilter(name: "CISharpenLuminance")!
    filter.setValue(input, forKey: kCIInputImageKey)
    filter.setValue(0.15, forKey: "inputSharpness")
    let output = filter.outputImage!.cropped(to: input.extent)
    return context.createCGImage(output, from: input.extent)!
}

for color in colors {
    let sourceURL = URL(fileURLWithPath: root).appendingPathComponent("legacy_gui/res/\(color).bmp")
    let source = loadImage(sourceURL)
    precondition(source.width % tileWidth == 0 && source.height == tileHeight)
    let tileCount = source.width / tileWidth
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let canvas = CGContext(data: nil, width: source.width, height: source.height,
                           bitsPerComponent: 8, bytesPerRow: source.width * 4,
                           space: colorSpace,
                           bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    canvas.interpolationQuality = .none
    for index in 0..<tileCount {
        let rect = CGRect(x: index * tileWidth, y: 0, width: tileWidth, height: tileHeight)
        let tile = source.cropping(to: rect)!
        canvas.draw(sharpen(tile), in: rect)
    }
    savePNG(canvas.makeImage()!, to: outputDir.appendingPathComponent("\(color)-sharp.png"))
}
