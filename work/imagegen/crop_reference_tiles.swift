import CoreGraphics
import Foundation
import ImageIO

let inputURL = URL(fileURLWithPath: "/var/folders/k8/pdj7kydd1gb_7gjz07_j27xh0000gn/T/codex-clipboard-467a39ce-d248-4232-bdcb-d113fa2d6877.png")
let outputDir = URL(fileURLWithPath: "/Users/zenghang/Documents/Codex/2026-07-24/https-git-woa-com-freezeng-junqi-2/outputs/reference_orange_tiles_tight")
try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)

let source = CGImageSourceCreateWithURL(inputURL as CFURL, nil)!
let image = CGImageSourceCreateImageAtIndex(source, 0, nil)!
let width = image.width
let height = image.height
let colorSpace = CGColorSpaceCreateDeviceRGB()
let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
                        bytesPerRow: width * 4, space: colorSpace,
                        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
let pixels = context.data!.assumingMemoryBound(to: UInt8.self)

var mask = Array(repeating: false, count: width * height)
for y in 0..<height {
    for x in 0..<width {
        let p = (y * width + x) * 4
        let r = Int(pixels[p]), g = Int(pixels[p + 1]), b = Int(pixels[p + 2])
        mask[y * width + x] = r > 120 && r > g + 35 && g > b + 15 && b < 145
    }
}

struct Component { var area: Int; var minX: Int; var minY: Int; var maxX: Int; var maxY: Int }
var components: [Component] = []
var queue = [(Int, Int)]()
for y in 0..<height {
    for x in 0..<width where mask[y * width + x] {
        mask[y * width + x] = false
        queue.removeAll(keepingCapacity: true)
        queue.append((x, y))
        var head = 0
        var area = 0
        var minX = x, maxX = x, minY = y, maxY = y
        while head < queue.count {
            let (cx, cy) = queue[head]; head += 1; area += 1
            minX = min(minX, cx); maxX = max(maxX, cx); minY = min(minY, cy); maxY = max(maxY, cy)
            for dy in -1...1 {
                for dx in -1...1 where dx != 0 || dy != 0 {
                    let nx = cx + dx, ny = cy + dy
                    if nx >= 0 && nx < width && ny >= 0 && ny < height && mask[ny * width + nx] {
                        mask[ny * width + nx] = false
                        queue.append((nx, ny))
                    }
                }
            }
        }
        if area > 1000 { components.append(Component(area: area, minX: minX, minY: minY, maxX: maxX, maxY: maxY)) }
    }
}
components.sort { ($0.minY, $0.minX) < ($1.minY, $1.minX) }
precondition(components.count == 25, "expected 25 orange tiles, found \(components.count)")
var rows: [[Component]] = []
for component in components {
    if let last = rows.indices.last,
       abs(component.minY - rows[last][0].minY) < 30 {
        rows[last].append(component)
    } else {
        rows.append([component])
    }
}
for index in rows.indices { rows[index].sort { $0.minX < $1.minX } }
components = rows.flatMap { $0 }
precondition(rows.count == 6, "expected 6 rows, found \(rows.count)")

func save(_ image: CGImage, to url: URL) {
    let destination = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil)!
    CGImageDestinationAddImage(destination, image, nil)
    precondition(CGImageDestinationFinalize(destination))
}

var crops: [CGImage] = []
for (index, c) in components.enumerated() {
    // Use the detected orange card bounds without expanding into the railway
    // or green board that sits immediately behind several pieces.
    let x = c.minX
    let y = c.minY
    let right = c.maxX
    let bottom = c.maxY
    let rect = CGRect(x: x, y: y, width: right - x + 1, height: bottom - y + 1)
    let crop = image.cropping(to: rect)!
    crops.append(crop)
    save(crop, to: outputDir.appendingPathComponent(String(format: "tile_%02d.png", index + 1)))
}

let cellWidth = 106
let cellHeight = 82
let contact = CGContext(data: nil, width: cellWidth * 5, height: cellHeight * 6,
                        bitsPerComponent: 8, bytesPerRow: cellWidth * 5 * 4,
                        space: colorSpace, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
contact.setFillColor(CGColor(red: 0.12, green: 0.12, blue: 0.12, alpha: 1))
contact.fill(CGRect(x: 0, y: 0, width: cellWidth * 5, height: cellHeight * 6))
contact.interpolationQuality = .none
let columnCenters = [59, 167, 276, 384, 493]
var cropIndex = 0
for (rowIndex, rowComponents) in rows.enumerated() {
    for component in rowComponents {
        let crop = crops[cropIndex]
        cropIndex += 1
        let center = (component.minX + component.maxX) / 2
        let col = columnCenters.enumerated().min(by: { abs($0.element - center) < abs($1.element - center) })!.offset
        let contactRow = 5 - rowIndex
        let x = col * cellWidth + (cellWidth - crop.width) / 2
        let y = contactRow * cellHeight + (cellHeight - crop.height) / 2
        contact.draw(crop, in: CGRect(x: x, y: y, width: crop.width, height: crop.height))
    }
}
save(contact.makeImage()!, to: outputDir.appendingPathComponent("contact.png"))
print("cropped", crops.count, "tiles to", outputDir.path)
