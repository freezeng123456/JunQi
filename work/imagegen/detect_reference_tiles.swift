import CoreGraphics
import Foundation
import ImageIO

let inputURL = URL(fileURLWithPath: "/var/folders/k8/pdj7kydd1gb_7gjz07_j27xh0000gn/T/codex-clipboard-467a39ce-d248-4232-bdcb-d113fa2d6877.png")
let source = CGImageSourceCreateWithURL(inputURL as CFURL, nil)!
let image = CGImageSourceCreateImageAtIndex(source, 0, nil)!
let width = image.width
let height = image.height
let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
                        bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
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
print("components", components.count)
for (i, c) in components.enumerated() {
    print(i, "area", c.area, "bbox", c.minX, c.minY, c.maxX, c.maxY,
          "center", (c.minX + c.maxX) / 2, (c.minY + c.maxY) / 2)
}
