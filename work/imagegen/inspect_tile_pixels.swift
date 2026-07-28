import CoreGraphics
import Foundation
import ImageIO
let url = URL(fileURLWithPath: "/Users/zenghang/Documents/Codex/2026-07-24/https-git-woa-com-freezeng-junqi-2/JunQi/work/imagegen/original-bmp/orange.bmp")
let source = CGImageSourceCreateWithURL(url as CFURL, nil)!
let sheet = CGImageSourceCreateImageAtIndex(source, 0, nil)!
let tile = sheet.cropping(to: CGRect(x: 360, y: 0, width: 36, height: 27))!
let cs = CGColorSpaceCreateDeviceRGB()
let c = CGContext(data: nil, width: 36, height: 27, bitsPerComponent: 8, bytesPerRow: 144, space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
c.draw(tile, in: CGRect(x: 0, y: 0, width: 36, height: 27))
let p = c.data!.assumingMemoryBound(to: UInt8.self)
var values: [(Int, Int, Int)] = []
for y in 4..<25 { for x in 1..<35 { let i=(y*36+x)*4; values.append((Int(p[i]),Int(p[i+1]),Int(p[i+2]))) } }
for t in values.sorted(by: { $0.2 > $1.2 }).prefix(30) { print(t) }
var buckets = [Int: Int]()
for t in values { buckets[t.2 / 10, default: 0] += 1 }
print("blue buckets", buckets.sorted { $0.key < $1.key })
