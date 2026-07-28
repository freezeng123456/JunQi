import CoreText
for name in ["Heiti SC", "STHeitiSC-Medium", "STHeiti Medium", "PingFangSC-Semibold", "Songti SC"] {
    let f = CTFontCreateWithName(name as CFString, 16, nil)
    print(name, "=>", CTFontCopyPostScriptName(f), CTFontGetAscent(f), CTFontGetDescent(f))
}
