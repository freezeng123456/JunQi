"""tools/decode_missing_actions.py — decode flat action ids into coordinates."""

NC = 289
BOARD_SIZE = 17


def decode(aid):
    src = aid // NC
    dst = aid % NC
    sx, sy = src % BOARD_SIZE, src // BOARD_SIZE
    dx, dy = dst % BOARD_SIZE, dst // BOARD_SIZE
    return (sx, sy), (dx, dy)


# Missing from batched: 55952, 57114
for a in (55952, 57114):
    s, d = decode(a)
    print(f"  {a}: {s} -> {d}")
