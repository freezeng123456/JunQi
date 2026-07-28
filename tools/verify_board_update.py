"""tools/verify_board_update.py — quick check board.py after refactor."""
from junqi_core.board import CELL_TABLE, cell_info

rails = sum(1 for ci in CELL_TABLE if ci.is_railway)
nine  = sum(1 for ci in CELL_TABLE if ci.is_nine_grid)
curves = sum(1 for ci in CELL_TABLE if ci.curve_rail > 0)
print(f'Rails: {rails}  (expect 73)')
print(f'NineGrid: {nine}  (expect 9)')
print(f'Curve rails: {curves}  (expect 48 = 12 per rail × 4)')
print()
# Verify specific curve rail assignments
print(f'(6,11) curve: {cell_info(6,11).curve_rail}  (expect RAIL1 = 1)')
print(f'(5,10) curve: {cell_info(5,10).curve_rail}  (expect RAIL1 = 1)')
print(f'(5,6)  curve: {cell_info(5,6).curve_rail}   (expect RAIL2 = 2)')
print(f'(10,5) curve: {cell_info(10,5).curve_rail}  (expect RAIL3 = 3)')
print(f'(11,10) curve: {cell_info(11,10).curve_rail} (expect RAIL4 = 4)')
print(f'(8,8) is_rail: {cell_info(8,8).is_railway}  (expect True)')
print(f'(8,8) is_9grid: {cell_info(8,8).is_nine_grid}  (expect True)')
