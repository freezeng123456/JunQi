# 单枚棋子资源

这里的图片是按照参考截图逐枚裁剪得到的，不是重新绘制的棋子样式。

- 每个颜色目录包含 13 张 36×27 BMP：`dark` 加 12 种明棋。
- 12 种明棋都由 96×72 的独立母版缩放得到；母版保存在 `assets/chess_pieces/reference`。
- `orange`、`green` 来自截图中的横向棋子区。
- `blue`、`purple` 来自截图中的纵向棋子区，裁剪后分别旋转 270°、90°，使四个方向都使用同一张横向基准图。
- 截图没有单独展示暗棋，因此 `dark.bmp` 沿用项目原始暗棋图片，四种颜色共用。

文件名与 `enum ChessType` 对应：`junqi`、`dilei`、`zhadan`、`siling`、`junzh`、`shizh`、`lvzh`、`tuanzh`、`yingzh`、`lianzh`、`paizh`、`gongb`。

客户端会优先读取这些单枚资源；在旧安装包缺少 `res/pieces` 时，自动回退到由同一批单枚资源生成的四条颜色图片。

精确坐标、文字识别结果和全量预览见 `assets/chess_pieces/README.md`。
