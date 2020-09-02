# 1章 画像分類と転移学習(VGG)

## 1.1
- convolutionは前のチャンネルごとに，あとのチャンネル数だけフィルタを作っている．
- 1x1のconvは計算量が結果的に減るので，最近のモデルで多く用いられている．
- Pytorchは(ch,h,w)で画像を扱うがPILでは(h,w,ch)で画像を扱う．
- torchvision内に，画像の前処理(transform)を担当するtansformsパッケージがある．それを使って，前処理をクラス簡単に記述できる．
- transformを連続で使用するときは，transforms.composeというものがある．
```
transforms.Compose([
    transforms.Resize(resize),
    transforms.CenterCrop(resize),
    ...
])
```
- 本読んでて思うけど，やっぱり型ヒントはなるべく付けるべき．特にPIL.Imageなのか，Tensorなのかこんがらがる．
- pytorchでのtensorは前の関数の情報とかを持っている(自動微分の仕組み的に)→出力を利用するときはdetouchしてあげないといけない．
- 画像のaugmentetionってエポックごとに適用されるんや．そらそうか
- 前処理クラスの処理を辞書で保持して処理を分けて，実際の処理のときに，引数でスイッチできたら，かなり楽やな．(本当か？？)
- ↑どっちにしろ，validとtrainでクラスを分けないと，肥大化するからNG
