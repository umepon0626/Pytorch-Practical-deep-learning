# 1 章 画像分類と転移学習(VGG)

## 1.1

- convolution は前のチャンネルごとに，あとのチャンネル数だけフィルタを作っている．
- 1x1 の conv は計算量が結果的に減るので，最近のモデルで多く用いられている．
- Pytorch は(ch,h,w)で画像を扱うが PIL では(h,w,ch)で画像を扱う．
- torchvision 内に，画像の前処理(transform)を担当する transforms パッケージがある．それを使って，前処理をクラス簡単に記述できる．
- transform を連続で使用するときは，transforms.compose というものがある．

```
transforms.Compose([
    transforms.Resize(resize),
    transforms.CenterCrop(resize),
    ...
])
```

- 本読んでて思うけど，やっぱり型ヒントはなるべく付けるべき．特に PIL.Image なのか，Tensor なのかこんがらがる．
- pytorch での tensor は前の関数の情報とかを持っている(自動微分の仕組み的に)→ 出力を利用するときは detach してあげないといけない．
- 画像の augmentation ってエポックごとに適用されるんや．そらそうか
- 前処理クラスの処理を辞書で保持して処理を分けて，実際の処理のときに，引数でスイッチできたら，かなり楽やな．(本当か？？)
- ↑ どっちにしろ，valid と train でクラスを分けないと，肥大化するから NG
- iter は順番に collection から値を返すインターフェースを提供する．next で次の値を返す．
- net の最終層を付け替えるときは以下のようにする(これは nn.Sequential だからできるのかな？)

```
net.classifier[6] = nn.Linear(in_features=4096, out_features)
```

- pytorch のモデルを訓練モードにするときは`net.train()`とする．推論は`net.eval()`
- 損失関数 →criterion
- pytorch のニューラルネットで使う各種(loss やモデルのパーツ)は nn にある．ただし，最適化手法は`torch.optim`に存在
- 転移学習で学習・変化させるパラメータは，`requires_grad = True`(default)となっている．逆に変化させたくない場合は，`requires_grad=False`とする．
- optim に渡すパラメーターの内，`params=`には更新するパラメーターをリストで渡す．
- pytorch で train()と eval()を使うのは，両者で挙動が違う層があるから(Dropout など)
- [continue と pass の違い](https://seesaawiki.jp/python-project/d/continue%2cbreak%2cpass)
- [なぜ`optim.zero_grad()`を呼ぶ必要があるのか](https://teratail.com/questions/261005)
- `with torch.set_grad_enabled(phase=="train")`で学習時のみ勾配を更新する．つまり引数が True の時のみ勾配更新．
- GPU を使用するときは`device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")`として，モデルや変数に`.to(device)`とする．
  ↑ は net と入力(ラベル)に対してだけでよい．
- model の save&load は以下の方法で行う．

```
torch.save(model.state_dict(),Path)# save
model = TheModelClass()
model.load_state_dict(torch.load(Path))
```
