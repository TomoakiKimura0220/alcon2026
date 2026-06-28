水田のイネと雑草を識別するAI基盤データ「RiceSEG」公開――スマート農業と育種を加速する国際共同研究――
https://www.a.u-tokyo.ac.jp/topics/topics_20250916-1.html
これめっちゃ使えそう

# ALCON2026 水田の景観を損ねる雑草の程度の定量化

## 概要

ALCON2026の課題「水田の景観を損ねる雑草の程度の定量化」に対するプロトタイプ実装。

入力画像から水稲領域と雑草領域を推定し、以下を出力する。

- クラスごとに色分けした出力画像
- 水稲画素数 `p`
- 雑草画素数 `w`
- 雑草比率 `r = w / (p + w) * 100`
- 雑草量の4段階判定 `judge`

現時点では提出用の完成版ではなく、RiceSEGを用いた推論プロトタイプである。

## 参考データセット

### RiceSEG

水田のイネと雑草を識別するAI基盤データ「RiceSEG」公開――スマート農業と育種を加速する国際共同研究――  
https://www.a.u-tokyo.ac.jp/topics/topics_20250916-1.html

Hugging Face:  
https://huggingface.co/datasets/PheniX-Lab/RiceSEG

RiceSEGは、水稲圃場画像に対するセマンティックセグメンテーション用データセットである。ラベルは以下の6クラスで構成されている。

| ID | クラス |
|---:|---|
| 0 | background |
| 1 | green vegetation |
| 2 | senescent vegetation |
| 3 | panicle |
| 4 | weeds |
| 5 | duckweed |

ALCONの出力仕様に合わせるため、最初のプロトタイプでは以下の3クラスに統合して学習した。

| ALCON用ID | 意味 | RiceSEGクラス |
|---:|---|---|
| 0 | other | background |
| 1 | rice | green vegetation, senescent vegetation, panicle |
| 2 | weed | weeds, duckweed |

## 現在の処理フロー

```text
input.csv
↓
画像読み込み
↓
RiceSEG U-Netで rice / weed 推論
↓
rice_mask, weed_mask 作成
↓
色分け出力画像を保存
↓
p, w, r, judge を output.csv に保存
```

現在の出力画像は目視確認しやすいように以下の色にしている。

| クラス | 色 |
|---|---|
| other | 黒 |
| rice | 緑 |
| weed | 赤 |

注意: ALCON提出仕様では、出力画像は以下に戻す必要がある。

| クラス | 提出仕様の色 |
|---|---|
| other | 黒 `0x00,0x00,0x00` |
| rice | 灰色 `0x80,0x80,0x80` |
| weed | 白 `0xff,0xff,0xff` |

## 実装ファイル

| ファイル | 役割 |
|---|---|
| `main.py` | input.csv読み込み、推論実行、画像保存、output.csv保存 |
| `io_utils.py` | CSV入出力、Windows風パス変換、出力パス生成 |
| `riceseg_dataset.py` | RiceSEGデータセット読み込み、ALCON用3クラス変換 |
| `train_riceseg_unet.py` | RiceSEG Japanデータを用いたU-Net学習 |
| `predict_riceseg_val.py` | RiceSEG検証画像に対する推論確認 |
| `riceseg_segmenter.py` | ALCON公開データに対するRiceSEG U-Net推論 |
| `yolo_segmenter.py` | YOLO-seg版の試作用コード |
| `segmenter.py` | 古典処理・仮処理用の初期プロトタイプ |

## checkpoint配置

学習済み `.pth` はGit管理しない。

ローカル実行時は以下に配置する。

```text
alcon2026/
└── checkpoints/
    └── riceseg_unet_alcon3_w45_best.pth
```

`riceseg_segmenter.py` の `CHECKPOINT_PATH` が参照するファイルと一致させること。

例:

```python
CHECKPOINT_PATH = Path("checkpoints/riceseg_unet_alcon3_w45_best.pth")
```

## これまでの試行錯誤

### 1. YOLO-segの検討

最初にYOLO-seg形式での推論を検討した。

しかし、ALCONの出力はインスタンス単位ではなく画素単位の `rice / weed / other` である。そのため、YOLO-segよりもセマンティックセグメンテーションの方が自然であると判断した。

結論:

```text
YOLO-segは試作・比較対象としては使えるが、ALCON本命はセマンティックセグメンテーション。
```

### 2. RiceSEGを生のセマンティック形式で使用

RiceSEGのラベル画像を確認したところ、RGB色画像ではなく、画素値がクラスIDになっていることを確認した。

例:

```text
label unique: [0, 1, 4]
label unique: [0, 1, 2, 4]
```

そのため、YOLO形式へ変換せず、U-Netで直接学習する方針にした。

### 3. 6クラス学習

最初はRiceSEG本来の6クラスで学習した。

結果として、背景・稲系クラスはある程度学習できたが、weed / duckweed はクラス不均衡の影響で出にくかった。

また、ALCONの評価に必要なのは最終的な `rice / weed / other` であるため、途中から3クラス学習に切り替えた。

### 4. ALCON用3クラス学習

RiceSEGの6クラスを以下の3クラスに統合した。

```text
0 other = background
1 rice  = green vegetation + senescent vegetation + panicle
2 weed  = weeds + duckweed
```

Japanデータ704枚を対象に、ランダム8:2分割で学習した。

```text
train images: 563
val images  : 141
image_size  : 512
batch_size  : 4
model       : U-Net
```

### 5. class weightの調整

weedクラスが少ないため、CrossEntropyLossにclass weightを設定した。

試した重み:

```text
w3.0: [0.5, 1.0, 3.0]
w4.0: [0.5, 1.0, 4.0]
w4.5: [0.5, 1.0, 4.5]
w5.0: [0.5, 1.0, 5.0]
```

傾向:

| weed weight | 傾向 |
|---:|---|
| 3.0 | weedをほとんど出さない |
| 4.0 | 過検出は少ないが、weedの取り逃がしが多い |
| 4.5 | weedをある程度拾う。過検出も一部ある。現時点のバランス型 |
| 5.0 | weedを出しやすいが、過検出が強い |

採用候補:

```text
riceseg_unet_alcon3_w45_best.pth
```

### 6. RiceSEG検証画像での確認

RiceSEG Japan画像に対して推論した結果、w4.5はw4.0よりweedを拾うようになった。

一方で、true weedが0の画像に対してpred weedが出る例もあり、false positiveは残った。

代表例:

```text
true weed: 7635
pred weed: 6926
```

のように良好なケースもあるが、

```text
true weed: 0
pred weed: 数千
```

のような過検出も見られた。

### 7. ALCON公開データでの全体リサイズ推論

最初のALCON公開データ推論では、画像全体を512×512に縮小して推論した。

結果:

- `.pth` の読み込みは成功
- `output.csv` の生成は成功
- 出力画像の保存は成功
- ただし、背景がrice / weedとして誤分類された
- 青空がriceとして塗られる例があった
- `judge` は暫定閾値の影響でほぼ0になった

全体リサイズでは、ALCON画像の遠景・空・建物・森などがRiceSEGの学習分布と異なるため、ドメインギャップが大きい。

### 8. HSVベースの田んぼ領域マスク

背景除去のため、HSVベースで田んぼ領域らしいマスクを作る処理を追加した。

目的:

```text
背景・空・建物などに出た rice / weed の誤検出を抑える
```

追加した関数:

```python
estimate_field_mask(image_bgr)
apply_field_mask(rice_mask, weed_mask, field_mask)
make_field_mask_debug_image(image_bgr, field_mask)
```

ただし、色ベースの処理では森・木・畦草などの緑背景と水田領域を安定して分離するのが難しい。

また、画像上部を強制的にotherにするようなルールは、公開データに対する過剰適合になりやすいため、本命構成としては避ける方針。

### 9. パッチ分割推論

RiceSEGは512×512パッチで学習されているため、ALCON画像全体を512×512に縮小せず、元画像を512×512パッチに分けて推論する方式を試した。

設定:

```python
PATCH_SIZE = 512
PATCH_STRIDE = 512
```

目的:

```text
RiceSEGの学習条件に近い入力サイズで推論する
```

結果:

- 全体リサイズよりweedが出るようになった
- 建物はotherになりやすくなった
- しかし処理が非常に遅い
- パッチ境界が目立つ
- 森・木がweed扱いされる
- 実った穂がweed扱いされる

ALCON公開データでの例:

```text
000/IMG_5568.JPG  r=19.8
111/P1060914.JPG  r=24.9
222/P1060910.JPG  r=25.0
333/P1060923.JPG  r=25.2
```

`111 / 222 / 333` が近い値に固まり、`000` でも高いrが出たため、パッチ分割単体では十分ではない。

## 現時点の課題

### 1. ドメインギャップ

RiceSEGの学習画像は基本的に水田内パッチである。一方、ALCON公開データは景観画像であり、以下が写り込む。

- 青空
- 雲
- 建物
- 道路
- 森
- 木
- 畦
- 遠景

そのため、RiceSEGモデルは背景をrice / weedとして誤分類する。

### 2. 背景と田んぼ領域の分離

ALCON画像では、まず画像中の田んぼ領域を抽出する必要がある。

```text
背景 / 前景、つまり田んぼ領域の区別
↓
田んぼ領域内で rice / weed の区別
```

という二段構えが必要と考えられる。

### 3. 森・木・遠景植生の誤検出

森や木は色・テクスチャとしては植生であるため、RiceSEGモデルがweedとして扱いやすい。

色ベースのHSV処理では、田んぼ内の稲・雑草と、遠景の森・木を安定して区別するのは難しい。

### 4. 実った穂の誤分類

ALCON公開データでは、実った穂がweedとして判定される例がある。

3クラスに統合して学習したことで、RiceSEG本来の `panicle` クラス情報が失われ、weedとの混同が起きている可能性がある。

### 5. パッチ境界

重なりなしのパッチ分割では、パッチ境界が出力画像に現れる。

重なりあり推論を行えば軽減できる可能性があるが、CPU推論ではさらに処理時間が増える。

### 6. 処理速度

Mac CPUでパッチ分割推論を行うと非常に遅い。

提出・評価実行環境を考えると、推論速度の改善が必要である。

### 7. judge閾値

現在の `judge_level()` は暫定である。

```python
if ratio < 25.0:
    return 0
if ratio < 50.0:
    return 1
if ratio < 75.0:
    return 2
return 3
```

ALCON公開データで得られる `r` の範囲とは合っていないため、公開データ・検証データに基づいて調整する必要がある。

## 今後のアイデア

### 1. field / non-field セグメンテーション

最も筋が良い構成は、前段で田んぼ領域を抽出し、後段で田んぼ領域内のrice / weedを分類する二段構えである。

```text
ALCON景観画像
↓
field / non-field セグメンテーション
↓
field領域だけRiceSEG推論
↓
rice / weed / other
↓
p, w, r, judge
```

前段モデルのクラス:

```text
0: non-field = 空・建物・道路・森・山など
1: field     = 水田領域
```

ALCON公開データ10枚に対して、粗い田んぼ領域マスクを作るだけなら、rice / weedを細かく塗るより負担は小さい。

### 2. パッチ単位のfield判定

画素単位のfield segmentationより簡単な方法として、512×512パッチごとにfield / non-fieldを判定する方法がある。

```text
画像を512×512パッチに分割
↓
各パッチが田んぼか非田んぼか判定
↓
田んぼパッチだけRiceSEG推論
```

ただし、田んぼと背景が混在する境界パッチの扱いが難しい。

### 3. SAM等を使った半自動アノテーション

SAMなどを用いて田んぼ領域の初期マスクを作成し、手動修正してfield / non-field教師データを作る。

本番推論にSAMを組み込むというより、学習用マスク作成の補助として使う。

### 4. 6クラス学習へ戻す

現在はALCON用3クラスに統合して学習しているが、実った穂の誤分類を抑えるには、RiceSEG本来の6クラスで学習してから推論後に3クラスへ統合する方が自然である。

```text
推論時:
  0 background           → other
  1 green vegetation     → rice
  2 senescent vegetation → rice
  3 panicle              → rice
  4 weeds                → weed
  5 duckweed             → weed
```

この方式なら、`panicle` を明示的にriceへ統合できる。

### 5. field_mask + パッチ推論

最終的には、田んぼ領域抽出とパッチ推論を組み合わせる。

```text
ALCON画像
↓
field_mask作成
↓
field領域内を中心に512×512パッチ推論
↓
field_mask外をotherに強制
↓
p, w, rを計算
```

ただし、field_maskをHSVなどの手作りルールだけで作ると限界があるため、field / non-fieldモデル化を優先する。

### 6. weed_mask後処理

推論後のweed_maskに対して、小面積除去や形状フィルタを入れる。

候補:

- 小さい連結成分の除去
- rice領域に囲まれた小さなweedをriceへ戻す
- 孤立したweed点を除去

ただし、雑草自体が小さい場合もあるため、過剰な後処理は避ける必要がある。

### 7. 推論速度改善

候補:

- 全体リサイズ版とパッチ版の使い分け
- field判定で不要パッチをスキップ
- `PATCH_STRIDE` の調整
- モデル軽量化
- ONNX化
- GPU環境での推論

## 現時点の結論

プロトタイプとして、以下は達成できた。

```text
RiceSEGの読み込み
RiceSEG JapanデータでのU-Net学習
class weight調整
ALCON公開データでの推論
output画像生成
output.csv生成
パッチ分割推論の検証
```

一方で、ALCON公開データはRiceSEGの学習分布と異なるため、RiceSEG単体では不十分である。

今後の本命構成は以下。

```text
field / non-field で田んぼ領域を抽出
↓
田んぼ領域内でRiceSEG 6クラスまたは3クラス推論
↓
ALCON仕様の3クラスへ統合
↓
p, w, r, judgeを計算
```

特に、次に取り組むべきことは以下である。

```text
1. ALCON公開データに対してfield / non-field教師マスクを作成
2. 前段のfield segmentationモデルを作成
3. field領域内でRiceSEG推論
4. 可能なら6クラス推論後にALCON 3クラスへ統合
```