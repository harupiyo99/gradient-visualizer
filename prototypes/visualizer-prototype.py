# =============================================================================
# 参考文献: 『Pythonで動かして学ぶ! 新しい機械学習の教科書』 （著者：伊藤　真 氏）
# 
# このコードは上記書籍のサンプルプログラムを学習用に引用し、以下の機能を独自に拡張・変更したものです：
# 1. SymPyを使用した導関数の自動生成
# 2. ベクトルの大きさ（Magnitude）に基づいたカラーマップによる色分け表示
# =============================================================================

# ライブラリのインポート
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

import sympy

# 変数・関数の定義
w0s, w1s = sympy.symbols('w0 w1') # 代数的な変数の定義
f_sym = 5 * (sympy.exp((-1) * (w0s ** 2 + w1s ** 2)) + sympy.exp(-1 * ((w0s - 1) ** 2 + (w1s - 1) ** 2))) # 変数の定義

# 偏微分を計算
df_dw0_sym = sympy.diff(f_sym, w0s)
df_dw1_sym = sympy.diff(f_sym, w1s)

# SymPyの式を NumPyで使える関数に変換 (lambdify)
# 第1引数は引数のリスト、第2引数は式、第3引数は使用するライブラリ
f = sympy.lambdify((w0s, w1s), f_sym, 'numpy')
df_dw0 = sympy.lambdify((w0s, w1s), df_dw0_sym, 'numpy')
df_dw1 = sympy.lambdify((w0s, w1s), df_dw1_sym, 'numpy')

# 表示データの計算 ----------
w0_min, w0_max = -2, 2
w1_min, w1_max = -2, 2
w0_n, w1_n = 30, 30
w0 = np.linspace(w0_min, w0_max, w0_n)
w1 = np.linspace(w1_min, w1_max, w1_n)
ww0, ww1 = np.meshgrid(w0, w1)          # (D) グリッド座標の作成
f_num = f(ww0, ww1)                     # (E) fの値の計算
df_dw0_num = df_dw0(ww0, ww1)           #     fの偏微分の値の計算
df_dw1_num = df_dw1(ww0, ww1)           #     fの偏微分の値の計算

# 面で表示
ww00, ww01 = np.meshgrid(w0, w1)     # (A) グリッド座標の作成

plt.figure(figsize=(5, 3.5))
ax = plt.subplot(projection="3d")  # (B) 3Dグラフの準備
ax.plot_surface(                   # (C) サーフェスの描画
    ww00,                # x座標のデータ
    ww01,                # y座標のデータ
    f_num,                  # z座標のデータ
    rstride=1,          # 何行おきに線を引くか
    cstride=1,          # 何列おきに線を引くか
    alpha=0.3,          # 面の透明度
    color="skyblue",       # 面の色
    edgecolor="black",  # 線の色
)
ax.set_zticks((0, np.pi / 8))            # (D) z軸の目盛りの指定
ax.view_init(75, -95)              # (E) グラフの向きの指定
plt.show()

# グラフ描画 ----------
plt.figure(figsize=(9, 4))
plt.subplots_adjust(wspace=0.3)
# 等高線表示
plt.subplot(1, 2, 1)
cont = plt.contour(                     # (F) fの等高線表示
    ww0, ww1, f_num, levels=50, colors="black")
cont.clabel(fmt="%d", fontsize=8)
plt.xticks(range(w0_min, w0_max + 1, 1))
plt.yticks(range(w1_min, w1_max + 1, 1))
plt.xlim(w0_min - 0.5, w0_max + 0.5)
plt.ylim(w1_min - 0.5, w1_max + 0.5)
plt.xlabel("$w_0$", fontsize=14)
plt.ylabel("$w_1$", fontsize=14)

# ベクトル表示
plt.subplot(1, 2, 2)
plt.xlabel("$w_0$", fontsize=14)
plt.ylabel("$w_1$", fontsize=14)
plt.xticks(range(w0_min, w0_max + 1, 1))
plt.yticks(range(w1_min, w1_max + 1, 1))
plt.xlim(w0_min - 0.5, w0_max + 0.5)
plt.ylim(w1_min - 0.5, w1_max + 0.5)

# 変更箇所(色分け)
# ------
# 1. ベクトルの大きさを計算
grad = np.array([df_dw0_num, df_dw1_num])
magnitudes = np.linalg.norm(grad,axis = 0)

# 2. quiver の第5引数に大きさを渡し、cmap を指定
# units='height' や scale を調整すると見やすくなります
plt.quiver(ww0, ww1, df_dw0_num, df_dw1_num, magnitudes, 
           cmap='viridis', angles='xy', scale_units='xy', scale=10)

# カラーバーを表示（色の基準がわかるようになる）
plt.colorbar(label='Magnitude')
# ----------------
plt.show()
