import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

CIKTI = '/content/onisleme_cikti'
os.makedirs(f'{CIKTI}/birlestirme_grafikleri', exist_ok=True)

df_baraj = pd.read_csv(f'{CIKTI}/baraj_islenmis.csv')
df_hava  = pd.read_csv(f'{CIKTI}/hava_islenmis.csv')
df_su    = pd.read_csv(f'{CIKTI}/su_islenmis.csv')

df_merge = (df_baraj
            .merge(df_hava, on=['YIL', 'AY'], how='inner')
            .merge(df_su,   on=['YIL', 'AY'], how='inner'))
df_merge = df_merge.sort_values(['YIL', 'AY']).reset_index(drop=True)

sns.set_theme(style="whitegrid")

BARAJLAR = [c for c in df_merge.columns
            if 'Barajı' in c and '_PREV' not in c and c not in ['YIL', 'AY', 'MEVSIM']]

model_sutunlar = BARAJLAR + ['SICAKLIK_ORT', 'YAGIS', 'RADYASYON',
                              'ONCEKI_YAGIS', 'YAGIS_3AY_LOG', 'SU_TUKETIM', 'MEVSIM']
model_sutunlar = [c for c in model_sutunlar if c in df_merge.columns]

plt.figure(figsize=(14, 12))
sns.heatmap(df_merge[model_sutunlar].corr(), annot=True, fmt='.2f',
            cmap='RdBu_r', center=0, linewidths=0.3)
plt.title('Birleşik Veri Seti — Tüm Değişkenler Korelasyon Matrisi', fontsize=13)
plt.tight_layout()
plt.savefig(f'{CIKTI}/birlestirme_grafikleri/tum_korelasyon.png', dpi=150, bbox_inches='tight')
plt.close()

hava_cols = ['SICAKLIK_ORT', 'YAGIS', 'RADYASYON', 'ONCEKI_YAGIS', 'YAGIS_3AY_LOG', 'SU_TUKETIM']
hava_cols = [c for c in hava_cols if c in df_merge.columns]

kor_baraj_hava = df_merge[BARAJLAR + hava_cols].corr().loc[BARAJLAR, hava_cols]
plt.figure(figsize=(12, max(4, len(BARAJLAR) * 1.2)))
sns.heatmap(kor_baraj_hava, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, linewidths=0.5, cbar_kws={'label': 'Pearson r'})
plt.title('Baraj Doluluğu × Hava/Tüketim Değişkenleri Korelasyonu', fontsize=13)
plt.tight_layout()
plt.savefig(f'{CIKTI}/birlestirme_grafikleri/baraj_hava_korelasyon.png', dpi=150, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, len(BARAJLAR), figsize=(20, 5))
fig.suptitle('Yağış — Baraj Doluluğu İlişkisi', fontsize=13, fontweight='bold')
renkler = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
for i, (baraj, renk) in enumerate(zip(BARAJLAR, renkler)):
    ax = axes[i]
    ax.scatter(df_merge['YAGIS'], df_merge[baraj], color=renk, alpha=0.6, s=40)
    kisaltma = baraj.replace(' Barajı', '').replace('Alaçatı Kutlu Aktaş', 'Alaçatı')
    ax.set_title(kisaltma, fontsize=10, fontweight='bold')
    ax.set_xlabel('Aylık Yağış (mm)')
    if i == 0:
        ax.set_ylabel('Doluluk (%)')
plt.tight_layout()
plt.savefig(f'{CIKTI}/birlestirme_grafikleri/yagis_doluluk_scatter.png', dpi=150, bbox_inches='tight')
plt.close()

ozet = df_merge[BARAJLAR + hava_cols].describe().round(2)
fig, ax = plt.subplots(figsize=(14, 5))
ax.axis('off')
tablo = ax.table(cellText=ozet.values.round(2),
                 rowLabels=ozet.index,
                 colLabels=ozet.columns,
                 cellLoc='center', loc='center')
tablo.auto_set_font_size(False)
tablo.set_fontsize(8)
tablo.scale(1.2, 1.4)
plt.title('Birleşik Veri Seti — Temel İstatistikler', fontsize=13, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f'{CIKTI}/birlestirme_grafikleri/ozet_istatistikler.png', dpi=150, bbox_inches='tight')
plt.close()

df_merge.to_csv(f'{CIKTI}/birlesmis_veri.csv', index=False, encoding='utf-8-sig')

try:
    from IPython.display import display, FileLink
    for f in sorted(os.listdir(f'{CIKTI}/birlestirme_grafikleri')):
        display(FileLink(f'{CIKTI}/birlestirme_grafikleri/{f}',
                result_html_prefix=f"{f} → "))
    display(FileLink(f'{CIKTI}/birlesmis_veri.csv',
            result_html_prefix="birlesmis_veri.csv → "))
except Exception:
    pass