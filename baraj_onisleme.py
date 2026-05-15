import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

CIKTI = '/content/onisleme_cikti'
os.makedirs(f'{CIKTI}/baraj_grafikleri', exist_ok=True)

DONMUS_ESIK        = 3
SIGMA              = 3
WINDOW_SIZE        = 12
SERT_DEGISIM_ESIGI = 50

df_raw      = pd.read_csv('izmir_baraj_gunluk.csv')
date_col    = 'Date'         if 'Date'         in df_raw.columns else 'tarih'
baraj_col   = 'baraj_adi'    if 'baraj_adi'    in df_raw.columns else 'Reservoir'
doluluk_col = 'doluluk_oran' if 'doluluk_oran' in df_raw.columns else 'OccupancyRate'

df_raw[date_col] = pd.to_datetime(df_raw[date_col])
mask = (df_raw[date_col] >= '2014-01-01') & (df_raw[date_col] <= '2023-12-31')
df_raw = df_raw.loc[mask].copy()
df_raw['YIL'] = df_raw[date_col].dt.year
df_raw['AY']  = df_raw[date_col].dt.month

df_b = df_raw.groupby(["YIL", "AY", baraj_col])[doluluk_col].mean().unstack().reset_index()
df_b.columns.name = None

BARAJLAR_TERCIH = [
    'Alaçatı Kutlu Aktaş Barajı', 'Balçova Barajı',
    'Gördes Barajı', 'Tahtalı Barajı', 'Ürkmez Barajı'
]
sutunlar = [c for c in df_b.columns if c not in ["YIL", "AY"]]
BARAJLAR = [b for b in BARAJLAR_TERCIH if b in sutunlar] or sutunlar

def donmus_indeksleri_bul(seri, min_tekrar):
    degerler = seri.values
    sorunlu  = []
    i = 0
    while i < len(degerler):
        j = i + 1
        while j < len(degerler) and np.isclose(degerler[j], degerler[i], atol=1e-5):
            j += 1
        if j - i >= min_tekrar:
            sorunlu.extend(range(i, j))
        i = j
    return sorunlu

def ay_to_mevsim(ay):
    if ay in [12, 1, 2]: return 1
    if ay in [3, 4, 5]:  return 2
    if ay in [6, 7, 8]:  return 3
    return 4

df_oncesi = df_b.copy()
toplam_anomali = {}

for baraj in BARAJLAR:
    df_b[baraj] = pd.to_numeric(df_b[baraj], errors="coerce")
    anomali = 0

    idx = donmus_indeksleri_bul(df_b[baraj].fillna(-999), DONMUS_ESIK)
    if idx:
        df_b.loc[df_b.index[idx], baraj] = np.nan
        anomali += len(idx)

    degisim = df_b[baraj].diff().abs()
    sert    = (degisim > SERT_DEGISIM_ESIGI).sum()
    df_b.loc[degisim > SERT_DEGISIM_ESIGI, baraj] = np.nan
    anomali += sert

    aylik_ort = df_b.groupby("AY")[baraj].transform("mean")
    df_b[baraj] = df_b[baraj].fillna(aylik_ort)
    df_b[baraj] = df_b[baraj].interpolate(method="linear").bfill().ffill()

    r_mean = df_b[baraj].rolling(window=WINDOW_SIZE, center=True, min_periods=1).mean()
    r_std  = df_b[baraj].rolling(window=WINDOW_SIZE, center=True, min_periods=1).std()
    ust = (r_mean + SIGMA * r_std).clip(upper=100)
    alt = (r_mean - SIGMA * r_std).clip(lower=0)
    df_b[baraj] = df_b[baraj].clip(lower=alt, upper=ust)

    toplam_anomali[baraj] = anomali

assert df_b[BARAJLAR].isnull().sum().sum() == 0
assert (df_b[BARAJLAR] >= 0).all().all() and (df_b[BARAJLAR] <= 100).all().all()

df_b['MEVSIM'] = df_b['AY'].apply(ay_to_mevsim)
for baraj in BARAJLAR:
    df_b[f"{baraj}_PREV"] = df_b[baraj].shift(1).fillna(df_b[baraj].iloc[0])

sns.set_theme(style="whitegrid")

for baraj in BARAJLAR:
    plt.figure(figsize=(12, 6))
    pivot = df_b.pivot(index="AY", columns="YIL", values=baraj)
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu",
                cbar_kws={'label': 'Doluluk (%)'})
    plt.title(f"{baraj} — Aylık Doluluk Analizi (2014-2023)", fontsize=13)
    dosya = baraj.replace(' ', '_').replace('ı','i').replace('İ','I').replace('ç','c').replace('Ç','C')
    plt.savefig(f'{CIKTI}/baraj_grafikleri/{dosya}_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()

for baraj in BARAJLAR:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x='MEVSIM', y=baraj, data=df_b, hue='MEVSIM', palette='husl', legend=False)
    plt.xticks(ticks=[0,1,2,3], labels=['Kış','İlkbahar','Yaz','Sonbahar'])
    plt.title(f"{baraj} — Mevsimsel Doluluk Dağılımı", fontsize=13)
    plt.ylabel("Doluluk (%)")
    dosya = baraj.replace(' ', '_').replace('ı','i').replace('İ','I').replace('ç','c').replace('Ç','C')
    plt.savefig(f'{CIKTI}/baraj_grafikleri/{dosya}_boxplot.png', dpi=150, bbox_inches='tight')
    plt.close()

for baraj in BARAJLAR:
    if baraj not in df_oncesi.columns:
        continue
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(range(len(df_oncesi)), df_oncesi[baraj], color='tomato', linewidth=1)
    axes[0].set_title(f"{baraj} — Ham Veri", fontsize=11)
    axes[0].set_ylabel("Doluluk (%)"); axes[0].set_ylim(0, 110)
    axes[1].plot(range(len(df_b)), df_b[baraj], color='steelblue', linewidth=1)
    axes[1].set_title(f"{baraj} — Temizlenmiş Veri", fontsize=11)
    axes[1].set_ylabel("Doluluk (%)"); axes[1].set_ylim(0, 110)
    plt.suptitle(f"{baraj} — Önce / Sonra", fontsize=13, fontweight='bold')
    plt.tight_layout()
    dosya = baraj.replace(' ', '_').replace('ı','i').replace('İ','I').replace('ç','c').replace('Ç','C')
    plt.savefig(f'{CIKTI}/baraj_grafikleri/{dosya}_once_sonra.png', dpi=150, bbox_inches='tight')
    plt.close()

yillik = df_b.groupby("YIL")[BARAJLAR].mean()
plt.figure(figsize=(13, 6))
for baraj in BARAJLAR:
    kisaltma = baraj.replace(' Barajı','').replace('Alaçatı Kutlu Aktaş','Alaçatı')
    plt.plot(yillik.index, yillik[baraj], marker='o', linewidth=2, label=kisaltma)
plt.title("İzmir Barajları Yıllık Ortalama Doluluk Trendi (2014-2023)", fontsize=13)
plt.xlabel("Yıl"); plt.ylabel("Ortalama Doluluk (%)"); plt.ylim(0, 100)
plt.xticks(yillik.index, rotation=45); plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f'{CIKTI}/baraj_grafikleri/yillik_trend.png', dpi=150, bbox_inches='tight')
plt.close()

plt.figure(figsize=(9, 7))
kor = df_b[BARAJLAR].corr()
mask_triu = np.triu(np.ones_like(kor, dtype=bool), k=1)
sns.heatmap(kor, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            mask=mask_triu, square=True, linewidths=0.5)
plt.title("Barajlar Arası Doluluk Korelasyonu (2014-2023)", fontsize=13)
plt.tight_layout()
plt.savefig(f'{CIKTI}/baraj_grafikleri/korelasyon_matrisi.png', dpi=150, bbox_inches='tight')
plt.close()

df_b.to_csv(f'{CIKTI}/baraj_islenmis.csv', index=False, encoding='utf-8-sig')

try:
    from IPython.display import display, FileLink
    for f in sorted(os.listdir(f'{CIKTI}/baraj_grafikleri')):
        display(FileLink(f'{CIKTI}/baraj_grafikleri/{f}',
                result_html_prefix=f"  📄 {f} → "))
    display(FileLink(f'{CIKTI}/baraj_islenmis.csv',
            result_html_prefix="  📄 baraj_islenmis.csv → "))
except Exception:
    pass