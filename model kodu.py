import copy
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

CIKTI = '/content/model_cikti'
os.makedirs(f'{CIKTI}/baraj_grafikleri', exist_ok=True)
os.makedirs(f'{CIKTI}/karsilastirma', exist_ok=True)
os.makedirs(f'{CIKTI}/sunum', exist_ok=True)


def dosya_bul(isimler):
    olasi_klasorler = [
        '/content/',
        '/content/onisleme_cikti/',
        '/content/model_cikti/',
        './',
    ]
    for klasor in olasi_klasorler:
        if not os.path.exists(klasor):
            continue
        for f in os.listdir(klasor):
            for isim in isimler:
                if f == isim or f.startswith(isim.replace('.csv', '')):
                    return os.path.join(klasor, f)
    return None


baraj_yol = dosya_bul(['baraj_islenmis.csv'])
hava_yol = dosya_bul(['hava_islenmis.csv'])
su_yol = dosya_bul(['su_islenmis.csv'])

if not all([baraj_yol, hava_yol, su_yol]):
    raise FileNotFoundError("İşlenmiş CSV dosyaları bulunamadı.")

df_baraj = pd.read_csv(baraj_yol)
df_hava = pd.read_csv(hava_yol)
df_su = pd.read_csv(su_yol)

df_merge = (
    df_baraj.merge(df_hava, on=['YIL', 'AY'], how='inner').merge(
        df_su[['YIL', 'AY', 'SU_TUKETIM', 'KISI_BASI']]
        if 'SU_TUKETIM' in df_su.columns
        else df_su,
        on=['YIL', 'AY'],
        how='inner',
    )
)
df_merge = df_merge.sort_values(['YIL', 'AY']).reset_index(drop=True)

BARAJLAR_TERCIH = [
    'Alaçatı Kutlu Aktaş Barajı',
    'Balçova Barajı',
    'Gördes Barajı',
    'Tahtalı Barajı',
    'Ürkmez Barajı',
]
BARAJLAR = [b for b in BARAJLAR_TERCIH if b in df_merge.columns]

df_merge.to_csv(f'{CIKTI}/birlesmis_veri.csv', index=False, encoding='utf-8-sig')

TRAIN_SON_YIL = 2021
df_train = df_merge[df_merge['YIL'] <= TRAIN_SON_YIL].copy().reset_index(drop=True)
df_test = df_merge[df_merge['YIL'] > TRAIN_SON_YIL].copy().reset_index(drop=True)

ML_MODELLER = {
    'Linear Reg.': Pipeline(
        [('scaler', StandardScaler()), ('model', LinearRegression())]
    ),
    'Ridge': Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=1.0))]),
    'Random Forest': RandomForestRegressor(
        n_estimators=100, max_depth=10, min_samples_leaf=3, random_state=42
    ),
    'Grad. Boosting': GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=42,
    ),
}
MODEL_ADLARI = list(ML_MODELLER.keys()) + ['Naive Baseline']
ML_ONLY = list(ML_MODELLER.keys())

OZELLIKLER = [
    'SICAKLIK_ORT',
    'YAGIS',
    'RADYASYON',
    'ONCEKI_YAGIS',
    'YAGIS_3AY_LOG',
    'MEVSIM',
    'SU_TUKETIM',
    'KISI_BASI',
]

sonuclar = {}
tahminler = {}
TSCV = TimeSeriesSplit(n_splits=5)
OVERFIT_ESIGI = 0.15

for baraj in BARAJLAR:
    sonuclar[baraj] = {}
    tahminler[baraj] = {}

    prev_col = f"{baraj}_PREV"
    ozellikler = OZELLIKLER + [prev_col]

    if baraj == 'Gördes Barajı':
        gordes_ekstra = [
            b + '_PREV'
            for b in BARAJLAR
            if b != 'Gördes Barajı' and b + '_PREV' in df_merge.columns
        ]
        ozellikler = ozellikler + gordes_ekstra

    X_train = df_train[ozellikler].values
    y_train = df_train[baraj].values
    X_test = df_test[ozellikler].values
    y_test = df_test[baraj].values

    for model_adi, model in ML_MODELLER.items():
        m = copy.deepcopy(model)
        m.fit(X_train, y_train)
        y_pred_train = m.predict(X_train)
        y_pred_test = m.predict(X_test)

        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)

        cv_skorlar = cross_val_score(
            copy.deepcopy(model), X_train, y_train, cv=TSCV, scoring='r2'
        )
        cv_ort = cv_skorlar.mean()
        cv_std = cv_skorlar.std()

        gap = r2_train - r2_test

        sonuclar[baraj][model_adi] = {
            'R2_test': r2_test,
            'RMSE_test': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'MAE_test': mean_absolute_error(y_test, y_pred_test),
            'R2_train': r2_train,
            'CV_ort': cv_ort,
            'CV_std': cv_std,
            'Gap': gap,
        }
        tahminler[baraj][model_adi] = {'train': y_pred_train, 'test': y_pred_test}

    y_naive_train = df_train[prev_col].values
    y_naive_test = df_test[prev_col].values
    r2_naive_train = r2_score(y_train, y_naive_train)
    r2_naive_test = r2_score(y_test, y_naive_test)

    sonuclar[baraj]['Naive Baseline'] = {
        'R2_test': r2_naive_test,
        'RMSE_test': np.sqrt(mean_squared_error(y_test, y_naive_test)),
        'MAE_test': mean_absolute_error(y_test, y_naive_test),
        'R2_train': r2_naive_train,
        'CV_ort': float('nan'),
        'CV_std': float('nan'),
        'Gap': r2_naive_train - r2_naive_test,
    }
    tahminler[baraj]['Naive Baseline'] = {
        'train': y_naive_train,
        'test': y_naive_test,
    }

sns.set_theme(style="whitegrid")
RENKLER = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#95a5a6']

zaman_train = pd.to_datetime(
    df_train['YIL'].astype(int).astype(str)
    + '-'
    + df_train['AY'].astype(int).astype(str),
    format='%Y-%m',
)
zaman_test = pd.to_datetime(
    df_test['YIL'].astype(int).astype(str)
    + '-'
    + df_test['AY'].astype(int).astype(str),
    format='%Y-%m',
)

for baraj in BARAJLAR:
    y_train_plot = df_train[baraj].values
    y_test_plot = df_test[baraj].values
    ML_RENKLER = RENKLER[:4]

    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    fig.suptitle(
        f"{baraj} — Makine Öğrenmesi Modelleri (Test: 2022-2023)",
        fontsize=14,
        fontweight='bold',
    )
    axes = axes.flatten()

    for i, (model_adi, renk) in enumerate(zip(ML_ONLY, ML_RENKLER)):
        ax = axes[i]
        y_pred = tahminler[baraj][model_adi]['test']
        r2 = sonuclar[baraj][model_adi]['R2_test']
        rmse = sonuclar[baraj][model_adi]['RMSE_test']
        naive_r2 = sonuclar[baraj]['Naive Baseline']['R2_test']

        ax.plot(
            zaman_train,
            y_train_plot,
            color='lightgray',
            linewidth=1,
            label='Eğitim Verisi',
            zorder=1,
        )
        ax.plot(
            zaman_test,
            y_test_plot,
            color='black',
            linewidth=2,
            label='Gerçek (Test)',
            zorder=2,
        )
        ax.plot(
            zaman_test,
            y_pred,
            color=renk,
            linewidth=2,
            linestyle='--',
            label='Tahmin',
            zorder=3,
        )
        ax.axvline(
            x=zaman_test.iloc[0], color='gray', linestyle=':', linewidth=1.5
        )

        fark = r2 - naive_r2
        karsilastirma = f"  (Naive: {naive_r2:.3f}, Δ={fark:+.3f})"
        ax.set_title(
            f"{model_adi}  |  Test R²={r2:.3f}  RMSE={rmse:.2f}{karsilastirma}",
            fontsize=10,
            fontweight='bold',
        )
        ax.set_ylabel("Doluluk (%)")
        ax.set_ylim(0, 105)
        ax.legend(fontsize=8, loc='upper left')
        ax.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    dosya = (
        baraj.replace(' ', '_')
        .replace('ı', 'i')
        .replace('İ', 'I')
        .replace('ç', 'c')
        .replace('Ç', 'C')
    )
    plt.savefig(
        f'{CIKTI}/baraj_grafikleri/{dosya}_modeller.png',
        dpi=150,
        bbox_inches='tight',
    )
    plt.close()

r2_df = pd.DataFrame(
    {b: {m: sonuclar[b][m]['R2_test'] for m in MODEL_ADLARI} for b in BARAJLAR}
).T
rmse_df = pd.DataFrame(
    {b: {m: sonuclar[b][m]['RMSE_test'] for m in MODEL_ADLARI} for b in BARAJLAR}
).T

plt.figure(figsize=(13, max(5, len(BARAJLAR) * 1.5)))
sns.heatmap(
    r2_df,
    annot=True,
    fmt='.3f',
    cmap='RdYlGn',
    vmin=0,
    vmax=1,
    linewidths=0.5,
    cbar_kws={'label': 'Test R²'},
)
plt.title(
    'Test R² Karşılaştırması — Baraj × Model', fontsize=14, fontweight='bold'
)
plt.tight_layout()
plt.savefig(
    f'{CIKTI}/karsilastirma/r2_heatmap.png', dpi=150, bbox_inches='tight'
)
plt.close()

plt.figure(figsize=(13, max(5, len(BARAJLAR) * 1.5)))
sns.heatmap(
    rmse_df,
    annot=True,
    fmt='.2f',
    cmap='RdYlGn_r',
    linewidths=0.5,
    cbar_kws={'label': 'RMSE (%)'},
)
plt.title(
    'RMSE Karşılaştırması — Baraj × Model', fontsize=14, fontweight='bold'
)
plt.tight_layout()
plt.savefig(
    f'{CIKTI}/karsilastirma/rmse_heatmap.png', dpi=150, bbox_inches='tight'
)
plt.close()

ml_modeller_adlari = list(ML_MODELLER.keys())
cv_df = pd.DataFrame(
    {b: {m: sonuclar[b][m]['CV_ort'] for m in ml_modeller_adlari} for b in BARAJLAR}
).T
plt.figure(figsize=(11, max(5, len(BARAJLAR) * 1.5)))
sns.heatmap(
    cv_df,
    annot=True,
    fmt='.3f',
    cmap='RdYlGn',
    vmin=0,
    vmax=1,
    linewidths=0.5,
    cbar_kws={'label': 'CV R² (Ortalama)'},
)
plt.title(
    'CV R² Karşılaştırması — Baraj × Model\n(TimeSeriesSplit, 5 Katlı)',
    fontsize=13,
    fontweight='bold',
)
plt.tight_layout()
plt.savefig(
    f'{CIKTI}/karsilastirma/cv_r2_heatmap.png', dpi=150, bbox_inches='tight'
)
plt.close()

fig, axes = plt.subplots(1, len(BARAJLAR), figsize=(20, 6))
fig.suptitle(
    'Overfit Analizi — Train vs Test R² (ML Modeller)',
    fontsize=13,
    fontweight='bold',
)
x = np.arange(len(ml_modeller_adlari))
genislik = 0.35

for i, baraj in enumerate(BARAJLAR):
    ax = axes[i]
    train_r2 = [sonuclar[baraj][m]['R2_train'] for m in ml_modeller_adlari]
    test_r2 = [sonuclar[baraj][m]['R2_test'] for m in ml_modeller_adlari]
    gaps = [sonuclar[baraj][m]['Gap'] for m in ml_modeller_adlari]

    ax.bar(
        x - genislik / 2,
        train_r2,
        genislik,
        label='Train R²',
        color='#3498db',
        alpha=0.8,
    )
    ax.bar(
        x + genislik / 2,
        test_r2,
        genislik,
        label='Test R²',
        color='#e74c3c',
        alpha=0.8,
    )

    for j, gap in enumerate(gaps):
        renk = 'red' if gap > OVERFIT_ESIGI else 'green'
        ax.text(
            j,
            max(train_r2[j], test_r2[j]) + 0.02,
            f'Δ{gap:.2f}',
            ha='center',
            fontsize=7,
            color=renk,
            fontweight='bold',
        )

    kisaltma = baraj.replace(' Barajı', '').replace('Alaçatı Kutlu Aktaş', 'Alaçatı')
    ax.set_title(kisaltma, fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(
        [m.replace('. ', '.\n') for m in ml_modeller_adlari], fontsize=7
    )
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('R²' if i == 0 else '')
    ax.axhline(y=0.8, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    if i == 0:
        ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(
    f'{CIKTI}/karsilastirma/overfit_analizi.png', dpi=150, bbox_inches='tight'
)
plt.close()

en_iyi = {}
for baraj in BARAJLAR:
    en_iyi_m = max(
        ML_ONLY,
        key=lambda m: sonuclar[baraj][m]['R2_test']
        if not np.isnan(sonuclar[baraj][m]['R2_test'])
        else -999,
    )
    naive_r2 = sonuclar[baraj]['Naive Baseline']['R2_test']
    ml_r2 = sonuclar[baraj][en_iyi_m]['R2_test']
    naive_gecildi = ml_r2 > naive_r2

    en_iyi[baraj] = {
        'Model': en_iyi_m,
        'R2': ml_r2,
        'RMSE': sonuclar[baraj][en_iyi_m]['RMSE_test'],
        'MAE': sonuclar[baraj][en_iyi_m]['MAE_test'],
        'Naive R2': naive_r2,
        'Naive Geçti': naive_gecildi,
    }

df_en_iyi = pd.DataFrame(en_iyi).T.reset_index()
df_en_iyi.columns = [
    'Baraj',
    'Model',
    'R2',
    'RMSE',
    'MAE',
    'Naive R2',
    'Naive Geçti',
]
df_en_iyi['R2'] = df_en_iyi['R2'].astype(float)
df_en_iyi['RMSE'] = df_en_iyi['RMSE'].astype(float)
df_en_iyi['Naive R2'] = df_en_iyi['Naive R2'].astype(float)

model_renk_map = dict(zip(MODEL_ADLARI, RENKLER))
fig, ax = plt.subplots(figsize=(14, max(5, len(BARAJLAR) * 1.2)))
bar_renkler = [model_renk_map[m] for m in df_en_iyi['Model']]
bars = ax.barh(
    df_en_iyi['Baraj'], df_en_iyi['R2'], color=bar_renkler, edgecolor='white', height=0.6
)
for bar, row in zip(bars, df_en_iyi.itertuples()):
    uyari = "" if row._7 else "   ⚠️ Naive altında"
    ax.text(
        bar.get_width() + 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"R²={row.R2:.3f}  [{row.Model}]{uyari}",
        va='center',
        fontsize=10,
        fontweight='bold',
    )
ax.set_xlim(0, 1.55)
ax.set_xlabel('Test R²', fontsize=12)
ax.set_title(
    'Her Baraj İçin En İyi Makine Öğrenmesi Modeli\n(Naive Baseline kıyas referansı; en iyi model seçimine dahil değil)',
    fontsize=12,
    fontweight='bold',
)
legend_elems = [Patch(facecolor=model_renk_map[m], label=m) for m in ML_ONLY]
ax.legend(handles=legend_elems, loc='lower right', fontsize=9, title='ML Modelleri')
plt.tight_layout()
plt.savefig(
    f'{CIKTI}/karsilastirma/en_iyi_model_per_baraj.png',
    dpi=150,
    bbox_inches='tight',
)
plt.close()

ort_r2 = r2_df.mean(axis=0).sort_values(ascending=False)
plt.figure(figsize=(11, 6))
sns.barplot(
    x=ort_r2.index, y=ort_r2.values, hue=ort_r2.index, palette=RENKLER, legend=False
)
for i, v in enumerate(ort_r2.values):
    plt.text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold', fontsize=11)
plt.ylim(0, 1.1)
plt.title(
    'Modellerin Ortalama Test R² Sıralaması (5 Baraj)',
    fontsize=13,
    fontweight='bold',
)
plt.ylabel('Ortalama Test R²')
plt.tight_layout()
plt.savefig(
    f'{CIKTI}/karsilastirma/model_genel_siralama.png',
    dpi=150,
    bbox_inches='tight',
)
plt.close()

df_en_iyi.to_csv(
    f'{CIKTI}/karsilastirma/ozet_rapor.csv', index=False, encoding='utf-8-sig'
)
r2_df.to_csv(f'{CIKTI}/karsilastirma/r2_test_tablosu.csv', encoding='utf-8-sig')
rmse_df.to_csv(f'{CIKTI}/karsilastirma/rmse_tablosu.csv', encoding='utf-8-sig')
cv_df.to_csv(f'{CIKTI}/karsilastirma/cv_r2_tablosu.csv', encoding='utf-8-sig')

fi_modeller = {
    'Random Forest': ML_MODELLER['Random Forest'],
    'Grad. Boosting': ML_MODELLER['Grad. Boosting'],
}

for model_adi, model in fi_modeller.items():
    fig, axes = plt.subplots(1, len(BARAJLAR), figsize=(22, 6))
    fig.suptitle(
        f'Değişken Önemi — {model_adi}', fontsize=14, fontweight='bold'
    )

    for i, baraj in enumerate(BARAJLAR):
        prev_col = f"{baraj}_PREV"
        ozellikler = OZELLIKLER + [prev_col]
        if baraj == 'Gördes Barajı':
            gordes_ekstra = [
                b + '_PREV'
                for b in BARAJLAR
                if b != 'Gördes Barajı' and b + '_PREV' in df_merge.columns
            ]
            ozellikler = ozellikler + gordes_ekstra

        X_tr = df_train[ozellikler].values
        y_tr = df_train[baraj].values
        m = copy.deepcopy(model)
        m.fit(X_tr, y_tr)

        importances = pd.Series(m.feature_importances_, index=ozellikler).sort_values(
            ascending=True
        )
        kisaltma = {
            'SICAKLIK_ORT': 'Sıcaklık',
            'YAGIS': 'Yağış',
            'RADYASYON': 'Radyasyon',
            'ONCEKI_YAGIS': 'Önceki Yağış',
            'YAGIS_3AY_LOG': '3 Ay Yağış',
            'MEVSIM': 'Mevsim',
            'SU_TUKETIM': 'Su Tüketim',
            'KISI_BASI': 'Kişi Başı',
        }
        importances.index = [
            kisaltma.get(
                c,
                c.replace('_PREV', ' Prev')
                .replace('Alaçatı Kutlu Aktaş Barajı', 'Alaçatı')
                .replace('Balçova Barajı', 'Balçova')
                .replace('Tahtalı Barajı', 'Tahtalı')
                .replace('Ürkmez Barajı', 'Ürkmez'),
            )
            for c in importances.index
        ]

        renkler_fi = [
            '#e74c3c' if v == importances.max() else '#3498db'
            for v in importances.values
        ]
        axes[i].barh(importances.index, importances.values, color=renkler_fi)
        kisaltma_b = (
            baraj.replace(' Barajı', '').replace('Alaçatı Kutlu Aktaş', 'Alaçatı')
        )
        axes[i].set_title(kisaltma_b, fontsize=10, fontweight='bold')
        axes[i].set_xlabel('Önem')
        axes[i].tick_params(axis='y', labelsize=8)

    plt.tight_layout()
    dosya_m = model_adi.replace('. ', '_').replace(' ', '_')
    plt.savefig(
        f'{CIKTI}/sunum/feature_importance_{dosya_m}.png',
        dpi=150,
        bbox_inches='tight',
    )
    plt.close()

SCATTER_RENKLER = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

for baraj in BARAJLAR:
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(
        f'{baraj} — Gerçek vs Tahmin (Test Seti)', fontsize=13, fontweight='bold'
    )
    y_test = df_test[baraj].values

    for i, (model_adi, renk) in enumerate(zip(ml_modeller_adlari, SCATTER_RENKLER)):
        ax = axes[i]
        y_pred = tahminler[baraj][model_adi]['test']
        r2 = sonuclar[baraj][model_adi]['R2_test']
        rmse = sonuclar[baraj][model_adi]['RMSE_test']

        ax.scatter(
            y_test,
            y_pred,
            color=renk,
            alpha=0.8,
            s=60,
            edgecolors='white',
            linewidth=0.5,
        )
        lim_min = min(y_test.min(), y_pred.min()) - 5
        lim_max = max(y_test.max(), y_pred.max()) + 5
        ax.plot(
            [lim_min, lim_max], [lim_min, lim_max], 'k--', linewidth=1.5, label='Mükemmel'
        )
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.set_title(
            f"{model_adi}\nR²={r2:.3f}  RMSE={rmse:.2f}",
            fontsize=10,
            fontweight='bold',
        )
        ax.set_xlabel('Gerçek Doluluk (%)')
        ax.set_ylabel('Tahmin Doluluk (%)')
        ax.legend(fontsize=8)

    plt.tight_layout()
    dosya_b = (
        baraj.replace(' ', '_')
        .replace('ı', 'i')
        .replace('İ', 'I')
        .replace('ç', 'c')
        .replace('Ç', 'C')
    )
    plt.savefig(
        f'{CIKTI}/sunum/scatter_{dosya_b}.png', dpi=150, bbox_inches='tight'
    )
    plt.close()

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(
    '2014 Kuraklığının Baraj Doluluk Oranlarına Etkisi\n(Gri bölge: 2014 tarihi kuraklık yılı)',
    fontsize=13,
    fontweight='bold',
)
axes = axes.flatten()

zaman_tum = pd.to_datetime(
    df_merge['YIL'].astype(int).astype(str)
    + '-'
    + df_merge['AY'].astype(int).astype(str),
    format='%Y-%m',
)

for i, baraj in enumerate(BARAJLAR):
    ax = axes[i]
    degerler = df_merge[baraj].values
    ort = np.mean(degerler)

    ax.fill_between(
        zaman_tum,
        0,
        100,
        where=df_merge['YIL'] == 2014,
        color='#e74c3c',
        alpha=0.15,
        label='2014 Kuraklık',
    )
    ax.plot(zaman_tum, degerler, color='#2980b9', linewidth=1.8)
    ax.axhline(
        y=ort, color='gray', linestyle='--', linewidth=1, alpha=0.7, label=f'Ort: %{ort:.1f}'
    )

    min_idx = np.argmin(degerler)
    ax.annotate(
        f'Min: %{degerler[min_idx]:.1f}',
        xy=(zaman_tum.iloc[min_idx], degerler[min_idx]),
        xytext=(10, 15),
        textcoords='offset points',
        fontsize=8,
        color='red',
        arrowprops=dict(arrowstyle='->', color='red', lw=1.2),
    )

    kisaltma_b = baraj.replace(' Barajı', '').replace('Alaçatı Kutlu Aktaş', 'Alaçatı')
    ax.set_title(kisaltma_b, fontsize=11, fontweight='bold')
    ax.set_ylabel('Doluluk (%)')
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7)
    ax.tick_params(axis='x', rotation=30, labelsize=7)

axes[5].set_visible(False)
plt.tight_layout()
plt.savefig(
    f'{CIKTI}/sunum/kurakhk_2014_analizi.png', dpi=150, bbox_inches='tight'
)
plt.close()

if 'Gördes Barajı' in BARAJLAR:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Gördes Barajı — Neden Farklı?', fontsize=13, fontweight='bold')

    ax = axes[0]
    for baraj in BARAJLAR:
        renk = '#e74c3c' if baraj == 'Gördes Barajı' else '#95a5a6'
        lw = 2.5 if baraj == 'Gördes Barajı' else 1.0
        alpha = 1.0 if baraj == 'Gördes Barajı' else 0.4
        kisaltma_b = (
            baraj.replace(' Barajı', '').replace('Alaçatı Kutlu Aktaş', 'Alaçatı')
        )
        ax.plot(
            zaman_tum,
            df_merge[baraj].values,
            color=renk,
            linewidth=lw,
            alpha=alpha,
            label=kisaltma_b,
        )
    ax.set_title('Doluluk Karşılaştırması (2014-2023)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Doluluk (%)')
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    ax.tick_params(axis='x', rotation=30)

    ax2 = axes[1]
    test_r2_gordes = [sonuclar['Gördes Barajı'][m]['R2_test'] for m in MODEL_ADLARI]
    test_r2_diger = [
        np.mean(
            [
                sonuclar[b][m]['R2_test']
                for b in BARAJLAR
                if b != 'Gördes Barajı'
            ]
        )
        for m in MODEL_ADLARI
    ]
    x = np.arange(len(MODEL_ADLARI))
    ax2.bar(
        x - 0.2,
        test_r2_diger,
        0.4,
        label='Diğer 4 Baraj (Ort.)',
        color='#3498db',
        alpha=0.8,
    )
    ax2.bar(
        x + 0.2, test_r2_gordes, 0.4, label='Gördes Barajı', color='#e74c3c', alpha=0.8
    )
    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(MODEL_ADLARI, rotation=15, fontsize=9)
    ax2.set_ylabel('Test R²')
    ax2.set_title(
        'Model Performansı: Gördes vs Diğer Barajlar', fontsize=11, fontweight='bold'
    )
    ax2.legend(fontsize=9)
    ax2.set_ylim(-0.3, 1.1)
    ax2.axhline(y=0.8, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    plt.tight_layout()
    plt.savefig(
        f'{CIKTI}/sunum/gordes_ozel_analiz.png', dpi=150, bbox_inches='tight'
    )
    plt.close()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(
    'Naive Baseline Analizi — Modeller Gerçekten Öğrendi mi?',
    fontsize=13,
    fontweight='bold',
)

ax = axes[0]
naive_r2 = [sonuclar[b]['Naive Baseline']['R2_test'] for b in BARAJLAR]
en_iyi_r2 = [
    max(
        sonuclar[b][m]['R2_test']
        for m in MODEL_ADLARI
        if m != 'Naive Baseline'
    )
    for b in BARAJLAR
]
kisaltma_barajlar = [
    b.replace(' Barajı', '').replace('Alaçatı Kutlu Aktaş', 'Alaçatı') for b in BARAJLAR
]
x = np.arange(len(BARAJLAR))
ax.bar(x - 0.2, naive_r2, 0.4, label='Naive Baseline', color='#95a5a6', alpha=0.9)
ax.bar(x + 0.2, en_iyi_r2, 0.4, label='En İyi Model', color='#2ecc71', alpha=0.9)

for j, (n, m_) in enumerate(zip(naive_r2, en_iyi_r2)):
    fark = m_ - n
    renk = '#27ae60' if fark > 0 else '#e74c3c'
    ax.text(
        j,
        max(n, m_) + 0.02,
        f'{fark:+.3f}',
        ha='center',
        fontsize=9,
        color=renk,
        fontweight='bold',
    )

ax.set_xticks(x)
ax.set_xticklabels(kisaltma_barajlar, rotation=15, fontsize=9)
ax.set_ylabel('Test R²')
ax.set_ylim(0, 1.15)
ax.set_title('Naive vs En İyi Model (Test R²)', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.axhline(y=0.8, color='gray', linestyle='--', linewidth=1, alpha=0.5)

ax2 = axes[1]
ax2.axis('off')
aciklama = (
    "💡   NAIVE BASELINE NEDİR?\n\n"
    "En basit tahmin yöntemi:\n"
    "\"Bu ay ne olur?\"\n"
    "→ Geçen ay ne idiyse o!\n\n"
    "─────────────────────────────\n\n"
    "NEDEN ÖNEMLİ?\n\n"
    "Eğer modelimiz bu kadar basit\n"
    "bir yaklaşımı geçemiyorsa,\n"
    "gerçek bir şey öğrenmemiş\n"
    "demektir.\n\n"
    "─────────────────────────────\n\n"
    "BULGUMUZ:\n\n"
    "✅   Alaçatı: +0.077\n"
    "✅   Balçova: +0.168\n"
    "✅   Tahtalı: +0.005\n"
    "✅   Ürkmez:  +0.040\n"
    "❌   Gördes:  -0.153\n\n"
    "4/5 barajda modeller\n"
    "naive'i geçiyor."
)
ax2.text(
    0.05,
    0.95,
    aciklama,
    transform=ax2.transAxes,
    fontsize=10,
    verticalalignment='top',
    fontfamily='monospace',
    bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.8),
)

plt.tight_layout()
plt.savefig(
    f'{CIKTI}/sunum/naive_baseline_analiz.png', dpi=150, bbox_inches='tight'
)
plt.close()

try:
    from IPython.display import FileLink, display

    for klasor in ['baraj_grafikleri', 'karsilastirma', 'sunum']:
        yol = f'{CIKTI}/{klasor}'
        for f in sorted(os.listdir(yol)):
            display(FileLink(f'{yol}/{f}', result_html_prefix=f"  📄 {klasor}/{f} → "))
except Exception:
    pass