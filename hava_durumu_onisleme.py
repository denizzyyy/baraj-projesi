import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

CIKTI = '/content/onisleme_cikti'
os.makedirs(f'{CIKTI}/hava_grafikleri', exist_ok=True)

df = pd.read_csv('hava_ham_veri.csv', skiprows=3)
df.columns = ['TARIH', 'SICAKLIK_ORT', 'SICAKLIK_MAX', 'BUHARLASMA', 'YAGIS', 'NEM_ORT', 'RADYASYON']
df['TARIH'] = pd.to_datetime(df['TARIH'])
df = df[df['TARIH'] < '2024-01-01'].copy()
df['YIL'] = df['TARIH'].dt.year
df['AY']  = df['TARIH'].dt.month

DEGISKENLER = ['SICAKLIK_ORT', 'SICAKLIK_MAX', 'BUHARLASMA', 'NEM_ORT', 'RADYASYON']

for col in DEGISKENLER:
    mean, std = df[col].mean(), df[col].std()
    df[col] = df[col].clip(lower=mean - 3*std, upper=mean + 3*std)

df['YAGIS']      = df['YAGIS'].clip(lower=0)
df['BUHARLASMA'] = df['BUHARLASMA'].clip(lower=0)
df['RADYASYON']  = df['RADYASYON'].clip(lower=0)
df['NEM_ORT']    = df['NEM_ORT'].clip(lower=0, upper=100)

df_aylik = df.groupby(['YIL', 'AY']).agg(
    SICAKLIK_ORT=('SICAKLIK_ORT', 'mean'),
    SICAKLIK_MAX=('SICAKLIK_MAX', 'mean'),
    BUHARLASMA  =('BUHARLASMA',   'sum'),
    YAGIS       =('YAGIS',        'sum'),
    NEM_ORT     =('NEM_ORT',      'mean'),
    RADYASYON   =('RADYASYON',    'sum')
).reset_index().round(2)

hata_max = df_aylik['SICAKLIK_MAX'] < df_aylik['SICAKLIK_ORT']
df_aylik.loc[hata_max, 'SICAKLIK_MAX'] = df_aylik.loc[hata_max, 'SICAKLIK_ORT'] + 1.5

yuksek_nem = df_aylik['NEM_ORT'] > 90
df_aylik.loc[yuksek_nem, 'BUHARLASMA'] = df_aylik.loc[yuksek_nem, 'BUHARLASMA'].clip(
    upper=df_aylik['BUHARLASMA'].quantile(0.25))

mask_r = ((df_aylik['RADYASYON'] < df_aylik['RADYASYON'].quantile(0.10)) &
          (df_aylik['BUHARLASMA'] > df_aylik['BUHARLASMA'].mean()))
df_aylik.loc[mask_r, 'BUHARLASMA'] = df_aylik['BUHARLASMA'].mean()

df_aylik['ONCEKI_YAGIS']  = np.log1p(df_aylik['YAGIS'].shift(1).fillna(0))
df_aylik['YAGIS_3AY_LOG'] = np.log1p(
    df_aylik['YAGIS'].shift(1).fillna(0) +
    df_aylik['YAGIS'].shift(2).fillna(0) +
    df_aylik['YAGIS'].shift(3).fillna(0)
)

assert df_aylik.isnull().sum().sum() == 0
assert (df_aylik['SICAKLIK_MAX'] >= df_aylik['SICAKLIK_ORT']).all()
assert (df_aylik['BUHARLASMA'] >= 0).all()
assert (df_aylik['RADYASYON']  >= 0).all()
assert (df_aylik['YAGIS']      >= 0).all()
assert (df_aylik['NEM_ORT'].between(0, 100)).all()

tum_adaylar = ['SICAKLIK_ORT', 'SICAKLIK_MAX', 'BUHARLASMA', 'YAGIS',
                'NEM_ORT', 'RADYASYON', 'ONCEKI_YAGIS', 'YAGIS_3AY_LOG']

plt.figure(figsize=(11, 9))
sns.heatmap(df_aylik[tum_adaylar].corr(), annot=True, fmt='.2f',
            cmap='RdBu_r', center=0)
plt.title('Hava Durumu Değişkenleri Korelasyon Matrisi', fontsize=13)
plt.tight_layout()
plt.savefig(f'{CIKTI}/hava_grafikleri/korelasyon_matrisi.png', dpi=150, bbox_inches='tight')
plt.close()

zaman = pd.to_datetime(
    df_aylik['YIL'].astype(str) + '-' + df_aylik['AY'].astype(str), format='%Y-%m')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Hava Durumu Genel Bakış (2014-2023)', fontsize=13, fontweight='bold')

axes[0].plot(zaman, df_aylik['SICAKLIK_ORT'], color='tomato', linewidth=1.5)
axes[0].set_title('Aylık Ortalama Sıcaklık')
axes[0].set_ylabel('°C'); axes[0].tick_params(axis='x', rotation=45); axes[0].grid(alpha=0.3)

sns.boxplot(x='AY', y='YAGIS', data=df_aylik, hue='AY',
            palette='Blues', legend=False, ax=axes[1])
axes[1].set_title('Aylık Yağış Dağılımı')
axes[1].set_ylabel('mm'); axes[1].grid(axis='y', alpha=0.3)

pivot_s = df_aylik.pivot(index='AY', columns='YIL', values='SICAKLIK_ORT')
sns.heatmap(pivot_s, annot=True, fmt='.0f', cmap='YlOrRd', ax=axes[2])
axes[2].set_title('Sıcaklık Isı Haritası (°C)')

plt.tight_layout()
plt.savefig(f'{CIKTI}/hava_grafikleri/genel_bakis.png', dpi=150, bbox_inches='tight')
plt.close()

tut = ['SICAKLIK_ORT', 'YAGIS', 'RADYASYON', 'ONCEKI_YAGIS', 'YAGIS_3AY_LOG']
df_model = df_aylik[['YIL', 'AY'] + tut].copy()

plt.figure(figsize=(8, 6))
sns.heatmap(df_model[tut].corr(), annot=True, fmt='.2f', cmap='YlGnBu')
plt.title('Final Model Değişkenleri Korelasyon Matrisi', fontsize=13)
plt.tight_layout()
plt.savefig(f'{CIKTI}/hava_grafikleri/final_korelasyon.png', dpi=150, bbox_inches='tight')
plt.close()

df_model.to_csv(f'{CIKTI}/hava_islenmis.csv', index=False, encoding='utf-8-sig')

try:
    from IPython.display import display, FileLink
    for f in sorted(os.listdir(f'{CIKTI}/hava_grafikleri')):
        display(FileLink(f'{CIKTI}/hava_grafikleri/{f}',
                result_html_prefix=f"{f} → "))
    display(FileLink(f'{CIKTI}/hava_islenmis.csv',
            result_html_prefix="hava_islenmis.csv → "))
except Exception:
    pass