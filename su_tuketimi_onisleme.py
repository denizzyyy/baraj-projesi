import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

CIKTI = '/content/onisleme_cikti'
os.makedirs(f'{CIKTI}/su_grafikleri', exist_ok=True)

df_raw = pd.read_csv('tuketim_ham_veri.csv', low_memory=False, encoding_errors='replace')
df_raw['ORTALAMA_TUKETIM'] = pd.to_numeric(df_raw['ORTALAMA_TUKETIM'], errors='coerce')
df_raw['YIL'] = pd.to_numeric(df_raw['YIL'], errors='coerce').astype('Int64')
df_raw['AY']  = pd.to_numeric(df_raw['AY'],  errors='coerce').astype('Int64')

df_su = df_raw[(df_raw['ABONELIK_TURU'] == 'Su') & (df_raw['YIL'].between(2014, 2023))].copy()

df_aylik = df_su.groupby(['YIL', 'AY'])['ORTALAMA_TUKETIM'].sum().reset_index()

nufus_map = {
    2014: 4113072, 2015: 4168415, 2016: 4223545, 2017: 4279677, 2018: 4320519,
    2019: 4367251, 2020: 4394694, 2021: 4425789, 2022: 4462056, 2023: 4479525
}
df_aylik['NUFUS']     = df_aylik['YIL'].map(nufus_map)
df_aylik['KISI_BASI'] = df_aylik['ORTALAMA_TUKETIM'] / df_aylik['NUFUS']

def aykiri_deger_temizle(group):
    group_2014  = group[group['YIL'] == 2014]
    group_diger = group[group['YIL'] != 2014].copy()
    if len(group_diger) > 1:
        medyan = group_diger['KISI_BASI'].median()
        std    = group_diger['KISI_BASI'].std()
        mask   = ((group_diger['KISI_BASI'] > medyan + 2 * std) |
                  (group_diger['KISI_BASI'] < medyan - 2 * std))
        group_diger.loc[mask, 'KISI_BASI']        = medyan
        group_diger.loc[mask, 'ORTALAMA_TUKETIM'] = (
            group_diger.loc[mask, 'KISI_BASI'] * group_diger.loc[mask, 'NUFUS'])
    return pd.concat([group_2014, group_diger])

df_final = (df_aylik
            .set_index('AY')
            .groupby(level='AY', group_keys=False)
            .apply(aykiri_deger_temizle)
            .reset_index())
df_final = df_final.sort_values(['YIL', 'AY']).reset_index(drop=True)

sns.set_theme(style="whitegrid")
zaman = pd.to_datetime(
    df_final['YIL'].astype(int).astype(str) + '-' +
    df_final['AY'].astype(int).astype(str),
    format='%Y-%m')

plt.figure(figsize=(14, 5))
ham_sirali = df_aylik.sort_values(['YIL', 'AY'])
plt.plot(zaman, ham_sirali['ORTALAMA_TUKETIM'], color='tomato', alpha=0.4, linestyle='--', label='Ham Veri')
plt.plot(zaman, df_final['ORTALAMA_TUKETIM'], color='steelblue', linewidth=2, label='Temizlenmiş Veri')
plt.title('Su Tüketimi Zaman Serisi (2014-2023)', fontsize=13)
plt.ylabel('Toplam Tüketim (m³)')
plt.legend()
plt.tight_layout()
plt.savefig(f'{CIKTI}/su_grafikleri/zaman_serisi.png', dpi=150, bbox_inches='tight')
plt.close()

plt.figure(figsize=(12, 7))
pivot = df_final.pivot(index='AY', columns='YIL', values='ORTALAMA_TUKETIM')
sns.heatmap(pivot / 1e6, annot=True, fmt='.1f', cmap='YlOrRd', cbar_kws={'label': 'Milyon m³'})
plt.title('İzmir Su Tüketimi Isı Haritası (Milyon m³)', fontsize=13)
plt.xlabel('Yıllar')
plt.ylabel('Aylar')
plt.tight_layout()
plt.savefig(f'{CIKTI}/su_grafikleri/isi_haritasi.png', dpi=150, bbox_inches='tight')
plt.close()

plt.figure(figsize=(12, 5))
sns.boxplot(x='AY', y='ORTALAMA_TUKETIM', data=df_final, hue='AY', palette='coolwarm', legend=False)
plt.title('Aylık Su Tüketim Değişkenliği', fontsize=13)
plt.tight_layout()
plt.savefig(f'{CIKTI}/su_grafikleri/boxplot.png', dpi=150, bbox_inches='tight')
plt.close()

yillik_kisi = df_final.groupby('YIL')['KISI_BASI'].mean()
plt.figure(figsize=(11, 5))
plt.bar(yillik_kisi.index, yillik_kisi.values, color='steelblue', alpha=0.8)
for i, (yil, val) in enumerate(yillik_kisi.items()):
    plt.text(yil, val + 0.001, f'{val:.3f}', ha='center', fontsize=9)
plt.title('Yıllık Kişi Başı Aylık Su Tüketimi (m³/kişi)', fontsize=13)
plt.ylabel('m³/kişi/ay')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{CIKTI}/su_grafikleri/kisi_basi_trend.png', dpi=150, bbox_inches='tight')
plt.close()

df_cikti = df_final[['YIL', 'AY', 'ORTALAMA_TUKETIM', 'NUFUS', 'KISI_BASI']].copy()
df_cikti.columns = ['YIL', 'AY', 'SU_TUKETIM', 'NUFUS', 'KISI_BASI']
df_cikti.to_csv(f'{CIKTI}/su_islenmis.csv', index=False, encoding='utf-8-sig')

try:
    from IPython.display import display, FileLink
    for f in sorted(os.listdir(f'{CIKTI}/su_grafikleri')):
        display(FileLink(f'{CIKTI}/su_grafikleri/{f}', result_html_prefix=f"{f} → "))
    display(FileLink(f'{CIKTI}/su_islenmis.csv', result_html_prefix="su_islenmis.csv → "))
except Exception:
    pass