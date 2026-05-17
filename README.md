# baraj-projesi
# İzmir Baraj Doluluk Analizi ve Tahmini

Gerçek İZSU altyapı verileri kullanılarak İzmir barajlarının doluluk oranlarını tahmin eden uçtan uca bir makine öğrenmesi projesi.

## 📌 Proje Hakkında

Bu proje; baraj doluluk verilerini, hava durumu verilerini ve su tüketim verilerini birleştirerek zaman serisi tahmini yapmayı amaçlamaktadır. Tahmin sistemi, su kaynağı yönetimi ve altyapı planlaması gibi kritik kararları desteklemek üzere tasarlanmıştır.

## 🗂️ Proje Yapısı
- `baraj_onisleme.py` — baraj verisi ön işleme
- `hava_durumu_onisleme.py` — hava durumu verisi ön işleme  
- `su_tuketimi_onisleme.py` — su tüketimi verisi ön işleme
- `veri birleştirme kodu.py` — veri kaynaklarını birleştirme
- `model kodu.py` — model eğitimi ve değerlendirme
