# Armut Association Rule Learning (ARL) Recommender System

Türkiye'nin en büyük online hizmet platformu Armut'ta müşterilerin aldıkları hizmetler arasındaki ilişkileri analiz ederek, **Association Rule Learning (Birliktelik Kuralları)** kullanarak intelligent hizmet tavsiye sistemi oluşturan proje.

## Proje Tanımı

Armut, hizmet verenler ve hizmet alanları bir platform üzerinde buluşturmaktadır. Temizlik, tadilat, nakliyat gibi çeşitli hizmetlerin sunulduğu bu platformda, müşterilerin hangi hizmetleri bir arada aldığı, bu hizmetler arasında güçlü ilişkiler kurarak personalized tavsiyeler sunmaya olanak sağlar.

Bu proje, **Apriori Algoritması** ve **Association Rules** kullanarak, müşteri satın alma davranışlarını analiz etmekte ve "Eğer bir müşteri X hizmetini aldıysa, Y hizmetini de alması olasıdır" gibi aksiyon alınabilir öneriler üretmektedir.

## Veri Seti

**Kaynak:** Armut hizmet satın alma verisi

### Veri Seti Özellikleri

- **UserId:** Müşteri numarası
- **ServiceId:** Hizmeti tanımlayan ID (kategoriye bağlı olarak değişebilir)
  - Örn: CategoryId=9, ServiceId=4 → Petek temizliği
  - Aynı ServiceId farklı kategoriler altında farklı hizmetleri temsil edebilir
- **CategoryId:** Hizmet kategorisinin ID'si (Temizlik, nakliyat, tadilat vb.)
- **CreateDate:** Hizmetin satın alındığı tarih ve saat

**Boyut:** 162.523 satır x 4 kolon

## Metodoloji

### GÖREV 1: Veriyi Hazırlama

#### Adım 1: Veri Setini Yükleme
```python
df = pd.read_csv("armut_data.csv")
# Veri seti hakkında temel bilgi alma
```

#### Adım 2: Hizmet Tanımı Oluşturma
Aynı ServiceId farklı kategoriler altında farklı hizmetleri temsil edebildiği için, ServiceId ve CategoryId'yi birleştirerek unik hizmetleri tanımlarız:

```python
df["Hizmet"] = df["CategoryId"].astype(str) + "_" + df["ServiceId"].astype(str)
# Örn: 9_4 = CategoryId 9'da ServiceId 4 olan hizmet
```

**Hizmet Tanımının Önemi:**
- Her hizmetin benzersiz tanımlanması
- Hizmetler arasındaki ilişkileri doğru şekilde kurabilmek
- Çapraz kategoril hizmetlerin tespit edilmesi

#### Adım 3: Sepet Tanımı Oluşturma
Association Rule Learning'in çalışabilmesi için "sepet" kavramı tanımlanması gerekir. Burada **sepet = her müşterinin aylık aldığı hizmetler** olarak tanımlanır:

```python
# Tarih sütununu datetime'a dönüştür
df["CreateDate"] = pd.to_datetime(df["CreateDate"])

# Sadece yıl ve ay bilgisini tut
df["NEW_DATE"] = df["CreateDate"].dt.strftime("%Y-%m")

# Sepet ID'sini oluştur (UserId_Ay)
df["SepetID"] = df["UserId"].astype(str) + "_" + df["NEW_DATE"]
# Örn: 7256_2017-08 = 7256 numaralı müşterinin 2017'in ağustos ayında aldığı hizmetler
```

**Sepet Tanımının Mantığı:**
- Müşteri 7256, Ağustos 2017'de 9_4 ve 46_4 hizmetlerini aldı → 1 sepet
- Aynı müşteri, Ekim 2017'de 9_4 ve 38_4 hizmetlerini aldı → Başka bir sepet
- Bu şekilde, birbirinden bağımsız alışveriş seansları oluşturulur

### GÖREV 2: Birliktelik Kuralları Üretelim

#### Adım 1: Pivot Tablo Oluşturma
Association Rule Learning'in çalışabilmesi için verinin aşağıdaki format olması gerekir:

```
Hizmet    0_8  10_9  11_11  12_7  13_11  14_7  ...
SepetID
7256_2017-08   0    0     0     0      0     0   ...
7256_2017-10   1    0     0     0      0     1   ...
...
```

Pivot table oluşturma:
```python
invoice_product_df = df.groupby(['SepetID', 'Hizmet'])['Hizmet'].count()\
                       .unstack()\
                       .fillna(0)\
                       .applymap(lambda x: 1 if x > 0 else 0)
```

**Açıklama:**
- `groupby(['SepetID', 'Hizmet'])`: Her sepetteki her hizmeti gruplayarak say
- `.unstack()`: Hizmetleri kolona dönüştür
- `.fillna(0)`: Eksik değerleri 0 ile doldur
- `.applymap(lambda x: 1 if x > 0 else 0)`: Alındı (1) / Alınmadı (0) yapısına dönüştür

#### Adım 2: Apriori Algoritması ile Sık Hizmet Kombinasyonları Bulma

**Apriori Algoritması Nedir?**
- Veri madenciliğinde sık itemset (frequently occurring itemsets) bulma algoritması
- Minimum destek eşiğine uyan tüm hizmet kombinasyonlarını bulur
- Combinatorial explosion sorununu çözerek verimli çalışır

```python
from mlxtend.frequent_patterns import apriori

frequent_itemsets = apriori(invoice_product_df, 
                            min_support=0.01,  # Minimum %1 desteği olan kombinasyonlar
                            use_colnames=True)
```

**min_support=0.01 Nedir?**
- Eğer bir hizmet kombinasyonu tüm sepetlerin %1'inde veya daha fazlasında ortaya çıkıyorsa, sık kombinasyon olarak kabul edilir
- Çok nadir kombinasyonları (gürültü) filtreler
- Anlamlı kuralları ön plana çıkarır

#### Adım 3: Association Rules Oluşturma

```python
from mlxtend.frequent_patterns import association_rules

rules = association_rules(frequent_itemsets, 
                          metric="support",      # Kural seçim metriği
                          min_threshold=0.01)    # Minimum destek eşiği
```

**Elde Edilen Sütunlar:**

| Sütun | Açıklama | Formül | Yorumu |
|-------|----------|--------|--------|
| **support** | Antecedent ve consequent'in birlikte görülme sıklığı | P(A ∩ B) | Satın alma gücü |
| **confidence** | Antecedent verildiğinde consequent'in satın alınma olasılığı | P(B\|A) = P(A∩B) / P(A) | Kural gücü |
| **lift** | Antecedent ve consequent'in bağımsız olup olmadığı | P(A∩B) / (P(A)×P(B)) | En önemli metrik |

**Lift Açıklaması:**
- **Lift = 1:** Hizmetler bağımsız (satın alma ilişkisi yok)
- **Lift > 1:** Pozitif korelasyon (A alındığında B daha çok satılır) ✅
- **Lift < 1:** Negatif korelasyon (A alındığında B daha az satılır) ❌

### GÖREV 3: Tavsiye Sistemi

#### Tavsiye Fonksiyonu

```python
def arl_recommender(rules_df, product_id, rec_count=1):
    """
    Verilen bir hizmete dayalı olarak tavsiye sunma fonksiyonu
    
    Parametreler:
    - rules_df: Association rules
    - product_id: Tavsiye yapılacak hizmet ID'si (örn: "2_0")
    - rec_count: Kaç adet tavsiye sunulacağı
    
    Çıktı: Tavsiye edilen hizmetler listesi
    """
    
    # Kuralları lift değerine göre büyükten küçüğe sırala
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    
    # Verilen hizmeti "antecedent" içeren kuralları bul
    recommendation_list = []
    
    for idx, row in sorted_rules.iterrows():
        for product in row["antecedents"]:
            if product == product_id:
                recommendation_list.append(row["consequents"])
    
    # İstenen sayı kadar tavsiye döndür
    return list(recommendation_list)[:rec_count]
```

**Tavsiye Mantığı:**
1. Kuralları lift değerine göre sırala (en güçlü ilişkiden başla)
2. Verilen hizmetin antecedent olduğu kuralları bul
3. Bu kuralların consequent'lerini tavsiye olarak sun

**Örnek:**
```python
# Müşteri 2_0 hizmetini aldı
# Tavsiye: Hangi hizmet alması olasıdır?

recommendation = arl_recommender(rules, product_id="2_0", rec_count=1)
# Çıktı: Lift'i en yüksek olan hizmet
```

## Proje Çıktıları

### Ana Çıktılar

1. **Sık Hizmet Kombinasyonları** (Frequent Itemsets)
   - Beraber satın alınan hizmetlerin listesi
   - Her kombinasyonun destek değeri

2. **Birliktelik Kuralları** (Association Rules)
   - Antecedent → Consequent ilişkileri
   - Support, confidence, lift metrikleri

3. **Personalized Tavsiyeler**
   - Müşterinin aldığı hizmete dayalı smart öneriler
   - Lift'e göre sıralanmış en güçlü öneriler

### Dosyalar

```
Armut_ARL_Recommender_System/
├── Armut_ARL_Recommender_System.py    # Analiz ve tavsiye kodu
├── armut_data.csv                     # İnput veri seti
└── README.md                          # Bu dokümantasyon
```

## İş Değeri ve Faydaları

### Armut Platformu İçin
- **Revenue Artışı:** Ek hizmet satışları sunarak gelir artırma
- **Müşteri Deneyimi:** Personalized öneriler müşteri memnuniyetini arttırır
- **Cross-Selling Fırsatı:** Müşterilerin alacakları hizmetleri proaktif olarak sunma

### Hizmet Verenler İçin
- Müşterilerin hangi hizmetleri bir arada aldığını anlama
- Hizmet paketlemesi stratejisi geliştirme
- Müşteri satın alma davranışının öğrenilmesi

### Müşteriler İçin
- **Keşif:** İhtiyaç duydukları ama bilmediği hizmetleri öğrenme
- **Kolaylık:** Ayrı ayrı arama yerine paket hizmet sunuları
- **Tasarruf:** Platform tarafından sunulan fırsat önceliği

## Teknik Detaylar

### Kullanılan Kütüphaneler

```python
import pandas as pd               # Veri işleme
from mlxtend.frequent_patterns import apriori, association_rules
```

### Parametreler

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| `min_support` | 0.01 | Minimum %1 desteği olan kombinasyonlar |
| `metric` | "support" | Kural seçim kriteri |
| `min_threshold` | 0.01 | Minimum destek eşiği |

### Örnek Hesaplama

```
Hizmet A: 100 sepette
Hizmet B: 80 sepette
A ve B birlikte: 40 sepette
Toplam sepet: 5000

Support(A→B) = 40/5000 = 0.008 = %0.8
Confidence(A→B) = 40/100 = 0.4 = %40
Lift(A→B) = 0.008 / (0.02 × 0.016) = 25

İnterpretasyon:
- %0.8 tüm satışların bu kombinasyonunun payı
- A alındığında B satın alınma ihtimali %40
- A alındığında B 25 kat daha sık satılır (çok güçlü ilişki!)
```

## Geliştirme Önerileri

- [ ] Zamansal trend analizi (mevsimsellik)
- [ ] Hizmet kategorilerine göre segment analizi
- [ ] Fiyat elastikiyeti ile lift değerinin birleştirilmesi
- [ ] Görselleştirme dashboard'u (network graph)
- [ ] Reel-time tavsiye sistemi API'si
- [ ] Confidence temelli alternatif tavsiye mekanizması
- [ ] Top-K önerileri (birden fazla tavsiye sunma)

## Kurulum ve Kullanım

```bash
# Gerekli kütüphaneleri yükleyin
pip install pandas mlxtend

# Projeyi çalıştırın
python Armut_ARL_Recommender_System.py
```

## Kaynaklar

- Agrawal, R., Imieliński, T., & Swami, A. (1993). "Mining Association Rules between Sets of Items in Large Databases"
- MLxtend Documentation: http://rasbt.github.io/mlxtend/
- Association Rule Learning Wikipedia: https://en.wikipedia.org/wiki/Association_rule_learning
