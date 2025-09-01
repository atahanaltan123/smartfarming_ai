# 🌱 Smart Farming AI - Akıllı Tarım Uygulaması

Claude AI teknolojisi kullanarak geliştirilmiş, bitki hastalık tespiti ve tarım önerileri sunan yapay zeka uygulaması.

## 🚀 Özellikler

### 🔍 **Bitki Hastalık Analizi**
- Görüntü yükleme ile bitki hastalık tespiti
- Claude AI ile gelişmiş görüntü analizi
- Detaylı hastalık raporu ve tedavi önerileri
- Bitki türüne özel analiz

### 💡 **Akıllı Bakım Önerileri**
- Bitki türüne göre özelleştirilmiş öneriler
- Mevsimsel bakım tavsiyeleri
- Sulama, gübreleme ve budama rehberi

### 📊 **Analiz Geçmişi**
- Tüm analizlerin kayıt altına alınması
- Görüntü arşivi ve sonuç takibi
- Zaman bazlı filtreleme

## 🛠️ Teknolojiler

- **Backend**: Python Flask
- **AI**: Claude 3 Sonnet API
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Görüntü İşleme**: Base64 encoding
- **Veri Saklama**: JSON dosya sistemi

## 📋 Kurulum

### 1. **Gereksinimler**
```bash
Python 3.8+
pip
```

### 2. **Proje Klonlama**
```bash
git clone <repository-url>
cd smart-farming-ai
```

### 3. **Bağımlılıkları Yükleme**
```bash
pip install -r requirements.txt
```

### 4. **API Anahtarı Yapılandırması**
```bash
# Environment variable olarak API anahtarını ayarlayın
export CLAUDE_API_KEY="your_claude_api_key_here"

# Windows için:
set CLAUDE_API_KEY=your_claude_api_key_here
```

Veya `env.example` dosyasını `.env` olarak kopyalayıp API anahtarınızı ekleyin.

### 5. **Uygulamayı Çalıştırma**
```bash
python app.py
```

Uygulama `http://localhost:5000` adresinde çalışmaya başlayacaktır.

## 🎯 Kullanım

### **Bitki Analizi**
1. Ana sayfada "Analiz" bölümüne gidin
2. Bitki türünü seçin (opsiyonel)
3. Bitki yaprağı fotoğrafını yükleyin
4. "Analiz Et" butonuna tıklayın
5. AI analiz sonuçlarını bekleyin

### **Bakım Önerileri**
1. "Öneriler" bölümüne gidin
2. Bitki türünü girin
3. Mevsimi seçin
4. "Öneri Al" butonuna tıklayın

### **Geçmiş Takibi**
1. "Geçmiş" bölümüne gidin
2. "Geçmişi Yükle" butonuna tıklayın
3. Önceki analizleri görüntüleyin

## 🔧 API Endpoints

### **POST /api/analyze**
Bitki görüntüsü analizi
```json
{
  "image": "file",
  "plant_type": "string"
}
```

### **POST /api/advice**
Bitki bakım önerisi
```json
{
  "plant_type": "string",
  "season": "string"
}
```

### **GET /api/history**
Analiz geçmişi

## 📁 Proje Yapısı

```
smart-farming-ai/
├── app.py                 # Flask ana uygulama
├── claude_client.py      # Claude API istemcisi
├── config.py             # Yapılandırma ayarları
├── requirements.txt      # Python bağımlılıkları
├── README.md            # Proje dokümantasyonu
├── templates/           # HTML şablonları
│   └── index.html      # Ana sayfa
├── uploads/            # Yüklenen görüntüler
└── analysis_history.json # Analiz geçmişi
```

## 🔒 Güvenlik

- API anahtarı güvenli şekilde saklanır
- Dosya yükleme güvenlik kontrolleri
- Güvenli dosya adlandırma
- Dosya türü doğrulaması

## 🚀 Gelecek Özellikler

- [ ] Mobil uygulama (React Native)
- [ ] Gerçek zamanlı sensör entegrasyonu
- [ ] Toprak analizi API'si
- [ ] Hava durumu entegrasyonu
- [ ] Çoklu dil desteği
- [ ] Kullanıcı hesapları
- [ ] Veritabanı entegrasyonu

## 🤝 Katkıda Bulunma

1. Projeyi fork edin
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some AmazingFeature'`)
4. Branch'inizi push edin (`git push origin feature/AmazingFeature`)
5. Pull Request oluşturun

## 📞 İletişim

- **Proje**: Smart Farming AI
- **Geliştirici**: [Adınız]
- **E-posta**: [E-posta adresiniz]

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın.

## 🙏 Teşekkürler

- **Anthropic** - Claude AI API'si için
- **Flask** - Web framework için
- **Bootstrap** - UI bileşenleri için
- **Font Awesome** - İkonlar için

---

⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!
