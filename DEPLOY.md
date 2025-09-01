# 🚀 Netlify Deploy Talimatları

Bu dosya Smart Farming AI uygulamasını Netlify üzerinde deploy etmek için gerekli adımları içerir.

## 📋 Ön Gereksinimler

1. **GitHub Repository**: Proje GitHub'da yayınlanmış olmalı
2. **Netlify Hesabı**: [netlify.com](https://netlify.com) üzerinde ücretsiz hesap
3. **Claude API Anahtarı**: Anthropic'ten alınmış API anahtarı

## 🔧 Deploy Adımları

### 1. GitHub Repository Hazırlığı

```bash
# Projeyi GitHub'a yükleyin
git add .
git commit -m "Initial commit for Netlify deployment"
git push origin main
```

### 2. Netlify'da Yeni Site Oluşturma

1. [Netlify Dashboard](https://app.netlify.com) açın
2. "New site from Git" butonuna tıklayın
3. "GitHub" seçeneğini seçin
4. Repository'nizi seçin: `atahanaltan123/smartfarming_ai`

### 3. Build Ayarları

**Build Command:**
```bash
pip install -r requirements.txt
```

**Publish Directory:**
```
.
```

**Python Version:**
```
3.9
```

### 4. Environment Variables

Netlify'da Site Settings > Environment Variables bölümünde:

```
CLAUDE_API_KEY = your_claude_api_key_here
FLASK_ENV = production
FLASK_DEBUG = False
```

### 5. Deploy

1. "Deploy site" butonuna tıklayın
2. Build işleminin tamamlanmasını bekleyin
3. Site URL'nizi alın

## 🔧 Alternatif Deploy Yöntemleri

### Manuel Deploy

```bash
# Netlify CLI ile
npm install -g netlify-cli
netlify login
netlify deploy --prod
```

### Drag & Drop Deploy

1. Proje klasörünü ZIP olarak sıkıştırın
2. Netlify dashboard'da "Deploy manually" seçin
3. ZIP dosyasını sürükleyip bırakın

## 🐛 Sorun Giderme

### Build Hataları

- **Python version**: `runtime.txt` dosyasında Python 3.9 belirtildi
- **Dependencies**: `requirements.txt` dosyası güncel
- **API Keys**: Environment variables doğru ayarlandı

### Runtime Hataları

- **Port**: Uygulama `PORT` environment variable'ını kullanıyor
- **Uploads**: `uploads` klasörü otomatik oluşturuluyor
- **CORS**: API endpoints CORS ayarları yapıldı

## 📊 Monitoring

Netlify'da şu bölümleri kontrol edin:

- **Deploys**: Deploy geçmişi
- **Functions**: Serverless functions
- **Analytics**: Site istatistikleri
- **Forms**: Form submissions

## 🔄 Otomatik Deploy

GitHub'a her push yaptığınızda Netlify otomatik olarak yeni bir deploy başlatacak.

## 📞 Destek

Sorun yaşarsanız:

1. Netlify build logs'unu kontrol edin
2. GitHub repository'yi kontrol edin
3. Environment variables'ları kontrol edin
4. API anahtarının geçerli olduğundan emin olun

---

✅ **Başarılı Deploy Sonrası**

Site URL'niz: `https://your-site-name.netlify.app`

Artık Smart Farming AI uygulamanız herkese açık!
