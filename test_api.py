import requests
import json
import os

def test_claude_api():
    """Claude API anahtarını test eder"""
    
    # API bilgileri
    api_key = os.getenv('CLAUDE_API_KEY', '')
    api_url = "https://api.anthropic.com/v1/messages"
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01"
    }
    
    # Basit bir test mesajı - Güncel model adı
    message = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 100,
        "messages": [
            {
                "role": "user",
                "content": "Merhaba! Sen bir AI asistanısın. Sadece 'Merhaba, API çalışıyor!' yanıtını ver."
            }
        ]
    }
    
    try:
        print("🔍 Claude API test ediliyor...")
        print(f"📡 API URL: {api_url}")
        print(f"🔑 API Anahtarı: {api_key[:20]}...")
        print("⏳ İstek gönderiliyor...")
        
        response = requests.post(api_url, headers=headers, json=message)
        
        print(f"📊 HTTP Durum Kodu: {response.status_code}")
        print(f"📝 Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API Başarılı!")
            print(f"🤖 AI Yanıtı: {result['content'][0]['text']}")
            return True
        else:
            print(f"❌ API Hatası: {response.status_code}")
            print(f"📄 Hata Detayı: {response.text}")
            
            # Hata analizi
            if response.status_code == 401:
                print("🔐 401 Hatası: API anahtarı geçersiz veya süresi dolmuş")
            elif response.status_code == 403:
                print("🚫 403 Hatası: API anahtarı yetkisiz")
            elif response.status_code == 404:
                print("🔍 404 Hatası: API endpoint bulunamadı")
            elif response.status_code == 429:
                print("⏰ 429 Hatası: Rate limit aşıldı")
            elif response.status_code == 500:
                print("💥 500 Hatası: Sunucu hatası")
            
            return False
            
    except Exception as e:
        print(f"💥 Bağlantı Hatası: {str(e)}")
        return False

def test_plant_advice():
    """Bitki önerisi API'sini test eder"""
    
    print("\n🌱 Bitki Önerisi API Testi...")
    
    api_key = os.getenv('CLAUDE_API_KEY', '')
    api_url = "https://api.anthropic.com/v1/messages"
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01"
    }
    
    message = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 200,
        "messages": [
            {
                "role": "user",
                "content": "Domates bitkisi için yaz mevsiminde sulama önerisi ver. Sadece 2-3 cümle yanıtla."
            }
        ]
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=message)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Bitki Önerisi API Başarılı!")
            print(f"🌿 Öneri: {result['content'][0]['text']}")
            return True
        else:
            print(f"❌ Bitki Önerisi API Hatası: {response.status_code}")
            print(f"📄 Hata: {response.text}")
            return False
            
    except Exception as e:
        print(f"💥 Bağlantı Hatası: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Claude API Test Başlıyor...\n")
    
    # Ana API testi
    api_works = test_claude_api()
    
    if api_works:
        # Bitki önerisi testi
        test_plant_advice()
    
    print("\n" + "="*50)
    if api_works:
        print("🎉 API Testi Başarılı! Uygulama çalışmaya hazır.")
    else:
        print("⚠️ API Testi Başarısız! Lütfen API anahtarını kontrol edin.")
