import requests
import json
import base64
from typing import Optional, Dict, Any
from config import Config

class ClaudeClient:
    """Claude API ile iletişim kuran istemci sınıfı"""
    
    def __init__(self):
        self.api_key = Config.get_claude_api_key()
        self.api_url = Config.CLAUDE_API_URL
        self.model = Config.CLAUDE_MODEL
        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
    
    def analyze_plant_image(self, image_path: str, plant_type: str = "genel") -> Dict[str, Any]:
        """
        Bitki görüntüsünü analiz eder ve hastalık tespiti yapar
        
        Args:
            image_path: Görüntü dosyasının yolu
            plant_type: Bitki türü (opsiyonel)
            
        Returns:
            Analiz sonuçları
        """
        try:
            # Görüntüyü base64'e çevir
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Claude API'ye gönderilecek mesaj
            message = {
                "model": self.model,
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""Bu bitki görüntüsünü analiz et ve aşağıdaki bilgileri ver:

1. **Hastalık Tespiti**: Görüntüde herhangi bir hastalık belirtisi var mı?
2. **Hastalık Türü**: Eğer hastalık varsa, ne tür bir hastalık?
3. **Belirtiler**: Hangi görsel belirtiler hastalığı işaret ediyor?
4. **Tedavi Önerileri**: Bu hastalık için hangi tedavi yöntemleri önerilir?
5. **Önleyici Tedbirler**: Gelecekte benzer hastalıkları önlemek için ne yapılmalı?
6. **Güvenilirlik**: Bu analizin güvenilirlik oranı nedir?

Bitki türü: {plant_type}

Lütfen Türkçe olarak detaylı bir rapor hazırla."""
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_data
                                }
                            }
                        ]
                    }
                ]
            }
            
            # API'ye istek gönder
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=message
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "analysis": result["content"][0]["text"],
                    "model": self.model,
                    "timestamp": result.get("usage", {}).get("input_tokens", 0)
                }
            else:
                return {
                    "success": False,
                    "error": f"API Hatası: {response.status_code}",
                    "details": response.text
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"İşlem hatası: {str(e)}"
            }
    
    def analyze_lab_results(self, test_type: str, lab_data: str, patient_info: str = "", reference_range: str = "adult") -> Dict[str, Any]:
        """
        Laboratuvar sonuçlarını analiz eder ve tıbbi öneriler sunar
        
        Args:
            test_type: Test türü (blood, urine, microbiology, pathology, biochemistry)
            lab_data: Laboratuvar verisi
            patient_info: Hasta bilgileri
            reference_range: Referans aralığı (adult, pediatric, geriatric, pregnancy)
            
        Returns:
            Analiz sonuçları
        """
        try:
            # Claude API'ye gönderilecek mesaj
            message = {
                "model": self.model,
                "max_tokens": 1500,
                "messages": [
                    {
                        "role": "user",
                        "content": f"""Sen bir tıbbi laboratuvar uzmanısın. Aşağıdaki laboratuvar sonuçlarını analiz et:

**Test Türü:** {test_type}
**Referans Aralığı:** {reference_range}
**Hasta Bilgileri:** {patient_info if patient_info else 'Belirtilmemiş'}

**Laboratuvar Verisi:**
{lab_data}

**Analiz Görevleri:**
1. **Anormal Değerler:** Referans aralığı dışındaki değerleri tespit et
2. **Kritik Uyarılar:** Acil müdahale gerektiren sonuçları belirle
3. **Klinik Yorum:** Sonuçların klinik anlamını açıkla
4. **Öneriler:** Hangi ek testler yapılmalı, tedavi önerileri
5. **Takip:** Ne zaman kontrol testleri yapılmalı

**Format:** Türkçe olarak detaylı bir rapor hazırla. Anormal değerleri tablo halinde göster.
**Güvenlik:** Bu analiz sadece eğitim amaçlıdır, tıbbi tanı yerine geçmez."""
                    }
                ]
            }
            
            # API'ye istek gönder
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=message
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "analysis": result["content"][0]["text"],
                    "model": self.model,
                    "test_type": test_type,
                    "reference_range": reference_range,
                    "timestamp": result.get("usage", {}).get("input_tokens", 0)
                }
            else:
                return {
                    "success": False,
                    "error": f"API Hatası: {response.status_code}",
                    "details": response.text
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"İşlem hatası: {str(e)}"
            }

    def get_plant_care_advice(self, plant_type: str, season: str = "genel") -> Dict[str, Any]:
        """
        Belirli bir bitki türü için bakım önerileri alır
        
        Args:
            plant_type: Bitki türü
            season: Mevsim (opsiyonel)
            
        Returns:
            Bakım önerileri
        """
        try:
            message = {
                "model": self.model,
                "max_tokens": 800,
                "messages": [
                    {
                        "role": "user",
                        "content": f"""'{plant_type}' bitkisi için detaylı bakım rehberi hazırla:

1. **Sulama**: Ne sıklıkla ve nasıl sulanmalı?
2. **Gübreleme**: Hangi gübreler kullanılmalı?
3. **Işık İhtiyacı**: Güneş ışığı ihtiyacı nasıl?
4. **Toprak**: Hangi toprak türü uygun?
5. **Budama**: Budama gereksinimleri neler?
6. **Mevsimsel Bakım**: {season} mevsiminde özel dikkat edilmesi gerekenler neler?

Lütfen Türkçe olarak pratik öneriler ver."""
                    }
                ]
            }
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=message
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "advice": result["content"][0]["text"],
                    "plant_type": plant_type,
                    "season": season
                }
            else:
                return {
                    "success": False,
                    "error": f"API Hatası: {response.status_code}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"İşlem hatası: {str(e)}"
            }
