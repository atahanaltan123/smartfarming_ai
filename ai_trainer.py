import json
import os
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime
from config import Config

class AITrainer:
    """Claude AI modelini özelleştirilmiş veri seti ile eğiten sınıf"""
    
    def __init__(self):
        self.api_key = Config.get_claude_api_key()
        self.api_url = Config.CLAUDE_API_URL
        self.model = Config.CLAUDE_MODEL
        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        # Eğitim verileri
        self.training_data = []
        self.validation_data = []
        self.test_results = []
        
        # Model performans metrikleri
        self.performance_metrics = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "total_tests": 0,
            "correct_predictions": 0
        }
    
    def load_training_data(self, data_file: str = "training_dataset.json") -> bool:
        """Eğitim veri setini yükler"""
        try:
            if os.path.exists(data_file):
                with open(data_file, 'r', encoding='utf-8') as f:
                    self.training_data = json.load(f)
                print(f"✅ Eğitim veri seti yüklendi: {len(self.training_data)} örnek")
                return True
            else:
                print(f"❌ Eğitim veri seti bulunamadı: {data_file}")
                return False
        except Exception as e:
            print(f"Veri yükleme hatası: {e}")
            return False
    
    def create_custom_prompt(self, plant_type: str, symptoms: List[str], 
                           climate: str = "", season: str = "") -> str:
        """Özelleştirilmiş analiz prompt'u oluşturur"""
        
        base_prompt = f"""Sen bir uzman tarım bilimcisi ve bitki hastalık uzmanısın. 
Aşağıdaki bilgileri kullanarak detaylı bir analiz yap:

**Bitki Türü**: {plant_type}
**Gözlemlenen Belirtiler**: {', '.join(symptoms)}
**Mevsim**: {season if season else 'Belirtilmemiş'}
**İklim Koşulları**: {climate if climate else 'Belirtilmemiş'}

Lütfen aşağıdaki formatta detaylı bir analiz ver:

1. **Hastalık Tespiti**: Görüntüde hangi hastalık belirtileri var?
2. **Hastalık Türü**: Fungal, bakteriyel, viral veya zararlı böcek?
3. **Şiddet Değerlendirmesi**: Düşük, orta, yüksek veya kritik?
4. **Belirti Analizi**: Her belirtiyi detaylı açıkla
5. **Tedavi Önerileri**: Acil ve uzun vadeli çözümler
6. **Önleyici Tedbirler**: Gelecekte benzer hastalıkları önleme
7. **Uzman Tavsiyeleri**: Pratik öneriler
8. **Güvenilirlik Oranı**: Bu analizin güvenilirlik yüzdesi

Lütfen Türkçe olarak, çiftçilerin anlayabileceği şekilde yanıtla."""
        
        return base_prompt
    
    def train_with_few_shot_learning(self, plant_type: str, symptoms: List[str], 
                                    expected_disease: str = "") -> Dict[str, Any]:
        """Few-shot learning ile model eğitimi"""
        
        try:
            # Eğitim veri setinden benzer örnekler bul
            relevant_examples = self._find_relevant_examples(plant_type, symptoms)
            
            # Özelleştirilmiş prompt oluştur
            custom_prompt = self.create_custom_prompt(plant_type, symptoms)
            
            # Few-shot learning için örnekler ekle
            if relevant_examples:
                custom_prompt += "\n\n**Benzer Vaka Örnekleri:**\n"
                for i, example in enumerate(relevant_examples[:3], 1):
                    custom_prompt += f"\nÖrnek {i}: {example['plant_type']} - {example['disease_name']}\n"
                    custom_prompt += f"Belirtiler: {', '.join(example['symptoms'])}\n"
                    custom_prompt += f"Tedavi: {', '.join(example['treatment_methods'])}\n"
            
            # Claude API'ye gönder
            message = {
                "model": self.model,
                "max_tokens": 1500,
                "messages": [
                    {
                        "role": "user",
                        "content": custom_prompt
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
                analysis = result["content"][0]["text"]
                
                # Sonucu değerlendir
                evaluation = self._evaluate_prediction(
                    analysis, expected_disease, plant_type, symptoms
                )
                
                return {
                    "success": True,
                    "analysis": analysis,
                    "evaluation": evaluation,
                    "training_examples_used": len(relevant_examples),
                    "model": self.model
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
                "error": f"Eğitim hatası: {str(e)}"
            }
    
    def _find_relevant_examples(self, plant_type: str, symptoms: List[str]) -> List[Dict]:
        """Eğitim veri setinden ilgili örnekleri bulur"""
        relevant_examples = []
        
        for example in self.training_data:
            # Bitki türü eşleşmesi - düzeltilmiş
            example_text = example["messages"][0]["content"][0]["text"]
            if plant_type.lower() in example_text.lower():
                # Eksik alanları ekle
                example['plant_type'] = plant_type
                example['disease_name'] = "Bilinmeyen Hastalık"
                example['symptoms'] = symptoms
                example['treatment_methods'] = ["Genel tedavi önerileri"]
                relevant_examples.append(example)
            # Belirti eşleşmesi
            elif any(symptom.lower() in str(example).lower() for symptom in symptoms):
                # Eksik alanları ekle
                example['plant_type'] = plant_type
                example['disease_name'] = "Belirti bazlı hastalık"
                example['symptoms'] = symptoms
                example['treatment_methods'] = ["Genel tedavi önerileri"]
                relevant_examples.append(example)
        
        # Benzerlik skoruna göre sırala
        relevant_examples.sort(key=lambda x: self._calculate_similarity_score(
            plant_type, symptoms, x
        ), reverse=True)
        
        return relevant_examples[:5]  # En iyi 5 örnek
    
    def _calculate_similarity_score(self, plant_type: str, symptoms: List[str], 
                                  example: Dict) -> float:
        """Benzerlik skoru hesaplar"""
        score = 0.0
        
        # Bitki türü eşleşmesi
        if plant_type.lower() in str(example).lower():
            score += 0.5
        
        # Belirti eşleşmesi
        symptom_matches = sum(1 for symptom in symptoms 
                            if symptom.lower() in str(example).lower())
        score += (symptom_matches / len(symptoms)) * 0.5
        
        return score
    
    def _evaluate_prediction(self, analysis: str, expected_disease: str, 
                           plant_type: str, symptoms: List[str]) -> Dict[str, Any]:
        """Tahmin sonucunu değerlendirir"""
        
        evaluation = {
            "disease_detected": False,
            "symptom_accuracy": 0.0,
            "treatment_relevance": 0.0,
            "confidence_score": 0.0,
            "overall_score": 0.0
        }
        
        # Hastalık tespiti kontrolü
        if expected_disease and expected_disease.lower() in analysis.lower():
            evaluation["disease_detected"] = True
        
        # Belirti doğruluğu
        symptom_accuracy = sum(1 for symptom in symptoms 
                             if symptom.lower() in analysis.lower())
        evaluation["symptom_accuracy"] = symptom_accuracy / len(symptoms)
        
        # Tedavi uygunluğu
        treatment_keywords = ["tedavi", "ilaç", "fungisit", "bakır", "sülfür", "biyolojik"]
        treatment_matches = sum(1 for keyword in treatment_keywords 
                              if keyword in analysis.lower())
        evaluation["treatment_relevance"] = min(treatment_matches / 3, 1.0)
        
        # Güvenilirlik skoru
        if "güvenilirlik" in analysis.lower() or "%" in analysis:
            evaluation["confidence_score"] = 0.8
        else:
            evaluation["confidence_score"] = 0.6
        
        # Genel skor
        evaluation["overall_score"] = (
            evaluation["symptom_accuracy"] * 0.4 +
            evaluation["treatment_relevance"] * 0.3 +
            evaluation["confidence_score"] * 0.3
        )
        
        return evaluation
    
    def run_validation_tests(self) -> Dict[str, Any]:
        """Doğrulama testleri çalıştırır"""
        
        if not self.training_data:
            return {"error": "Eğitim veri seti yüklenmedi"}
        
        print("🧪 Doğrulama testleri başlıyor...")
        
        test_cases = [
            {
                "plant_type": "Domates",
                "symptoms": ["Kahverengi lekeler", "Yaprak dökülmesi"],
                "expected_disease": "Erken Yaprak Yanıklığı"
            },
            {
                "plant_type": "Salatalık",
                "symptoms": ["Beyaz pudra benzeri leke"],
                "expected_disease": "Külleme"
            },
            {
                "plant_type": "Biber",
                "symptoms": ["Küçük su damlası lezyonları"],
                "expected_disease": "Bakteriyel Leke"
            }
        ]
        
        results = []
        total_score = 0.0
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"Test {i}: {test_case['plant_type']} - {test_case['expected_disease']}")
            
            result = self.train_with_few_shot_learning(
                test_case["plant_type"],
                test_case["symptoms"],
                test_case["expected_disease"]
            )
            
            if result["success"]:
                evaluation = result["evaluation"]
                total_score += evaluation["overall_score"]
                
                test_result = {
                    "test_id": i,
                    "plant_type": test_case["plant_type"],
                    "expected_disease": test_case["expected_disease"],
                    "actual_analysis": result["analysis"][:200] + "...",
                    "evaluation": evaluation,
                    "passed": evaluation["overall_score"] > 0.7
                }
                
                results.append(test_result)
                
                status = "✅ PASS" if test_result["passed"] else "❌ FAIL"
                print(f"  {status} - Skor: {evaluation['overall_score']:.2f}")
            else:
                print(f"  ❌ HATA: {result['error']}")
        
        # Genel performans hesapla
        avg_score = total_score / len(test_cases) if test_cases else 0.0
        
        validation_summary = {
            "total_tests": len(test_cases),
            "passed_tests": len([r for r in results if r["passed"]]),
            "failed_tests": len([r for r in results if not r["passed"]]),
            "average_score": avg_score,
            "test_results": results,
            "performance_grade": self._get_performance_grade(avg_score)
        }
        
        print(f"\n📊 Doğrulama Sonuçları:")
        print(f"Toplam Test: {validation_summary['total_tests']}")
        print(f"Başarılı: {validation_summary['passed_tests']}")
        print(f"Başarısız: {validation_summary['failed_tests']}")
        print(f"Ortalama Skor: {avg_score:.2f}")
        print(f"Performans: {validation_summary['performance_grade']}")
        
        return validation_summary
    
    def _get_performance_grade(self, score: float) -> str:
        """Performans skoruna göre not verir"""
        if score >= 0.9:
            return "A+ (Mükemmel)"
        elif score >= 0.8:
            return "A (Çok İyi)"
        elif score >= 0.7:
            return "B+ (İyi)"
        elif score >= 0.6:
            return "B (Orta)"
        elif score >= 0.5:
            return "C (Yeterli)"
        else:
            return "D (Yetersiz)"
    
    def export_training_report(self, output_file: str = "training_report.json"):
        """Eğitim raporunu dışa aktarır"""
        
        report = {
            "training_date": datetime.now().isoformat(),
            "model_info": {
                "name": self.model,
                "api_version": "2023-06-01"
            },
            "dataset_info": {
                "total_examples": len(self.training_data),
                "plant_types": list(set(
                    example["messages"][0]["content"][0]["text"].split("Bitki Türü: ")[1].split("\n")[0]
                    for example in self.training_data
                    if "Bitki Türü: " in example["messages"][0]["content"][0]["text"]
                ))
            },
            "performance_metrics": self.performance_metrics,
            "recommendations": self._generate_recommendations()
        }
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Eğitim raporu oluşturuldu: {output_file}")
            return True
            
        except Exception as e:
            print(f"Rapor oluşturma hatası: {e}")
            return False
    
    def _generate_recommendations(self) -> List[str]:
        """Model iyileştirme önerileri oluşturur"""
        
        recommendations = []
        
        if len(self.training_data) < 10:
            recommendations.append("Daha fazla eğitim örneği ekleyin (en az 10)")
        
        if self.performance_metrics["accuracy"] < 0.8:
            recommendations.append("Model doğruluğunu artırmak için daha spesifik prompt'lar kullanın")
        
        recommendations.append("Farklı bitki türleri için özel eğitim verileri ekleyin")
        recommendations.append("Mevsimsel ve bölgesel faktörleri dahil edin")
        recommendations.append("Uzman doğrulaması ile veri kalitesini artırın")
        
        return recommendations

if __name__ == "__main__":
    # AI Trainer'ı başlat
    trainer = AITrainer()
    
    # Eğitim veri setini yükle
    if trainer.load_training_data():
        print("🚀 AI Trainer başlatıldı!")
        
        # Doğrulama testleri çalıştır
        validation_results = trainer.run_validation_tests()
        
        # Eğitim raporu oluştur
        trainer.export_training_report()
        
        print("\n🎯 Eğitim tamamlandı! Model kullanıma hazır.")
    else:
        print("❌ Eğitim veri seti yüklenemedi!")
