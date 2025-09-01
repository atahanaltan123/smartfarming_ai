import json
import os
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from config import Config
from image_processor import ImageProcessor

@dataclass
class ModelPrediction:
    """Model tahmin sonucu"""
    model_name: str
    disease_detected: str
    confidence_score: float
    analysis_text: str
    processing_time: float
    features_used: List[str]
    metadata: Dict[str, Any]

class ModelOrchestrator:
    """Çoklu AI model entegrasyonu ve ensemble learning"""
    
    def __init__(self):
        self.api_key = Config.get_claude_api_key()
        self.api_url = Config.CLAUDE_API_URL
        self.models = {
            "claude-3-5-sonnet": {
                "name": "Claude 3.5 Sonnet",
                "version": "20241022",
                "specialization": "Genel bitki hastalık analizi",
                "weight": 0.4
            },
            "claude-3-haiku": {
                "name": "Claude 3 Haiku",
                "version": "20240307",
                "specialization": "Hızlı ön analiz",
                "weight": 0.2
            },
            "claude-3-opus": {
                "name": "Claude 3 Opus",
                "version": "20240229",
                "specialization": "Detaylı uzman analizi",
                "weight": 0.4
            }
        }
        
        self.image_processor = ImageProcessor()
        self.ensemble_weights = {
            "claude-3-5-sonnet": 0.4,
            "claude-3-haiku": 0.2,
            "claude-3-opus": 0.4
        }
        
        # Model performans geçmişi
        self.performance_history = {}
        
    def analyze_with_multiple_models(self, image_path: str, plant_type: str = "genel") -> Dict[str, Any]:
        """Birden fazla model ile paralel analiz"""
        
        print("🚀 Çoklu model analizi başlıyor...")
        
        # Görüntü ön işleme
        image_features = self._extract_image_features(image_path)
        
        # Paralel model analizleri
        predictions = {}
        start_time = datetime.now()
        
        for model_id, model_info in self.models.items():
            try:
                print(f"🔍 {model_info['name']} analiz ediyor...")
                
                prediction = self._analyze_with_single_model(
                    model_id, image_path, plant_type, image_features
                )
                
                if prediction:
                    predictions[model_id] = prediction
                    print(f"✅ {model_info['name']} tamamlandı")
                else:
                    print(f"❌ {model_info['name']} başarısız")
                    
            except Exception as e:
                print(f"❌ {model_info['name']} hatası: {e}")
        
        # Ensemble sonuçları
        if predictions:
            ensemble_result = self._create_ensemble_prediction(predictions, image_features)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": True,
                "ensemble_result": ensemble_result,
                "individual_predictions": predictions,
                "processing_time": processing_time,
                "models_used": list(predictions.keys()),
                "image_features": image_features
            }
        else:
            return {
                "success": False,
                "error": "Hiçbir model analiz tamamlayamadı"
            }
    
    def _analyze_with_single_model(self, model_id: str, image_path: str, 
                                  plant_type: str, image_features: Dict) -> Optional[ModelPrediction]:
        """Tek model ile analiz"""
        
        try:
            start_time = datetime.now()
            
            # Model-spesifik prompt oluştur
            prompt = self._create_model_specific_prompt(model_id, plant_type, image_features)
            
            # API isteği
            message = {
                "model": model_id,
                "max_tokens": 1500,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": self.image_processor.image_to_base64(image_path)
                                }
                            }
                        ]
                    }
                ]
            }
            
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01"
            }
            
            response = requests.post(self.api_url, headers=headers, json=message)
            
            if response.status_code == 200:
                result = response.json()
                analysis_text = result["content"][0]["text"]
                
                # Tahmin sonucunu parse et
                prediction = self._parse_prediction_result(analysis_text, model_id)
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                return ModelPrediction(
                    model_name=self.models[model_id]["name"],
                    disease_detected=prediction.get("disease", "Bilinmeyen"),
                    confidence_score=prediction.get("confidence", 0.0),
                    analysis_text=analysis_text,
                    processing_time=processing_time,
                    features_used=list(image_features.keys()),
                    metadata=prediction
                )
            else:
                print(f"API Hatası {model_id}: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Model analiz hatası {model_id}: {e}")
            return None
    
    def _create_model_specific_prompt(self, model_id: str, plant_type: str, 
                                    image_features: Dict) -> str:
        """Model-spesifik prompt oluşturur"""
        
        base_prompt = f"""Sen bir uzman tarım bilimcisi ve bitki hastalık uzmanısın.
Aşağıdaki bilgileri kullanarak detaylı bir analiz yap:

**Bitki Türü**: {plant_type}
**Görüntü Özellikleri**: {json.dumps(image_features, ensure_ascii=False, indent=2)}

**Analiz Gereksinimleri**:"""

        if "haiku" in model_id:
            # Hızlı analiz için kısa prompt
            base_prompt += """
1. Hastalık tespiti (varsa)
2. Şiddet seviyesi
3. Acil tedavi önerisi
4. Güvenilirlik skoru

Lütfen kısa ve öz yanıtla."""
            
        elif "opus" in model_id:
            # Detaylı analiz için kapsamlı prompt
            base_prompt += """
1. **Hastalık Tespiti**: Detaylı belirti analizi
2. **Hastalık Türü**: Fungal/Bakteriyel/Viral/Zararlı böcek
3. **Şiddet Değerlendirmesi**: Düşük/Orta/Yüksek/Kritik
4. **Belirti Analizi**: Her belirtiyi detaylı açıkla
5. **Tedavi Önerileri**: Acil, kısa ve uzun vadeli çözümler
6. **Önleyici Tedbirler**: Gelecekte benzer hastalıkları önleme
7. **Uzman Tavsiyeleri**: Pratik öneriler ve dikkat edilecek noktalar
8. **Güvenilirlik Oranı**: Bu analizin güvenilirlik yüzdesi
9. **Diferansiyel Tanı**: Benzer hastalıklarla karşılaştırma
10. **Prognoz**: Hastalığın seyri ve sonucu

Lütfen Türkçe olarak, çiftçilerin anlayabileceği şekilde detaylı yanıtla."""
            
        else:
            # Standart analiz
            base_prompt += """
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
    
    def _extract_image_features(self, image_path: str) -> Dict[str, Any]:
        """Görüntü özelliklerini çıkarır"""
        
        try:
            # Görüntü işleme
            features = self.image_processor.extract_image_features(image_path)
            
            if features['success']:
                return {
                    'basic_stats': features['basic_stats'],
                    'texture_features': features['texture_features'],
                    'color_features': features['color_features'],
                    'edge_features': features['edge_features'],
                    'leaf_analysis': features['leaf_analysis']
                }
            else:
                return {'error': 'Görüntü özellikleri çıkarılamadı'}
                
        except Exception as e:
            return {'error': f'Özellik çıkarma hatası: {str(e)}'}
    
    def _parse_prediction_result(self, analysis_text: str, model_id: str) -> Dict[str, Any]:
        """Analiz metninden tahmin sonucunu parse eder"""
        
        try:
            # Basit parsing (gerçek uygulamada daha gelişmiş NLP kullanılabilir)
            result = {
                "disease": "Bilinmeyen",
                "confidence": 0.0,
                "severity": "Bilinmeyen",
                "disease_type": "Bilinmeyen",
                "treatment_suggestions": [],
                "prevention_methods": []
            }
            
            # Hastalık tespiti
            if "hastalık" in analysis_text.lower() or "disease" in analysis_text.lower():
                # Basit keyword matching
                disease_keywords = ["erken yaprak yanıklığı", "külleme", "bakteriyel leke", 
                                 "late blight", "powdery mildew", "bacterial spot"]
                
                for keyword in disease_keywords:
                    if keyword.lower() in analysis_text.lower():
                        result["disease"] = keyword
                        break
            
            # Güvenilirlik skoru
            if "%" in analysis_text:
                import re
                confidence_match = re.search(r'(\d+)%', analysis_text)
                if confidence_match:
                    result["confidence"] = float(confidence_match.group(1)) / 100
            
            # Şiddet seviyesi
            severity_keywords = {
                "düşük": ["düşük", "low", "hafif", "mild"],
                "orta": ["orta", "medium", "moderate"],
                "yüksek": ["yüksek", "high", "şiddetli", "severe"],
                "kritik": ["kritik", "critical", "çok şiddetli"]
            }
            
            for severity, keywords in severity_keywords.items():
                if any(keyword in analysis_text.lower() for keyword in keywords):
                    result["severity"] = severity
                    break
            
            # Hastalık türü
            disease_type_keywords = {
                "fungal": ["fungal", "mantar", "fungus", "mikoz"],
                "bacterial": ["bakteriyel", "bacterial", "bakteri"],
                "viral": ["viral", "virus", "virüs"],
                "pest": ["zararlı", "pest", "böcek", "insect"]
            }
            
            for disease_type, keywords in disease_type_keywords.items():
                if any(keyword in analysis_text.lower() for keyword in keywords):
                    result["disease_type"] = disease_type
                    break
            
            return result
            
        except Exception as e:
            print(f"Parsing hatası: {e}")
            return {"disease": "Bilinmeyen", "confidence": 0.0}
    
    def _create_ensemble_prediction(self, predictions: Dict[str, ModelPrediction], 
                                   image_features: Dict) -> Dict[str, Any]:
        """Ensemble tahmin sonucu oluşturur"""
        
        try:
            # Ağırlıklı oy verme
            disease_votes = {}
            confidence_scores = []
            severity_votes = {}
            disease_type_votes = {}
            
            for model_id, prediction in predictions.items():
                weight = self.ensemble_weights.get(model_id, 0.1)
                
                # Hastalık oylaması
                disease = prediction.disease_detected
                if disease not in disease_votes:
                    disease_votes[disease] = 0
                disease_votes[disease] += weight
                
                # Güvenilirlik skorları
                confidence_scores.append(prediction.confidence_score * weight)
                
                # Şiddet oylaması
                severity = prediction.metadata.get("severity", "Bilinmeyen")
                if severity not in severity_votes:
                    severity_votes[severity] = 0
                severity_votes[severity] += weight
                
                # Hastalık türü oylaması
                disease_type = prediction.metadata.get("disease_type", "Bilinmeyen")
                if disease_type not in disease_type_votes:
                    disease_type_votes[disease_type] = 0
                disease_type_votes[disease_type] += weight
            
            # En yüksek oy alan sonuçlar
            final_disease = max(disease_votes.items(), key=lambda x: x[1])[0]
            final_severity = max(severity_votes.items(), key=lambda x: x[1])[0]
            final_disease_type = max(disease_type_votes.items(), key=lambda x: x[1])[0]
            
            # Ağırlıklı ortalama güvenilirlik
            final_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
            
            # Model uyumu analizi
            model_agreement = self._analyze_model_agreement(predictions)
            
            ensemble_result = {
                "final_disease": final_disease,
                "final_confidence": final_confidence,
                "final_severity": final_severity,
                "final_disease_type": final_disease_type,
                "model_agreement": model_agreement,
                "voting_details": {
                    "disease_votes": disease_votes,
                    "severity_votes": severity_votes,
                    "disease_type_votes": disease_type_votes
                },
                "ensemble_method": "weighted_voting",
                "models_used": list(predictions.keys())
            }
            
            return ensemble_result
            
        except Exception as e:
            print(f"Ensemble tahmin hatası: {e}")
            return {"error": f"Ensemble tahmin hatası: {str(e)}"}
    
    def _analyze_model_agreement(self, predictions: Dict[str, ModelPrediction]) -> Dict[str, Any]:
        """Model uyumunu analiz eder"""
        
        try:
            # Hastalık tespitinde uyum
            diseases = [pred.disease_detected for pred in predictions.values()]
            disease_agreement = len(set(diseases)) == 1  # Tüm modeller aynı hastalığı tespit etti mi?
            
            # Güvenilirlik skorlarında uyum
            confidences = [pred.confidence_score for pred in predictions.values()]
            confidence_variance = sum((c - sum(confidences)/len(confidences))**2 for c in confidences) / len(confidences)
            confidence_agreement = confidence_variance < 0.1  # Düşük varyans = yüksek uyum
            
            # Genel uyum skoru
            agreement_score = (disease_agreement + confidence_agreement) / 2
            
            return {
                "disease_agreement": disease_agreement,
                "confidence_agreement": confidence_agreement,
                "overall_agreement": agreement_score,
                "confidence_variance": confidence_variance,
                "agreement_level": self._get_agreement_level(agreement_score)
            }
            
        except Exception as e:
            return {"error": f"Uyum analizi hatası: {str(e)}"}
    
    def _get_agreement_level(self, score: float) -> str:
        """Uyum seviyesini belirler"""
        if score >= 0.8:
            return "Yüksek Uyum"
        elif score >= 0.6:
            return "Orta Uyum"
        elif score >= 0.4:
            return "Düşük Uyum"
        else:
            return "Düşük Uyum"
    
    def update_model_performance(self, model_id: str, prediction: ModelPrediction, 
                                actual_result: str = None) -> None:
        """Model performansını günceller"""
        
        if model_id not in self.performance_history:
            self.performance_history[model_id] = {
                "total_predictions": 0,
                "correct_predictions": 0,
                "average_confidence": 0.0,
                "average_processing_time": 0.0,
                "last_updated": datetime.now().isoformat()
            }
        
        history = self.performance_history[model_id]
        history["total_predictions"] += 1
        
        # Güvenilirlik skoru güncelleme
        total_confidence = history["average_confidence"] * (history["total_predictions"] - 1)
        history["average_confidence"] = (total_confidence + prediction.confidence_score) / history["total_predictions"]
        
        # İşlem süresi güncelleme
        total_time = history["average_processing_time"] * (history["total_predictions"] - 1)
        history["average_processing_time"] = (total_time + prediction.processing_time) / history["total_predictions"]
        
        # Doğruluk güncelleme (eğer gerçek sonuç biliniyorsa)
        if actual_result:
            if prediction.disease_detected.lower() in actual_result.lower():
                history["correct_predictions"] += 1
        
        history["last_updated"] = datetime.now().isoformat()
    
    def get_model_performance_report(self) -> Dict[str, Any]:
        """Model performans raporu oluşturur"""
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "total_models": len(self.models),
            "performance_summary": {},
            "recommendations": []
        }
        
        for model_id, history in self.performance_history.items():
            if history["total_predictions"] > 0:
                accuracy = history["correct_predictions"] / history["total_predictions"]
                report["performance_summary"][model_id] = {
                    "accuracy": accuracy,
                    "average_confidence": history["average_confidence"],
                    "average_processing_time": history["average_processing_time"],
                    "total_predictions": history["total_predictions"]
                }
                
                # Öneriler
                if accuracy < 0.7:
                    report["recommendations"].append(f"{model_id} modelinin doğruluğu düşük. Daha fazla eğitim verisi gerekli.")
                if history["average_processing_time"] > 10:
                    report["recommendations"].append(f"{model_id} modeli yavaş. Optimizasyon gerekli.")
        
        return report
    
    def export_ensemble_analysis(self, analysis_result: Dict, output_file: str = "ensemble_analysis.json"):
        """Ensemble analiz sonucunu dışa aktarır"""
        
        try:
            export_data = {
                "analysis_timestamp": datetime.now().isoformat(),
                "ensemble_result": analysis_result,
                "model_performance": self.get_model_performance_report(),
                "system_info": {
                    "total_models": len(self.models),
                    "ensemble_method": "weighted_voting",
                    "api_version": "2023-06-01"
                }
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Ensemble analiz sonucu dışa aktarıldı: {output_file}")
            return True
            
        except Exception as e:
            print(f"❌ Dışa aktarma hatası: {e}")
            return False

if __name__ == "__main__":
    # Model Orchestrator'ı test et
    orchestrator = ModelOrchestrator()
    
    print("🚀 Model Orchestrator başlatıldı!")
    print(f"📊 Toplam {len(orchestrator.models)} model mevcut:")
    
    for model_id, info in orchestrator.models.items():
        print(f"  - {info['name']} ({model_id})")
        print(f"    Uzmanlık: {info['specialization']}")
        print(f"    Ağırlık: {info['weight']}")
    
    print("\n🎯 Test için bir görüntü dosyası gerekli.")
    print("💡 Kullanım: orchestrator.analyze_with_multiple_models('image.jpg', 'Domates')")
