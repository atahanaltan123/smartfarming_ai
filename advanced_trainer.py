import json
import os
import requests
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from config import Config
from data_collector import DataCollector
from ai_trainer import AITrainer

@dataclass
class TrainingExample:
    """Eğitim örneği"""
    id: str
    plant_type: str
    disease_name: str
    symptoms: List[str]
    image_path: str
    expert_analysis: str
    confidence_score: float
    difficulty_level: str  # easy, medium, hard
    metadata: Dict[str, Any]

@dataclass
class TrainingSession:
    """Eğitim oturumu"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    total_examples: int
    correct_predictions: int
    accuracy: float
    models_trained: List[str]
    performance_metrics: Dict[str, Any]

class AdvancedTrainer:
    """Gelişmiş model eğitimi ve fine-tuning sistemi"""
    
    def __init__(self):
        self.api_key = Config.get_claude_api_key()
        self.api_url = Config.CLAUDE_API_URL
        self.data_collector = DataCollector()
        self.ai_trainer = AITrainer()
        
        # Eğitim veri seti
        self.training_examples = []
        self.validation_examples = []
        self.test_examples = []
        
        # Eğitim oturumları
        self.training_sessions = []
        self.current_session = None
        
        # Model performans geçmişi
        self.model_performance = {}
        
        # Eğitim parametreleri
        self.training_params = {
            "learning_rate": 0.001,
            "batch_size": 5,
            "epochs": 10,
            "validation_split": 0.2,
            "test_split": 0.1,
            "early_stopping_patience": 3
        }
    
    def load_training_dataset(self, data_file: str = "training_dataset.json") -> bool:
        """Eğitim veri setini yükler ve böler"""
        
        try:
            if not os.path.exists(data_file):
                print(f"❌ Eğitim veri seti bulunamadı: {data_file}")
                return False
            
            with open(data_file, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            # Veri setini eğitim, doğrulama ve test olarak böl
            total_examples = len(raw_data)
            validation_size = int(total_examples * self.training_params["validation_split"])
            test_size = int(total_examples * self.training_params["test_split"])
            training_size = total_examples - validation_size - test_size
            
            # Veri setini karıştır ve böl
            import random
            random.shuffle(raw_data)
            
            self.training_examples = raw_data[:training_size]
            self.validation_examples = raw_data[training_size:training_size + validation_size]
            self.test_examples = raw_data[training_size + validation_size:]
            
            print(f"✅ Veri seti yüklendi ve bölündü:")
            print(f"  📚 Eğitim: {len(self.training_examples)} örnek")
            print(f"  🔍 Doğrulama: {len(self.validation_examples)} örnek")
            print(f"  🧪 Test: {len(self.test_examples)} örnek")
            
            return True
            
        except Exception as e:
            print(f"❌ Veri seti yükleme hatası: {e}")
            return False
    
    def start_training_session(self, session_name: str = None) -> str:
        """Yeni eğitim oturumu başlatır"""
        
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if session_name:
            session_id = f"{session_name}_{session_id}"
        
        self.current_session = TrainingSession(
            session_id=session_id,
            start_time=datetime.now(),
            end_time=None,
            total_examples=len(self.training_examples),
            correct_predictions=0,
            accuracy=0.0,
            models_trained=[],
            performance_metrics={}
        )
        
        print(f"🚀 Eğitim oturumu başlatıldı: {session_id}")
        print(f"📊 Toplam {len(self.training_examples)} eğitim örneği")
        
        return session_id
    
    def train_model_with_fine_tuning(self, model_name: str = "claude-3-5-sonnet") -> Dict[str, Any]:
        """Fine-tuning ile model eğitimi"""
        
        if not self.current_session:
            print("❌ Aktif eğitim oturumu bulunamadı")
            return {"success": False, "error": "Eğitim oturumu başlatılmadı"}
        
        if not self.training_examples:
            print("❌ Eğitim veri seti yüklenmedi")
            return {"success": False, "error": "Eğitim veri seti bulunamadı"}
        
        print(f"🎯 {model_name} modeli fine-tuning ile eğitiliyor...")
        
        # Eğitim parametreleri
        epochs = self.training_params["epochs"]
        batch_size = self.training_params["batch_size"]
        
        training_results = {
            "success": True,
            "model_name": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "epoch_results": [],
            "final_accuracy": 0.0,
            "training_time": 0.0
        }
        
        start_time = datetime.now()
        
        # Epoch bazlı eğitim
        for epoch in range(epochs):
            print(f"📚 Epoch {epoch + 1}/{epochs}")
            
            epoch_result = self._train_epoch(model_name, epoch, batch_size)
            training_results["epoch_results"].append(epoch_result)
            
            # Doğrulama
            validation_accuracy = self._validate_model(model_name)
            print(f"  ✅ Epoch {epoch + 1} doğrulama doğruluğu: {validation_accuracy:.3f}")
            
            # Early stopping kontrolü
            if self._should_stop_early(training_results["epoch_results"]):
                print(f"  ⏹️ Early stopping - Epoch {epoch + 1}'de durduruldu")
                break
        
        # Eğitim sonuçları
        training_time = (datetime.now() - start_time).total_seconds()
        final_accuracy = self._calculate_final_accuracy(training_results["epoch_results"])
        
        training_results.update({
            "final_accuracy": final_accuracy,
            "training_time": training_time,
            "total_epochs_completed": len(training_results["epoch_results"])
        })
        
        # Oturum güncelleme
        self.current_session.models_trained.append(model_name)
        self.current_session.performance_metrics[model_name] = training_results
        
        print(f"🎉 {model_name} eğitimi tamamlandı!")
        print(f"📊 Final doğruluk: {final_accuracy:.3f}")
        print(f"⏱️ Toplam eğitim süresi: {training_time:.2f} saniye")
        
        return training_results
    
    def _train_epoch(self, model_name: str, epoch: int, batch_size: int) -> Dict[str, Any]:
        """Tek epoch eğitimi"""
        
        epoch_results = {
            "epoch": epoch + 1,
            "batch_results": [],
            "epoch_accuracy": 0.0,
            "total_loss": 0.0
        }
        
        # Batch'lere böl
        total_batches = len(self.training_examples) // batch_size
        if len(self.training_examples) % batch_size != 0:
            total_batches += 1
        
        correct_predictions = 0
        total_predictions = 0
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(self.training_examples))
            
            batch_examples = self.training_examples[start_idx:end_idx]
            
            print(f"    📦 Batch {batch_idx + 1}/{total_batches} ({len(batch_examples)} örnek)")
            
            batch_result = self._train_batch(model_name, batch_examples, batch_idx)
            epoch_results["batch_results"].append(batch_result)
            
            correct_predictions += batch_result["correct_predictions"]
            total_predictions += batch_result["total_predictions"]
        
        # Epoch sonuçları
        epoch_results["epoch_accuracy"] = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        epoch_results["correct_predictions"] = correct_predictions
        epoch_results["total_predictions"] = total_predictions
        
        return epoch_results
    
    def _train_batch(self, model_name: str, batch_examples: List, batch_idx: int) -> Dict[str, Any]:
        """Tek batch eğitimi"""
        
        batch_result = {
            "batch_idx": batch_idx,
            "examples_processed": len(batch_examples),
            "correct_predictions": 0,
            "total_predictions": 0,
            "example_results": []
        }
        
        for example in batch_examples:
            try:
                # Örnek eğitimi
                example_result = self._train_single_example(model_name, example)
                batch_result["example_results"].append(example_result)
                
                if example_result["correct"]:
                    batch_result["correct_predictions"] += 1
                
                batch_result["total_predictions"] += 1
                
            except Exception as e:
                print(f"      ❌ Örnek eğitim hatası: {e}")
        
        return batch_result
    
    def _train_single_example(self, model_name: str, example: Dict) -> Dict[str, Any]:
        """Tek örnek eğitimi"""
        
        try:
            # Claude API ile eğitim
            user_message = example["messages"][0]["content"][0]["text"]
            expected_response = example["messages"][1]["content"]
            
            # Model eğitimi (few-shot learning)
            training_prompt = self._create_training_prompt(user_message, expected_response)
            
            # API isteği
            message = {
                "model": model_name,
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": training_prompt
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
                actual_response = result["content"][0]["text"]
                
                # Doğruluk kontrolü
                is_correct = self._evaluate_response_accuracy(actual_response, expected_response)
                
                return {
                    "correct": is_correct,
                    "expected": expected_response,
                    "actual": actual_response,
                    "accuracy_score": self._calculate_response_similarity(actual_response, expected_response)
                }
            else:
                return {
                    "correct": False,
                    "error": f"API Hatası: {response.status_code}",
                    "accuracy_score": 0.0
                }
                
        except Exception as e:
            return {
                "correct": False,
                "error": str(e),
                "accuracy_score": 0.0
            }
    
    def _create_training_prompt(self, user_message: str, expected_response: str) -> str:
        """Eğitim prompt'u oluşturur"""
        
        prompt = f"""Sen bir bitki hastalık uzmanısın. Aşağıdaki örnek analizi incele ve benzer şekilde yanıtla:

**Kullanıcı Sorusu:**
{user_message}

**Beklenen Uzman Yanıtı:**
{expected_response}

**Görev:** Yukarıdaki örnekteki gibi detaylı ve profesyonel bir analiz yap. Aynı format ve stil kullan.

**Önemli:** Yanıtın beklenen yanıtla tutarlı olmalı ve aynı bilgi kalitesinde olmalı."""
        
        return prompt
    
    def _evaluate_response_accuracy(self, actual: str, expected: str) -> bool:
        """Yanıt doğruluğunu değerlendirir"""
        
        # Basit keyword matching
        expected_keywords = self._extract_keywords(expected)
        actual_keywords = self._extract_keywords(actual)
        
        # Ortak keyword sayısı
        common_keywords = set(expected_keywords) & set(actual_keywords)
        
        # Doğruluk oranı
        accuracy_ratio = len(common_keywords) / len(expected_keywords) if expected_keywords else 0
        
        return accuracy_ratio >= 0.7  # %70 üzeri doğru kabul et
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Metinden anahtar kelimeleri çıkarır"""
        
        # Basit keyword extraction
        import re
        
        # Türkçe anahtar kelimeler
        keywords = [
            "hastalık", "tedavi", "önleme", "belirti", "fungal", "bakteriyel",
            "viral", "zararlı", "sulama", "gübreleme", "budama", "havalandırma",
            "disease", "treatment", "prevention", "symptom", "fungus", "bacterial",
            "viral", "pest", "watering", "fertilization", "pruning", "ventilation"
        ]
        
        found_keywords = []
        text_lower = text.lower()
        
        for keyword in keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def _calculate_response_similarity(self, actual: str, expected: str) -> float:
        """Yanıt benzerliğini hesaplar"""
        
        # Basit benzerlik hesaplama
        actual_words = set(actual.lower().split())
        expected_words = set(expected.lower().split())
        
        if not expected_words:
            return 0.0
        
        intersection = actual_words & expected_words
        union = actual_words | expected_words
        
        jaccard_similarity = len(intersection) / len(union) if union else 0.0
        
        return jaccard_similarity
    
    def _validate_model(self, model_name: str) -> float:
        """Model doğrulama"""
        
        if not self.validation_examples:
            return 0.0
        
        correct_predictions = 0
        total_predictions = 0
        
        for example in self.validation_examples[:10]:  # İlk 10 örnek
            try:
                result = self._train_single_example(model_name, example)
                if result["correct"]:
                    correct_predictions += 1
                total_predictions += 1
            except:
                continue
        
        return correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    def _should_stop_early(self, epoch_results: List[Dict]) -> bool:
        """Early stopping kontrolü"""
        
        if len(epoch_results) < self.training_params["early_stopping_patience"]:
            return False
        
        # Son N epoch'ta iyileşme var mı?
        recent_accuracies = [r["epoch_accuracy"] for r in epoch_results[-self.training_params["early_stopping_patience"]:]]
        
        # Monoton artış kontrolü
        for i in range(1, len(recent_accuracies)):
            if recent_accuracies[i] > recent_accuracies[i-1]:
                return False
        
        return True
    
    def _calculate_final_accuracy(self, epoch_results: List[Dict]) -> float:
        """Final doğruluğu hesaplar"""
        
        if not epoch_results:
            return 0.0
        
        # Son epoch'un doğruluğu
        return epoch_results[-1]["epoch_accuracy"]
    
    def end_training_session(self) -> Dict[str, Any]:
        """Eğitim oturumunu sonlandırır"""
        
        if not self.current_session:
            return {"error": "Aktif eğitim oturumu bulunamadı"}
        
        self.current_session.end_time = datetime.now()
        
        # Oturum özeti
        session_summary = {
            "session_id": self.current_session.session_id,
            "start_time": self.current_session.start_time.isoformat(),
            "end_time": self.current_session.end_time.isoformat(),
            "duration_minutes": (self.current_session.end_time - self.current_session.start_time).total_seconds() / 60,
            "models_trained": self.current_session.models_trained,
            "total_examples": self.current_session.total_examples,
            "performance_summary": self.current_session.performance_metrics
        }
        
        # Oturumu kaydet
        self.training_sessions.append(self.current_session)
        
        print(f"🏁 Eğitim oturumu sonlandırıldı: {self.current_session.session_id}")
        print(f"📊 Eğitilen modeller: {', '.join(self.current_session.models_trained)}")
        
        # Oturum raporu oluştur
        self._export_session_report(session_summary)
        
        self.current_session = None
        return session_summary
    
    def _export_session_report(self, session_summary: Dict) -> bool:
        """Oturum raporunu dışa aktarır"""
        
        try:
            filename = f"training_session_{session_summary['session_id']}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(session_summary, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Oturum raporu kaydedildi: {filename}")
            return True
            
        except Exception as e:
            print(f"❌ Rapor kaydetme hatası: {e}")
            return False
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Eğitim istatistiklerini döndürür"""
        
        stats = {
            "total_sessions": len(self.training_sessions),
            "total_examples": len(self.training_examples),
            "validation_examples": len(self.validation_examples),
            "test_examples": len(self.test_examples),
            "current_session": self.current_session.session_id if self.current_session else None,
            "model_performance": self.model_performance
        }
        
        if self.training_sessions:
            # En iyi performans
            best_session = max(self.training_sessions, key=lambda s: s.accuracy)
            stats["best_session"] = {
                "session_id": best_session.session_id,
                "accuracy": best_session.accuracy,
                "models_trained": best_session.models_trained
            }
        
        return stats
    
    def export_complete_training_report(self, output_file: str = "complete_training_report.json"):
        """Tam eğitim raporunu dışa aktarır"""
        
        try:
            report = {
                "generated_at": datetime.now().isoformat(),
                "training_statistics": self.get_training_statistics(),
                "training_sessions": [
                    {
                        "session_id": session.session_id,
                        "start_time": session.start_time.isoformat(),
                        "end_time": session.end_time.isoformat() if session.end_time else None,
                        "models_trained": session.models_trained,
                        "performance_metrics": session.performance_metrics
                    }
                    for session in self.training_sessions
                ],
                "training_parameters": self.training_params,
                "recommendations": self._generate_training_recommendations()
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Tam eğitim raporu oluşturuldu: {output_file}")
            return True
            
        except Exception as e:
            print(f"❌ Rapor oluşturma hatası: {e}")
            return False
    
    def _generate_training_recommendations(self) -> List[str]:
        """Eğitim önerileri oluşturur"""
        
        recommendations = []
        
        if len(self.training_examples) < 50:
            recommendations.append("Daha fazla eğitim örneği ekleyin (en az 50)")
        
        if len(self.validation_examples) < 10:
            recommendations.append("Doğrulama veri setini genişletin (en az 10 örnek)")
        
        if self.training_params["epochs"] < 20:
            recommendations.append("Epoch sayısını artırın (en az 20)")
        
        recommendations.append("Farklı bitki türleri için özel eğitim verileri ekleyin")
        recommendations.append("Mevsimsel ve bölgesel faktörleri dahil edin")
        recommendations.append("Uzman doğrulaması ile veri kalitesini artırın")
        recommendations.append("Cross-validation kullanarak model performansını değerlendirin")
        
        return recommendations

if __name__ == "__main__":
    # Advanced Trainer'ı test et
    trainer = AdvancedTrainer()
    
    print("🚀 Advanced Trainer başlatıldı!")
    
    # Eğitim veri setini yükle
    if trainer.load_training_dataset():
        print("✅ Eğitim veri seti yüklendi")
        
        # Eğitim oturumu başlat
        session_id = trainer.start_training_session("Test Session")
        
        # Model eğitimi
        training_result = trainer.train_model_with_fine_tuning("claude-3-5-sonnet")
        
        if training_result["success"]:
            print("🎉 Model eğitimi başarılı!")
            
            # Oturumu sonlandır
            session_summary = trainer.end_training_session()
            
            # Tam rapor oluştur
            trainer.export_complete_training_report()
            
            # İstatistikleri göster
            stats = trainer.get_training_statistics()
            print(f"\n📊 Eğitim İstatistikleri:")
            print(f"Toplam Oturum: {stats['total_sessions']}")
            print(f"Toplam Örnek: {stats['total_examples']}")
            print(f"En İyi Doğruluk: {stats.get('best_session', {}).get('accuracy', 0):.3f}")
        else:
            print(f"❌ Model eğitimi başarısız: {training_result.get('error', 'Bilinmeyen hata')}")
    else:
        print("❌ Eğitim veri seti yüklenemedi!")
