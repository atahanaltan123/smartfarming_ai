import json
import os
import time
import requests
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from config import Config
from image_processor import ImageProcessor

@dataclass
class PerformanceMetrics:
    """Model performans metrikleri"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    processing_time: float
    memory_usage: float
    throughput: float  # örnek/saniye
    latency: float  # milisaniye
    error_rate: float
    confidence_distribution: Dict[str, float]

@dataclass
class EvaluationResult:
    """Değerlendirme sonucu"""
    model_name: str
    evaluation_timestamp: datetime
    test_dataset_size: int
    performance_metrics: PerformanceMetrics
    confusion_matrix: Dict[str, Dict[str, int]]
    error_analysis: List[Dict[str, Any]]
    recommendations: List[str]

class ModelEvaluator:
    """Model performans değerlendirme ve optimizasyon sistemi"""
    
    def __init__(self):
        self.api_key = Config.get_claude_api_key()
        self.api_url = Config.CLAUDE_API_URL
        self.image_processor = ImageProcessor()
        
        # Test veri seti
        self.test_dataset = []
        self.ground_truth = {}
        
        # Performans geçmişi
        self.performance_history = {}
        
        # Değerlendirme parametreleri
        self.evaluation_params = {
            "confidence_thresholds": [0.5, 0.6, 0.7, 0.8, 0.9],
            "batch_sizes": [1, 5, 10, 20],
            "timeout_seconds": 30,
            "max_retries": 3
        }
    
    def load_test_dataset(self, dataset_file: str = "test_dataset.json") -> bool:
        """Test veri setini yükler"""
        
        try:
            if os.path.exists(dataset_file):
                with open(dataset_file, 'r', encoding='utf-8') as f:
                    self.test_dataset = json.load(f)
                
                # Ground truth oluştur
                self._create_ground_truth()
                
                print(f"✅ Test veri seti yüklendi: {len(self.test_dataset)} örnek")
                return True
            else:
                print(f"❌ Test veri seti bulunamadı: {dataset_file}")
                return False
                
        except Exception as e:
            print(f"❌ Test veri seti yükleme hatası: {e}")
            return False
    
    def _create_ground_truth(self):
        """Ground truth verilerini oluşturur"""
        
        self.ground_truth = {}
        
        for example in self.test_dataset:
            example_id = example.get("id", f"example_{len(self.ground_truth)}")
            
            # Beklenen sonuçları çıkar
            expected_response = example.get("messages", [{}])[1].get("content", "")
            
            # Basit parsing
            ground_truth = {
                "disease": self._extract_disease_from_response(expected_response),
                "severity": self._extract_severity_from_response(expected_response),
                "disease_type": self._extract_disease_type_from_response(expected_response),
                "confidence": 0.95,  # Varsayılan güvenilirlik
                "response_text": expected_response
            }
            
            self.ground_truth[example_id] = ground_truth
    
    def _extract_disease_from_response(self, response: str) -> str:
        """Yanıttan hastalık adını çıkarır"""
        
        disease_keywords = [
            "erken yaprak yanıklığı", "külleme", "bakteriyel leke",
            "late blight", "powdery mildew", "bacterial spot"
        ]
        
        for keyword in disease_keywords:
            if keyword.lower() in response.lower():
                return keyword
        
        return "Bilinmeyen"
    
    def _extract_severity_from_response(self, response: str) -> str:
        """Yanıttan şiddet seviyesini çıkarır"""
        
        severity_keywords = {
            "düşük": ["düşük", "low", "hafif", "mild"],
            "orta": ["orta", "medium", "moderate"],
            "yüksek": ["yüksek", "high", "şiddetli", "severe"],
            "kritik": ["kritik", "critical", "çok şiddetli"]
        }
        
        for severity, keywords in severity_keywords.items():
            if any(keyword in response.lower() for keyword in keywords):
                return severity
        
        return "Bilinmeyen"
    
    def _extract_disease_type_from_response(self, response: str) -> str:
        """Yanıttan hastalık türünü çıkarır"""
        
        type_keywords = {
            "fungal": ["fungal", "mantar", "fungus", "mikoz"],
            "bacterial": ["bakteriyel", "bacterial", "bakteri"],
            "viral": ["viral", "virus", "virüs"],
            "pest": ["zararlı", "pest", "böcek", "insect"]
        }
        
        for disease_type, keywords in type_keywords.items():
            if any(keyword in response.lower() for keyword in keywords):
                return disease_type
        
        return "Bilinmeyen"
    
    def evaluate_model_performance(self, model_name: str, 
                                 test_subset_size: int = None) -> EvaluationResult:
        """Model performansını değerlendirir"""
        
        if not self.test_dataset:
            raise ValueError("Test veri seti yüklenmedi")
        
        print(f"🔍 {model_name} modeli değerlendiriliyor...")
        
        # Test alt kümesi seç
        test_subset = self.test_dataset[:test_subset_size] if test_subset_size else self.test_dataset
        
        # Performans metrikleri
        start_time = time.time()
        predictions = []
        errors = []
        
        for i, example in enumerate(test_subset):
            try:
                print(f"  📊 Örnek {i+1}/{len(test_subset)} değerlendiriliyor...")
                
                # Model tahmini
                prediction = self._get_model_prediction(model_name, example)
                
                if prediction:
                    predictions.append(prediction)
                else:
                    errors.append({
                        "example_id": example.get("id", f"example_{i}"),
                        "error": "Tahmin alınamadı",
                        "timestamp": datetime.now().isoformat()
                    })
                
            except Exception as e:
                errors.append({
                    "example_id": example.get("id", f"example_{i}"),
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        # Performans hesaplama
        total_time = time.time() - start_time
        performance_metrics = self._calculate_performance_metrics(predictions, total_time)
        
        # Confusion matrix
        confusion_matrix = self._create_confusion_matrix(predictions)
        
        # Hata analizi
        error_analysis = self._analyze_errors(errors, predictions)
        
        # Öneriler
        recommendations = self._generate_recommendations(performance_metrics, error_analysis)
        
        # Sonuç oluştur
        evaluation_result = EvaluationResult(
            model_name=model_name,
            evaluation_timestamp=datetime.now(),
            test_dataset_size=len(test_subset),
            performance_metrics=performance_metrics,
            confusion_matrix=confusion_matrix,
            error_analysis=error_analysis,
            recommendations=recommendations
        )
        
        # Performans geçmişini güncelle
        self._update_performance_history(model_name, evaluation_result)
        
        print(f"✅ {model_name} değerlendirmesi tamamlandı!")
        print(f"📊 Doğruluk: {performance_metrics.accuracy:.3f}")
        print(f"⏱️ Toplam süre: {total_time:.2f} saniye")
        
        return evaluation_result
    
    def _get_model_prediction(self, model_name: str, example: Dict) -> Optional[Dict]:
        """Model tahmini alır"""
        
        try:
            # API isteği
            message = {
                "model": model_name,
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": example["messages"][0]["content"][0]["text"]
                    }
                ]
            }
            
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01"
            }
            
            start_time = time.time()
            response = requests.post(self.api_url, headers=headers, json=message, 
                                  timeout=self.evaluation_params["timeout_seconds"])
            processing_time = (time.time() - start_time) * 1000  # milisaniye
            
            if response.status_code == 200:
                result = response.json()
                prediction_text = result["content"][0]["text"]
                
                # Tahmin sonucunu parse et
                prediction = {
                    "disease": self._extract_disease_from_response(prediction_text),
                    "severity": self._extract_severity_from_response(prediction_text),
                    "disease_type": self._extract_disease_type_from_response(prediction_text),
                    "confidence": self._extract_confidence_from_response(prediction_text),
                    "response_text": prediction_text,
                    "processing_time": processing_time,
                    "timestamp": datetime.now().isoformat()
                }
                
                return prediction
            else:
                print(f"    ❌ API Hatası: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"    ❌ Tahmin hatası: {e}")
            return None
    
    def _extract_confidence_from_response(self, response: str) -> float:
        """Yanıttan güvenilirlik skorunu çıkarır"""
        
        import re
        
        # Yüzde işareti ile güvenilirlik
        confidence_match = re.search(r'(\d+)%', response)
        if confidence_match:
            return float(confidence_match.group(1)) / 100
        
        # "Güvenilirlik" kelimesi ile
        confidence_keywords = ["güvenilirlik", "confidence", "güven"]
        for keyword in confidence_keywords:
            if keyword in response.lower():
                # Sayısal değer ara
                numbers = re.findall(r'\d+', response)
                if numbers:
                    return min(float(numbers[0]) / 100, 1.0)
        
        return 0.8  # Varsayılan güvenilirlik
    
    def _calculate_performance_metrics(self, predictions: List[Dict], total_time: float) -> PerformanceMetrics:
        """Performans metriklerini hesaplar"""
        
        if not predictions:
            return PerformanceMetrics(
                accuracy=0.0, precision=0.0, recall=0.0, f1_score=0.0,
                processing_time=total_time, memory_usage=0.0, throughput=0.0,
                latency=0.0, error_rate=1.0, confidence_distribution={}
            )
        
        # Doğruluk hesaplama
        correct_predictions = 0
        total_predictions = len(predictions)
        
        for pred in predictions:
            # Ground truth ile karşılaştır
            example_id = pred.get("example_id", "unknown")
            if example_id in self.ground_truth:
                gt = self.ground_truth[example_id]
                
                # Hastalık tespiti doğruluğu
                if pred["disease"] == gt["disease"]:
                    correct_predictions += 1
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        # Precision, Recall, F1-Score
        true_positives = correct_predictions
        false_positives = total_predictions - correct_predictions
        false_negatives = len(self.ground_truth) - correct_predictions
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Zaman metrikleri
        avg_processing_time = sum(p["processing_time"] for p in predictions) / len(predictions)
        throughput = len(predictions) / total_time if total_time > 0 else 0.0
        
        # Güvenilirlik dağılımı
        confidence_scores = [p["confidence"] for p in predictions]
        confidence_distribution = {
            "0.0-0.2": len([c for c in confidence_scores if 0.0 <= c < 0.2]),
            "0.2-0.4": len([c for c in confidence_scores if 0.2 <= c < 0.4]),
            "0.4-0.6": len([c for c in confidence_scores if 0.4 <= c < 0.6]),
            "0.6-0.8": len([c for c in confidence_scores if 0.6 <= c < 0.8]),
            "0.8-1.0": len([c for c in confidence_scores if 0.8 <= c <= 1.0])
        }
        
        return PerformanceMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            processing_time=total_time,
            memory_usage=0.0,  # API kullanımında ölçülemez
            throughput=throughput,
            latency=avg_processing_time,
            error_rate=1.0 - accuracy,
            confidence_distribution=confidence_distribution
        )
    
    def _create_confusion_matrix(self, predictions: List[Dict]) -> Dict[str, Dict[str, int]]:
        """Confusion matrix oluşturur"""
        
        confusion_matrix = {
            "disease": {"TP": 0, "FP": 0, "TN": 0, "FN": 0},
            "severity": {"TP": 0, "FP": 0, "TN": 0, "FN": 0},
            "disease_type": {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
        }
        
        for pred in predictions:
            example_id = pred.get("example_id", "unknown")
            if example_id in self.ground_truth:
                gt = self.ground_truth[example_id]
                
                # Hastalık tespiti
                if pred["disease"] == gt["disease"]:
                    confusion_matrix["disease"]["TP"] += 1
                else:
                    confusion_matrix["disease"]["FP"] += 1
                    confusion_matrix["disease"]["FN"] += 1
                
                # Şiddet seviyesi
                if pred["severity"] == gt["severity"]:
                    confusion_matrix["severity"]["TP"] += 1
                else:
                    confusion_matrix["severity"]["FP"] += 1
                    confusion_matrix["severity"]["FN"] += 1
                
                # Hastalık türü
                if pred["disease_type"] == gt["disease_type"]:
                    confusion_matrix["disease_type"]["TP"] += 1
                else:
                    confusion_matrix["disease_type"]["FP"] += 1
                    confusion_matrix["disease_type"]["FN"] += 1
        
        return confusion_matrix
    
    def _analyze_errors(self, errors: List[Dict], predictions: List[Dict]) -> List[Dict]:
        """Hataları analiz eder"""
        
        error_analysis = []
        
        # API hataları
        api_errors = [e for e in errors if "API Hatası" in e.get("error", "")]
        if api_errors:
            error_analysis.append({
                "error_type": "API Errors",
                "count": len(api_errors),
                "percentage": len(api_errors) / len(errors) if errors else 0,
                "examples": api_errors[:3]  # İlk 3 örnek
            })
        
        # Zaman aşımı hataları
        timeout_errors = [e for e in errors if "timeout" in e.get("error", "").lower()]
        if timeout_errors:
            error_analysis.append({
                "error_type": "Timeout Errors",
                "count": len(timeout_errors),
                "percentage": len(timeout_errors) / len(errors) if errors else 0,
                "examples": timeout_errors[:3]
            })
        
        # Tahmin hataları
        prediction_errors = []
        for pred in predictions:
            example_id = pred.get("example_id", "unknown")
            if example_id in self.ground_truth:
                gt = self.ground_truth[example_id]
                if pred["disease"] != gt["disease"]:
                    prediction_errors.append({
                        "example_id": example_id,
                        "predicted": pred["disease"],
                        "expected": gt["disease"],
                        "confidence": pred["confidence"]
                    })
        
        if prediction_errors:
            error_analysis.append({
                "error_type": "Prediction Errors",
                "count": len(prediction_errors),
                "percentage": len(prediction_errors) / len(predictions) if predictions else 0,
                "examples": prediction_errors[:3]
            })
        
        return error_analysis
    
    def _generate_recommendations(self, metrics: PerformanceMetrics, 
                                error_analysis: List[Dict]) -> List[str]:
        """İyileştirme önerileri oluşturur"""
        
        recommendations = []
        
        # Doğruluk önerileri
        if metrics.accuracy < 0.8:
            recommendations.append("Model doğruluğunu artırmak için daha fazla eğitim verisi ekleyin")
        
        if metrics.f1_score < 0.7:
            recommendations.append("F1-score düşük. Precision ve recall dengesini iyileştirin")
        
        # Performans önerileri
        if metrics.latency > 5000:  # 5 saniye
            recommendations.append("Model yanıt süresi yavaş. API optimizasyonu gerekli")
        
        if metrics.throughput < 0.1:  # 0.1 örnek/saniye
            recommendations.append("Model throughput düşük. Batch processing kullanın")
        
        # Hata analizi önerileri
        for error in error_analysis:
            if error["error_type"] == "API Errors" and error["percentage"] > 0.1:
                recommendations.append("API hata oranı yüksek. Rate limiting ve retry mekanizması ekleyin")
            
            if error["error_type"] == "Timeout Errors" and error["percentage"] > 0.05:
                recommendations.append("Zaman aşımı hataları fazla. Timeout süresini artırın")
            
            if error["error_type"] == "Prediction Errors" and error["percentage"] > 0.2:
                recommendations.append("Tahmin hata oranı yüksek. Model eğitimini iyileştirin")
        
        # Genel öneriler
        if not recommendations:
            recommendations.append("Model performansı iyi. Düzenli monitoring yapın")
        
        recommendations.append("Cross-validation ile model performansını değerlendirin")
        recommendations.append("A/B testing ile farklı model versiyonlarını karşılaştırın")
        
        return recommendations
    
    def _update_performance_history(self, model_name: str, result: EvaluationResult):
        """Performans geçmişini günceller"""
        
        if model_name not in self.performance_history:
            self.performance_history[model_name] = []
        
        self.performance_history[model_name].append({
            "timestamp": result.evaluation_timestamp.isoformat(),
            "accuracy": result.performance_metrics.accuracy,
            "f1_score": result.performance_metrics.f1_score,
            "processing_time": result.performance_metrics.processing_time,
            "test_dataset_size": result.test_dataset_size
        })
        
        # Geçmişi sınırla (son 10 değerlendirme)
        if len(self.performance_history[model_name]) > 10:
            self.performance_history[model_name] = self.performance_history[model_name][-10:]
    
    def get_performance_trends(self, model_name: str) -> Dict[str, Any]:
        """Model performans trendlerini analiz eder"""
        
        if model_name not in self.performance_history:
            return {"error": "Model performans geçmişi bulunamadı"}
        
        history = self.performance_history[model_name]
        
        if len(history) < 2:
            return {"error": "Trend analizi için yeterli veri yok"}
        
        # Trend hesaplama
        accuracies = [h["accuracy"] for h in history]
        f1_scores = [h["f1_score"] for h in history]
        processing_times = [h["processing_time"] for h in history]
        
        # Basit trend analizi
        accuracy_trend = "artış" if accuracies[-1] > accuracies[0] else "azalış" if accuracies[-1] < accuracies[0] else "stabil"
        f1_trend = "artış" if f1_scores[-1] > f1_scores[0] else "azalış" if f1_scores[-1] < f1_scores[0] else "stabil"
        time_trend = "iyileşme" if processing_times[-1] < processing_times[0] else "kötüleşme" if processing_times[-1] > processing_times[0] else "stabil"
        
        trends = {
            "model_name": model_name,
            "total_evaluations": len(history),
            "accuracy_trend": accuracy_trend,
            "f1_score_trend": f1_trend,
            "processing_time_trend": time_trend,
            "current_metrics": {
                "accuracy": accuracies[-1],
                "f1_score": f1_scores[-1],
                "processing_time": processing_times[-1]
            },
            "best_metrics": {
                "accuracy": max(accuracies),
                "f1_score": max(f1_scores),
                "processing_time": min(processing_times)
            },
            "average_metrics": {
                "accuracy": sum(accuracies) / len(accuracies),
                "f1_score": sum(f1_scores) / len(f1_scores),
                "processing_time": sum(processing_times) / len(processing_times)
            }
        }
        
        return trends
    
    def export_evaluation_report(self, result: EvaluationResult, 
                               output_file: str = None) -> bool:
        """Değerlendirme raporunu dışa aktarır"""
        
        try:
            if not output_file:
                output_file = f"evaluation_report_{result.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Dataclass'ları dict'e çevir
            export_data = {
                "model_name": result.model_name,
                "evaluation_timestamp": result.evaluation_timestamp.isoformat(),
                "test_dataset_size": result.test_dataset_size,
                "performance_metrics": {
                    "accuracy": result.performance_metrics.accuracy,
                    "precision": result.performance_metrics.precision,
                    "recall": result.performance_metrics.recall,
                    "f1_score": result.performance_metrics.f1_score,
                    "processing_time": result.performance_metrics.processing_time,
                    "throughput": result.performance_metrics.throughput,
                    "latency": result.performance_metrics.latency,
                    "error_rate": result.performance_metrics.error_rate,
                    "confidence_distribution": result.performance_metrics.confidence_distribution
                },
                "confusion_matrix": result.confusion_matrix,
                "error_analysis": result.error_analysis,
                "recommendations": result.recommendations
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Değerlendirme raporu kaydedildi: {output_file}")
            return True
            
        except Exception as e:
            print(f"❌ Rapor kaydetme hatası: {e}")
            return False
    
    def compare_models(self, model_names: List[str]) -> Dict[str, Any]:
        """Birden fazla modeli karşılaştırır"""
        
        comparison = {
            "comparison_timestamp": datetime.now().isoformat(),
            "models_compared": model_names,
            "performance_comparison": {},
            "ranking": {},
            "recommendations": []
        }
        
        for model_name in model_names:
            if model_name in self.performance_history and self.performance_history[model_name]:
                latest = self.performance_history[model_name][-1]
                comparison["performance_comparison"][model_name] = {
                    "accuracy": latest["accuracy"],
                    "f1_score": latest["f1_score"],
                    "processing_time": latest["processing_time"]
                }
        
        # Model sıralaması
        if comparison["performance_comparison"]:
            # Doğruluk bazında sıralama
            accuracy_ranking = sorted(
                comparison["performance_comparison"].items(),
                key=lambda x: x[1]["accuracy"],
                reverse=True
            )
            
            comparison["ranking"]["by_accuracy"] = [model for model, _ in accuracy_ranking]
            
            # F1-score bazında sıralama
            f1_ranking = sorted(
                comparison["performance_comparison"].items(),
                key=lambda x: x[1]["f1_score"],
                reverse=True
            )
            
            comparison["ranking"]["by_f1_score"] = [model for model, _ in f1_ranking]
            
            # En iyi model önerisi
            best_model = accuracy_ranking[0][0]
            comparison["recommendations"].append(f"En iyi performans: {best_model}")
            
            # İyileştirme önerileri
            for model_name, metrics in comparison["performance_comparison"].items():
                if metrics["accuracy"] < 0.8:
                    comparison["recommendations"].append(f"{model_name}: Doğruluk iyileştirmesi gerekli")
                if metrics["processing_time"] > 10:
                    comparison["recommendations"].append(f"{model_name}: Performans optimizasyonu gerekli")
        
        return comparison

if __name__ == "__main__":
    # Model Evaluator'ı test et
    evaluator = ModelEvaluator()
    
    print("🚀 Model Evaluator başlatıldı!")
    
    # Test veri setini yükle
    if evaluator.load_test_dataset():
        print("✅ Test veri seti yüklendi")
        
        # Model değerlendirmesi
        try:
            result = evaluator.evaluate_model_performance("claude-3-5-sonnet", test_subset_size=5)
            
            # Rapor oluştur
            evaluator.export_evaluation_report(result)
            
            # Performans trendleri
            trends = evaluator.get_performance_trends("claude-3-5-sonnet")
            print(f"\n📈 Performans Trendleri:")
            print(f"Doğruluk Trendi: {trends.get('accuracy_trend', 'N/A')}")
            print(f"F1-Score Trendi: {trends.get('f1_score_trend', 'N/A')}")
            
        except Exception as e:
            print(f"❌ Değerlendirme hatası: {e}")
    else:
        print("❌ Test veri seti yüklenemedi!")
