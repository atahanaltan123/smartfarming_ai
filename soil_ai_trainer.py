import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
from datetime import datetime

class SoilAITrainer:
    def __init__(self, data_path="training_data/soil_analysis_dataset.json"):
        self.data_path = data_path
        self.dataset = None
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.performance_metrics = {}
        
    def load_dataset(self):
        """Veri setini yükle"""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                self.dataset = json.load(f)
            print(f"✅ Veri seti yüklendi: {len(self.dataset['soil_samples'])} örnek")
            return True
        except Exception as e:
            print(f"❌ Veri seti yüklenemedi: {e}")
            return False
    
    def prepare_training_data(self):
        """Eğitim verilerini hazırla"""
        if not self.dataset:
            print("❌ Önce veri seti yüklenmeli")
            return None, None
        
        # Veri setini DataFrame'e çevir
        samples = []
        for sample in self.dataset['soil_samples']:
            row = {
                'region': sample['region'],
                'climate': sample['climate'],
                'irrigation': sample['irrigation'],
                'pH': sample['soil_values']['pH'],
                'organic_matter': sample['soil_values']['organic_matter'],
                'nitrogen': sample['soil_values']['nitrogen'],
                'phosphorus': sample['soil_values']['phosphorus'],
                'potassium': sample['soil_values']['potassium'],
                'calcium': sample['soil_values']['calcium'],
                'magnesium': sample['soil_values']['magnesium'],
                'iron': sample['soil_values']['iron'],
                'zinc': sample['soil_values']['zinc']
            }
            
            # Her ürün için ayrı satır oluştur
            for crop, success_data in sample['crop_success'].items():
                crop_row = row.copy()
                crop_row['crop'] = crop
                crop_row['yield_level'] = success_data['yield']
                crop_row['quality'] = success_data['quality']
                crop_row['success_rate'] = success_data['success_rate']
                samples.append(crop_row)
        
        df = pd.DataFrame(samples)
        print(f"✅ Eğitim verisi hazırlandı: {len(df)} satır")
        
        # Kategorik değişkenleri encode et
        categorical_cols = ['region', 'climate', 'irrigation', 'crop', 'yield_level', 'quality']
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
        
        # Özellikler ve hedef değişkenler
        feature_cols = ['region', 'climate', 'irrigation', 'pH', 'organic_matter', 
                       'nitrogen', 'phosphorus', 'potassium', 'calcium', 'magnesium', 'iron', 'zinc']
        
        X = df[feature_cols]
        y_yield = df['yield_level']
        y_quality = df['quality']
        y_success = df['success_rate']
        
        return X, y_yield, y_quality, y_success
    
    def train_models(self):
        """AI modellerini eğit"""
        X, y_yield, y_quality, y_success = self.prepare_training_data()
        
        if X is None:
            return False
        
        # Veriyi eğitim ve test olarak böl
        X_train, X_test, y_yield_train, y_yield_test = train_test_split(
            X, y_yield, test_size=0.2, random_state=42
        )
        _, _, y_quality_train, y_quality_test = train_test_split(
            X, y_quality, test_size=0.2, random_state=42
        )
        _, _, y_success_train, y_success_test = train_test_split(
            X, y_success, test_size=0.2, random_state=42
        )
        
        # Veriyi ölçeklendir
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['features'] = scaler
        
        print("🚀 Model eğitimi başlıyor...")
        
        # 1. Verim Tahmin Modeli (Classification)
        print("📊 Verim tahmin modeli eğitiliyor...")
        yield_model = RandomForestClassifier(n_estimators=100, random_state=42)
        yield_model.fit(X_train_scaled, y_yield_train)
        self.models['yield_predictor'] = yield_model
        
        # 2. Kalite Tahmin Modeli (Classification)
        print("⭐ Kalite tahmin modeli eğitiliyor...")
        quality_model = RandomForestClassifier(n_estimators=100, random_state=42)
        quality_model.fit(X_train_scaled, y_quality_train)
        self.models['quality_predictor'] = quality_model
        
        # 3. Başarı Oranı Tahmin Modeli (Regression)
        print("📈 Başarı oranı tahmin modeli eğitiliyor...")
        success_model = RandomForestRegressor(n_estimators=100, random_state=42)
        success_model.fit(X_train_scaled, y_success_train)
        self.models['success_predictor'] = success_model
        
        # Model performanslarını değerlendir
        self.evaluate_models(X_test_scaled, y_yield_test, y_quality_test, y_success_test)
        
        return True
    
    def evaluate_models(self, X_test, y_yield_test, y_quality_test, y_success_test):
        """Model performanslarını değerlendir"""
        print("\n📊 Model Performans Değerlendirmesi:")
        print("=" * 50)
        
        # Verim tahmin modeli
        yield_pred = self.models['yield_predictor'].predict(X_test)
        yield_accuracy = accuracy_score(y_yield_test, yield_pred)
        print(f"🌾 Verim Tahmin Modeli:")
        print(f"   Doğruluk: {yield_accuracy:.3f} ({yield_accuracy*100:.1f}%)")
        
        # Kalite tahmin modeli
        quality_pred = self.models['quality_predictor'].predict(X_test)
        quality_accuracy = accuracy_score(y_quality_test, quality_pred)
        print(f"⭐ Kalite Tahmin Modeli:")
        print(f"   Doğruluk: {quality_accuracy:.3f} ({quality_accuracy*100:.1f}%)")
        
        # Başarı oranı tahmin modeli
        success_pred = self.models['success_predictor'].predict(X_test)
        success_mse = mean_squared_error(y_success_test, success_pred)
        success_r2 = r2_score(y_success_test, success_pred)
        print(f"📈 Başarı Oranı Tahmin Modeli:")
        print(f"   MSE: {success_mse:.4f}")
        print(f"   R²: {success_r2:.3f} ({success_r2*100:.1f}%)")
        
        # Cross-validation scores (only if we have enough data)
        print("\n🔄 Cross-Validation Sonuçları:")
        min_samples_for_cv = 10
        if len(X_test) >= min_samples_for_cv:
            for name, model in self.models.items():
                if 'predictor' in name:
                    try:
                        if 'success' in name:
                            # Regression model
                            cv_scores = cross_val_score(model, X_test, y_success_test, cv=min(3, len(X_test)//2), scoring='r2')
                            print(f"   {name}: R² = {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
                        else:
                            # Classification model
                            cv_scores = cross_val_score(model, X_test, 
                                                     y_yield_test if 'yield' in name else y_quality_test, 
                                                     cv=min(3, len(X_test)//2), scoring='accuracy')
                            print(f"   {name}: Accuracy = {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
                    except Exception as e:
                        print(f"   {name}: Cross-validation hatası - {e}")
        else:
            print(f"   ⚠️ Cross-validation için yeterli veri yok (minimum {min_samples_for_cv} gerekli, mevcut: {len(X_test)})")
        
        # Performans metriklerini sakla
        self.performance_metrics = {
            'yield_accuracy': yield_accuracy,
            'quality_accuracy': quality_accuracy,
            'success_mse': success_mse,
            'success_r2': success_r2,
            'test_samples': len(X_test)
        }
    
    def predict_crop_success(self, soil_data):
        """Yeni toprak verisi için ürün başarısını tahmin et"""
        if not self.models:
            print("❌ Modeller henüz eğitilmemiş")
            return None
        
        try:
            # Veriyi hazırla
            features = np.array([
                soil_data['region'],
                soil_data['climate'],
                soil_data['irrigation'],
                soil_data['pH'],
                soil_data['organic_matter'],
                soil_data['nitrogen'],
                soil_data['phosphorus'],
                soil_data['potassium'],
                soil_data['calcium'],
                soil_data['magnesium'],
                soil_data['iron'],
                soil_data['zinc']
            ]).reshape(1, -1)
            
            # Ölçeklendir
            features_scaled = self.scalers['features'].transform(features)
            
            # Tahminler yap
            yield_pred = self.models['yield_predictor'].predict(features_scaled)[0]
            quality_pred = self.models['quality_predictor'].predict(features_scaled)[0]
            success_pred = self.models['success_predictor'].predict(features_scaled)[0]
            
            # Sonuçları decode et
            yield_level = self.label_encoders['yield_level'].inverse_transform([yield_pred])[0]
            quality = self.label_encoders['quality'].inverse_transform([quality_pred])[0]
            
            return {
                'yield_level': yield_level,
                'quality': quality,
                'success_rate': float(success_pred),
                'confidence': self.calculate_confidence(success_pred)
            }
            
        except Exception as e:
            print(f"❌ Tahmin hatası: {e}")
            return None
    
    def calculate_confidence(self, success_rate):
        """Başarı oranına göre güven skoru hesapla"""
        if success_rate >= 0.8:
            return "Yüksek"
        elif success_rate >= 0.6:
            return "Orta"
        elif success_rate >= 0.4:
            return "Düşük"
        else:
            return "Çok Düşük"
    
    def save_models(self, save_dir="trained_models"):
        """Eğitilmiş modelleri kaydet"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Modelleri kaydet
        for name, model in self.models.items():
            model_path = os.path.join(save_dir, f"{name}_{timestamp}.joblib")
            joblib.dump(model, model_path)
            print(f"💾 Model kaydedildi: {model_path}")
        
        # Scaler'ları kaydet
        for name, scaler in self.scalers.items():
            scaler_path = os.path.join(save_dir, f"{name}_scaler_{timestamp}.joblib")
            joblib.dump(scaler, scaler_path)
            print(f"💾 Scaler kaydedildi: {scaler_path}")
        
        # Label encoder'ları kaydet
        for name, encoder in self.label_encoders.items():
            encoder_path = os.path.join(save_dir, f"{name}_encoder_{timestamp}.joblib")
            joblib.dump(encoder, encoder_path)
            print(f"💾 Encoder kaydedildi: {encoder_path}")
        
        # Performans metriklerini kaydet
        metrics_path = os.path.join(save_dir, f"performance_metrics_{timestamp}.json")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(self.performance_metrics, f, indent=2, ensure_ascii=False)
        print(f"💾 Performans metrikleri kaydedildi: {metrics_path}")
    
    def generate_test_report(self):
        """Test raporu oluştur"""
        if not self.performance_metrics:
            print("❌ Performans metrikleri mevcut değil")
            return
        
        report = {
            "model_training_summary": {
                "timestamp": datetime.now().isoformat(),
                "total_models": len(self.models),
                "test_samples": self.performance_metrics['test_samples']
            },
            "model_performance": {
                "yield_predictor": {
                    "type": "Classification",
                    "metric": "Accuracy",
                    "value": self.performance_metrics['yield_accuracy'],
                    "percentage": f"{self.performance_metrics['yield_accuracy']*100:.1f}%"
                },
                "quality_predictor": {
                    "type": "Classification",
                    "metric": "Accuracy",
                    "value": self.performance_metrics['quality_accuracy'],
                    "percentage": f"{self.performance_metrics['quality_accuracy']*100:.1f}%"
                },
                "success_predictor": {
                    "type": "Regression",
                    "metric": "R² Score",
                    "value": self.performance_metrics['success_r2'],
                    "percentage": f"{self.performance_metrics['success_r2']*100:.1f}%"
                }
            },
            "recommendations": {
                "model_quality": "Yüksek" if self.performance_metrics['yield_accuracy'] > 0.8 else "Orta",
                "data_quality": "İyi" if self.performance_metrics['test_samples'] > 50 else "Geliştirilebilir",
                "next_steps": [
                    "Daha fazla toprak analiz verisi toplanmalı",
                    "Farklı bölgelerden veri eklenmeli",
                    "Model hiperparametreleri optimize edilmeli"
                ]
            }
        }
        
        # Raporu kaydet
        report_path = f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📋 Test raporu oluşturuldu: {report_path}")
        return report

def main():
    """Ana fonksiyon"""
    print("🌱 AgriSoilTech - Toprak AI Model Eğitimi")
    print("=" * 50)
    
    # Trainer'ı başlat
    trainer = SoilAITrainer()
    
    # Veri setini yükle
    if not trainer.load_dataset():
        return
    
    # Modelleri eğit
    if not trainer.train_models():
        return
    
    # Modelleri kaydet
    trainer.save_models()
    
    # Test raporu oluştur
    trainer.generate_test_report()
    
    # Örnek tahmin yap
    print("\n🧪 Örnek Tahmin Testi:")
    print("-" * 30)
    
    test_soil = {
        'region': 0,  # Marmara
        'climate': 0,  # Mediterranean
        'irrigation': 1,  # Irrigated
        'pH': 6.8,
        'organic_matter': 2.1,
        'nitrogen': 15,
        'phosphorus': 25,
        'potassium': 180,
        'calcium': 1200,
        'magnesium': 150,
        'iron': 8,
        'zinc': 1.2
    }
    
    prediction = trainer.predict_crop_success(test_soil)
    if prediction:
        print(f"🌾 Tahmin Sonucu:")
        print(f"   Verim Seviyesi: {prediction['yield_level']}")
        print(f"   Kalite: {prediction['quality']}")
        print(f"   Başarı Oranı: {prediction['success_rate']:.3f}")
        print(f"   Güven: {prediction['confidence']}")
    
    print("\n✅ Model eğitimi tamamlandı!")

if __name__ == "__main__":
    main()
