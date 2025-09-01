import os
import json
import csv
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass, asdict

@dataclass
class PlantDiseaseRecord:
    """Bitki hastalık kaydı veri yapısı"""
    id: str
    plant_type: str
    disease_name: str
    disease_type: str  # fungal, bacterial, viral, pest
    symptoms: List[str]
    severity: str  # low, medium, high, critical
    treatment_methods: List[str]
    prevention_methods: List[str]
    image_paths: List[str]
    climate_conditions: Dict[str, Any]
    soil_conditions: Dict[str, Any]
    season: str
    region: str
    expert_notes: str
    confidence_score: float
    created_date: str
    verified_by: str

class DataCollector:
    """Bitki hastalık veri seti toplayıcı sınıfı"""
    
    def __init__(self, data_dir: str = "training_data"):
        self.data_dir = data_dir
        self.records_file = os.path.join(data_dir, "plant_diseases.json")
        self.csv_file = os.path.join(data_dir, "plant_diseases.csv")
        self.images_dir = os.path.join(data_dir, "images")
        
        # Klasörleri oluştur
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Mevcut verileri yükle
        self.records = self.load_existing_records()
    
    def load_existing_records(self) -> List[PlantDiseaseRecord]:
        """Mevcut veri kayıtlarını yükler"""
        if os.path.exists(self.records_file):
            try:
                with open(self.records_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return [PlantDiseaseRecord(**record) for record in data]
            except Exception as e:
                print(f"Veri yükleme hatası: {e}")
        return []
    
    def add_record(self, record: PlantDiseaseRecord) -> bool:
        """Yeni hastalık kaydı ekler"""
        try:
            # ID kontrolü
            if any(r.id == record.id for r in self.records):
                print(f"ID {record.id} zaten mevcut!")
                return False
            
            self.records.append(record)
            self.save_records()
            print(f"✅ Kayıt eklendi: {record.plant_type} - {record.disease_name}")
            return True
            
        except Exception as e:
            print(f"Kayıt ekleme hatası: {e}")
            return False
    
    def save_records(self):
        """Kayıtları dosyaya kaydeder"""
        try:
            # JSON formatında kaydet
            with open(self.records_file, 'w', encoding='utf-8') as f:
                json.dump([asdict(record) for record in self.records], f, 
                         ensure_ascii=False, indent=2)
            
            # CSV formatında da kaydet
            self.save_to_csv()
            
        except Exception as e:
            print(f"Kaydetme hatası: {e}")
    
    def save_to_csv(self):
        """Kayıtları CSV formatında kaydeder"""
        try:
            if not self.records:
                return
            
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Başlık satırı
                headers = list(asdict(self.records[0]).keys())
                writer.writerow(headers)
                
                # Veri satırları
                for record in self.records:
                    row = [getattr(record, header) for header in headers]
                    writer.writerow(row)
                    
        except Exception as e:
            print(f"CSV kaydetme hatası: {e}")
    
    def create_sample_dataset(self):
        """Örnek veri seti oluşturur"""
        sample_records = [
            PlantDiseaseRecord(
                id="PD001",
                plant_type="Domates",
                disease_name="Erken Yaprak Yanıklığı",
                disease_type="fungal",
                symptoms=["Kahverengi lekeler", "Yaprak dökülmesi", "Gövde lezyonları"],
                severity="high",
                treatment_methods=["Fungisit uygulaması", "Hastalıklı yaprakların temizlenmesi"],
                prevention_methods=["Havalandırma", "Sulama kontrolü", "Düzenli gözlem"],
                image_paths=["domates_erken_yanik_1.jpg"],
                climate_conditions={"temperature": "20-30°C", "humidity": ">80%"},
                soil_conditions={"ph": "6.0-7.0", "drainage": "good"},
                season="yaz",
                region="Akdeniz",
                expert_notes="Alternaria solani mantarı neden olur",
                confidence_score=0.95,
                created_date=datetime.now().isoformat(),
                verified_by="Dr. Tarım Uzmanı"
            ),
            PlantDiseaseRecord(
                id="PD002",
                plant_type="Salatalık",
                disease_name="Külleme",
                disease_type="fungal",
                symptoms=["Beyaz pudra benzeri leke", "Yaprak sararması"],
                severity="medium",
                treatment_methods=["Sülfür bazlı ilaçlar", "Biyolojik kontrol"],
                prevention_methods=["Düzenli aralık", "Havalandırma"],
                image_paths=["salatalik_kulleme_1.jpg"],
                climate_conditions={"temperature": "15-25°C", "humidity": "60-80%"},
                soil_conditions={"ph": "6.0-7.5", "drainage": "moderate"},
                season="ilkbahar",
                region="Ege",
                expert_notes="Podosphaera xanthii mantarı",
                confidence_score=0.92,
                created_date=datetime.now().isoformat(),
                verified_by="Dr. Tarım Uzmanı"
            ),
            PlantDiseaseRecord(
                id="PD003",
                plant_type="Biber",
                disease_name="Bakteriyel Leke",
                disease_type="bacterial",
                symptoms=["Küçük su damlası lezyonları", "Yaprak delinmesi"],
                severity="high",
                treatment_methods=["Bakır bazlı ilaçlar", "Hijyenik önlemler"],
                prevention_methods=["Tohum dezenfeksiyonu", "Crop rotation"],
                image_paths=["biber_bakteriyel_1.jpg"],
                climate_conditions={"temperature": "25-35°C", "humidity": "70-90%"},
                soil_conditions={"ph": "5.5-7.0", "drainage": "excellent"},
                season="yaz",
                region="Güneydoğu",
                expert_notes="Xanthomonas campestris bakterisi",
                confidence_score=0.88,
                created_date=datetime.now().isoformat(),
                verified_by="Dr. Tarım Uzmanı"
            )
        ]
        
        for record in sample_records:
            self.add_record(record)
        
        print(f"✅ {len(sample_records)} örnek kayıt eklendi!")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Veri seti istatistiklerini döndürür"""
        if not self.records:
            return {"total_records": 0}
        
        stats = {
            "total_records": len(self.records),
            "plant_types": list(set(r.plant_type for r in self.records)),
            "disease_types": list(set(r.disease_type for r in self.records)),
            "severity_distribution": {},
            "season_distribution": {},
            "region_distribution": {}
        }
        
        # Severity dağılımı
        for record in self.records:
            stats["severity_distribution"][record.severity] = \
                stats["severity_distribution"].get(record.severity, 0) + 1
        
        # Mevsim dağılımı
        for record in self.records:
            stats["season_distribution"][record.season] = \
                stats["season_distribution"].get(record.season, 0) + 1
        
        # Bölge dağılımı
        for record in self.records:
            stats["region_distribution"][record.region] = \
                stats["region_distribution"].get(record.region, 0) + 1
        
        return stats
    
    def export_for_training(self, output_file: str = "training_dataset.json"):
        """Eğitim için veri setini dışa aktarır"""
        try:
            training_data = []
            
            for record in self.records:
                # Claude API formatında eğitim verisi
                training_example = {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"""Bu bitki görüntüsünü analiz et:
                                    
Bitki Türü: {record.plant_type}
Mevsim: {record.season}
Bölge: {record.region}

Lütfen hastalık tespiti yap ve detaylı analiz ver."""
                                }
                            ]
                        },
                        {
                            "role": "assistant",
                            "content": f"""**Hastalık Tespiti**: {record.disease_name}

**Hastalık Türü**: {record.disease_type.capitalize()} hastalık

**Belirtiler**: {', '.join(record.symptoms)}

**Şiddet**: {record.severity.capitalize()}

**Tedavi Önerileri**: {', '.join(record.treatment_methods)}

**Önleyici Tedbirler**: {', '.join(record.prevention_methods)}

**Uzman Notları**: {record.expert_notes}

**Güvenilirlik**: %{int(record.confidence_score * 100)}

**İklim Koşulları**: Sıcaklık {record.climate_conditions.get('temperature', 'N/A')}, Nem {record.climate_conditions.get('humidity', 'N/A')}

**Toprak Koşulları**: pH {record.soil_conditions.get('ph', 'N/A')}, Drenaj {record.soil_conditions.get('drainage', 'N/A')}"""
                        }
                    ]
                }
                
                training_data.append(training_example)
            
            # Eğitim veri setini kaydet
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Eğitim veri seti oluşturuldu: {output_file}")
            print(f"📊 Toplam {len(training_data)} eğitim örneği")
            
            return True
            
        except Exception as e:
            print(f"Eğitim veri seti oluşturma hatası: {e}")
            return False

if __name__ == "__main__":
    # Veri toplayıcıyı başlat
    collector = DataCollector()
    
    # Örnek veri seti oluştur
    collector.create_sample_dataset()
    
    # İstatistikleri göster
    stats = collector.get_statistics()
    print("\n📊 Veri Seti İstatistikleri:")
    print(f"Toplam Kayıt: {stats['total_records']}")
    print(f"Bitki Türleri: {', '.join(stats['plant_types'])}")
    print(f"Hastalık Türleri: {', '.join(stats['disease_types'])}")
    
    # Eğitim veri seti oluştur
    collector.export_for_training()
