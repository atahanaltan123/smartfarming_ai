from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os
import json
from datetime import datetime
from claude_client import ClaudeClient
from config import Config
from model_orchestrator import ModelOrchestrator
from advanced_trainer import AdvancedTrainer
from model_evaluator import ModelEvaluator

app = Flask(__name__)
app.config.from_object(Config)

# Upload klasörünü oluştur
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# AI sınıflarını başlat
claude_client = ClaudeClient()
model_orchestrator = ModelOrchestrator()
advanced_trainer = AdvancedTrainer()
model_evaluator = ModelEvaluator()

def allowed_file(filename):
    """Dosya uzantısı kontrolü"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    """Ana sayfa"""
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_plant():
    """Bitki görüntüsü analizi API endpoint'i"""
    try:
        # Dosya kontrolü
        if 'image' not in request.files:
            return jsonify({'error': 'Görüntü dosyası bulunamadı'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Dosya seçilmedi'}), 400
        
        if file and allowed_file(file.filename):
            # Güvenli dosya adı oluştur
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Dosyayı kaydet
            file.save(filepath)
            
            # Bitki türü bilgisini al
            plant_type = request.form.get('plant_type', 'genel')
            
            # Claude API ile analiz
            result = claude_client.analyze_plant_image(filepath, plant_type)
            
            # Analiz sonucunu kaydet
            analysis_record = {
                'filename': filename,
                'plant_type': plant_type,
                'timestamp': datetime.now().isoformat(),
                'result': result
            }
            
            # Sonuçları JSON dosyasına kaydet
            save_analysis_record(analysis_record)
            
            return jsonify(result)
        else:
            return jsonify({'error': 'Geçersiz dosya türü'}), 400
            
    except Exception as e:
        return jsonify({'error': f'İşlem hatası: {str(e)}'}), 500

@app.route('/api/advice', methods=['POST'])
def get_plant_advice():
    """Bitki bakım önerisi API endpoint'i"""
    try:
        data = request.get_json()
        plant_type = data.get('plant_type', 'genel')
        season = data.get('season', 'genel')
        
        result = claude_client.get_plant_care_advice(plant_type, season)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'İşlem hatası: {str(e)}'}), 500

@app.route('/api/history')
def get_analysis_history():
    """Analiz geçmişi API endpoint'i"""
    try:
        history = load_analysis_history()
        return jsonify({'history': history})
    except Exception as e:
        return jsonify({'error': f'İşlem hatası: {str(e)}'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Yüklenen dosyaları sunar"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ===== YENİ AI API ENDPOINT'LERİ =====

@app.route('/api/ensemble/analyze', methods=['POST'])
def ensemble_analysis():
    """Ensemble AI analizi"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Görüntü dosyası bulunamadı'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Dosya seçilmedi'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            file.save(filepath)
            plant_type = request.form.get('plant_type', 'genel')
            
            # Ensemble analizi
            result = model_orchestrator.analyze_with_multiple_models(filepath, plant_type)
            
            # Sonucu kaydet
            analysis_record = {
                'filename': filename,
                'plant_type': plant_type,
                'timestamp': datetime.now().isoformat(),
                'result': result,
                'type': 'ensemble'
            }
            save_analysis_record(analysis_record)
            
            return jsonify(result)
        else:
            return jsonify({'error': 'Geçersiz dosya türü'}), 400
            
    except Exception as e:
        return jsonify({'error': f'Ensemble analiz hatası: {str(e)}'}), 500

@app.route('/api/training/start', methods=['POST'])
def start_training_session():
    """Eğitim oturumu başlat"""
    try:
        data = request.get_json()
        session_name = data.get('session_name', 'Web Session')
        
        session_id = advanced_trainer.start_training_session(session_name)
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'total_examples': len(advanced_trainer.training_examples)
        })
        
    except Exception as e:
        return jsonify({'error': f'Eğitim oturumu başlatılamadı: {str(e)}'}), 500

@app.route('/api/training/train', methods=['POST'])
def train_model():
    """Model eğitimi"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        model_name = data.get('model_name', 'claude-3-5-sonnet')
        
        if not session_id:
            return jsonify({'error': 'Session ID gerekli'}), 400
        
        # Eğitim veri setini yükle
        if not advanced_trainer.training_examples:
            advanced_trainer.load_training_dataset()
        
        # Model eğitimi
        result = advanced_trainer.train_model_with_fine_tuning(model_name)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Model eğitimi başarısız: {str(e)}'}), 500

@app.route('/api/training/end', methods=['POST'])
def end_training_session():
    """Eğitim oturumunu sonlandır"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({'error': 'Session ID gerekli'}), 400
        
        result = advanced_trainer.end_training_session()
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Eğitim oturumu sonlandırılamadı: {str(e)}'}), 500

@app.route('/api/evaluation/evaluate', methods=['POST'])
def evaluate_model():
    """Model değerlendirmesi"""
    try:
        data = request.get_json()
        model_name = data.get('model_name', 'claude-3-5-sonnet')
        test_subset_size = data.get('test_subset_size', 5)
        
        # Test veri setini yükle
        if not model_evaluator.test_dataset:
            model_evaluator.load_test_dataset()
        
        # Model değerlendirmesi
        result = model_evaluator.evaluate_model_performance(model_name, test_subset_size)
        
        # Sonucu JSON'a çevir
        evaluation_data = {
            'success': True,
            'accuracy': result.performance_metrics.accuracy,
            'precision': result.performance_metrics.precision,
            'recall': result.performance_metrics.recall,
            'f1_score': result.performance_metrics.f1_score,
            'processing_time': result.performance_metrics.processing_time,
            'throughput': result.performance_metrics.throughput,
            'latency': result.performance_metrics.latency,
            'recommendations': result.recommendations
        }
        
        return jsonify(evaluation_data)
        
    except Exception as e:
        return jsonify({'error': f'Model değerlendirmesi başarısız: {str(e)}'}), 500

@app.route('/api/evaluation/compare', methods=['POST'])
def compare_models():
    """Model karşılaştırması"""
    try:
        data = request.get_json()
        model_names = data.get('model_names', ['claude-3-5-sonnet'])
        
        # Model karşılaştırması
        result = model_evaluator.compare_models(model_names)
        
        return jsonify({
            'success': True,
            'ranking': result.get('ranking', {}),
            'recommendations': result.get('recommendations', [])
        })
        
    except Exception as e:
        return jsonify({'error': f'Model karşılaştırması başarısız: {str(e)}'}), 500

@app.route('/api/lab/analyze', methods=['POST'])
def analyze_lab_results():
    """Laboratuvar sonuçları analizi"""
    try:
        data = request.get_json()
        test_type = data.get('test_type', 'blood')
        lab_data = data.get('lab_data', '')
        patient_info = data.get('patient_info', '')
        reference_range = data.get('reference_range', 'adult')
        
        if not lab_data:
            return jsonify({'error': 'Laboratuvar verisi bulunamadı'}), 400
        
        # Claude API ile laboratuvar analizi
        result = claude_client.analyze_lab_results(
            test_type, lab_data, patient_info, reference_range
        )
        
        # Sonucu kaydet
        analysis_record = {
            'type': 'lab_analysis',
            'test_type': test_type,
            'patient_info': patient_info,
            'reference_range': reference_range,
            'timestamp': datetime.now().isoformat(),
            'result': result
        }
        save_analysis_record(analysis_record)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Laboratuvar analizi hatası: {str(e)}'}), 500

@app.route('/api/export/report', methods=['POST'])
def export_analysis_report():
    """Analiz raporu dışa aktar"""
    try:
        # Tüm analiz verilerini topla
        history = load_analysis_history()
        
        # Model performans verilerini topla
        performance_data = {
            'orchestrator': model_orchestrator.get_model_performance_report(),
            'trainer': advanced_trainer.get_training_statistics() if advanced_trainer.training_sessions else {},
            'evaluator': {model: model_evaluator.get_performance_trends(model) for model in ['claude-3-5-sonnet']}
        }
        
        # Rapor oluştur
        report = {
            'generated_at': datetime.now().isoformat(),
            'analysis_history': history,
            'performance_data': performance_data,
            'system_info': {
                'total_models': len(model_orchestrator.models),
                'training_sessions': len(advanced_trainer.training_sessions),
                'api_version': '2023-06-01'
            }
        }
        
        # JSON dosyası olarak döndür
        from io import StringIO
        import json
        
        output = StringIO()
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.seek(0)
        
        from flask import Response
        return Response(
            output.getvalue(),
            mimetype='application/json',
            headers={'Content-Disposition': 'attachment;filename=smart_farming_ai_report.json'}
        )
        
    except Exception as e:
        return jsonify({'error': f'Rapor oluşturulamadı: {str(e)}'}), 500

def save_analysis_record(record):
    """Analiz kaydını JSON dosyasına kaydeder"""
    history_file = 'analysis_history.json'
    
    try:
        if os.path.exists(history_file):
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        else:
            history = []
        
        history.append(record)
        
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        print(f"Kayıt hatası: {e}")

def load_analysis_history():
    """Analiz geçmişini yükler"""
    history_file = 'analysis_history.json'
    
    try:
        if os.path.exists(history_file):
            with open(history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"Yükleme hatası: {e}")
        return []

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=app.config['DEBUG'], host='0.0.0.0', port=port)
