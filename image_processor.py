import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import os
from typing import Dict, List, Tuple, Any
import base64
from io import BytesIO

class ImageProcessor:
    """Gelişmiş görüntü işleme ve analiz sınıfı"""
    
    def __init__(self):
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        self.enhancement_factors = {
            'contrast': 1.5,
            'brightness': 1.2,
            'sharpness': 1.3,
            'color': 1.1
        }
    
    def preprocess_image(self, image_path: str, output_path: str = None) -> Dict[str, Any]:
        """Görüntüyü analiz için ön işleme"""
        
        try:
            # Görüntüyü yükle
            image = Image.open(image_path)
            
            # Görüntü bilgileri
            original_info = {
                'size': image.size,
                'mode': image.mode,
                'format': image.format,
                'file_size': os.path.getsize(image_path)
            }
            
            # Görüntü kalitesini artır
            enhanced_image = self._enhance_image_quality(image)
            
            # Gürültü azaltma
            denoised_image = self._denoise_image(enhanced_image)
            
            # Kontrast ve parlaklık optimizasyonu
            optimized_image = self._optimize_contrast_brightness(denoised_image)
            
            # Çıktı dosyası kaydet
            if output_path:
                optimized_image.save(output_path, quality=95, optimize=True)
            
            # İşlem sonrası bilgiler
            processed_info = {
                'original_info': original_info,
                'enhancement_applied': list(self.enhancement_factors.keys()),
                'processing_steps': ['quality_enhancement', 'denoising', 'contrast_optimization'],
                'output_path': output_path
            }
            
            return {
                'success': True,
                'processed_image': optimized_image,
                'info': processed_info
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Görüntü işleme hatası: {str(e)}'
            }
    
    def _enhance_image_quality(self, image: Image.Image) -> Image.Image:
        """Görüntü kalitesini artırır"""
        
        # Kontrast artırma
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(self.enhancement_factors['contrast'])
        
        # Parlaklık artırma
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(self.enhancement_factors['brightness'])
        
        # Keskinlik artırma
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(self.enhancement_factors['sharpness'])
        
        # Renk doygunluğu
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(self.enhancement_factors['color'])
        
        return image
    
    def _denoise_image(self, image: Image.Image) -> Image.Image:
        """Görüntü gürültüsünü azaltır"""
        
        # PIL ile basit gürültü azaltma
        denoised = image.filter(ImageFilter.MedianFilter(size=3))
        
        # Daha gelişmiş gürültü azaltma için OpenCV kullan
        try:
            # PIL'den numpy array'e çevir
            cv_image = cv2.cvtColor(np.array(denoised), cv2.COLOR_RGB2BGR)
            
            # Bilateral filter ile gürültü azaltma
            denoised_cv = cv2.bilateralFilter(cv_image, 9, 75, 75)
            
            # Numpy array'den PIL'e geri çevir
            denoised = Image.fromarray(cv2.cvtColor(denoised_cv, cv2.COLOR_BGR2RGB))
            
        except Exception:
            # OpenCV kullanılamazsa PIL ile devam et
            pass
        
        return denoised
    
    def _optimize_contrast_brightness(self, image: Image.Image) -> Image.Image:
        """Kontrast ve parlaklığı optimize eder"""
        
        # Histogram eşitleme için OpenCV kullan
        try:
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # LAB renk uzayına çevir
            lab = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
            
            # L kanalında histogram eşitleme
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            # LAB'dan BGR'ye geri çevir
            lab = cv2.merge([l, a, b])
            optimized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # PIL'e geri çevir
            return Image.fromarray(cv2.cvtColor(optimized, cv2.COLOR_BGR2RGB))
            
        except Exception:
            # OpenCV kullanılamazsa PIL ile devam et
            return image
    
    def extract_image_features(self, image_path: str) -> Dict[str, Any]:
        """Görüntüden özellik çıkarır"""
        
        try:
            # OpenCV ile görüntü yükle
            image = cv2.imread(image_path)
            if image is None:
                return {'success': False, 'error': 'Görüntü yüklenemedi'}
            
            # Gri tonlamaya çevir
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Temel istatistikler
            features = {
                'success': True,
                'basic_stats': {
                    'mean_intensity': float(np.mean(gray)),
                    'std_intensity': float(np.std(gray)),
                    'min_intensity': int(np.min(gray)),
                    'max_intensity': int(np.max(gray)),
                    'image_size': image.shape[:2],
                    'aspect_ratio': image.shape[1] / image.shape[0]
                },
                'texture_features': self._extract_texture_features(gray),
                'color_features': self._extract_color_features(image),
                'edge_features': self._extract_edge_features(gray),
                'leaf_analysis': self._analyze_leaf_characteristics(gray)
            }
            
            return features
            
        except Exception as e:
            return {'success': False, 'error': f'Özellik çıkarma hatası: {str(e)}'}
    
    def _extract_texture_features(self, gray_image: np.ndarray) -> Dict[str, float]:
        """Doku özelliklerini çıkarır"""
        
        try:
            # GLCM (Gray Level Co-occurrence Matrix) özellikleri
            # Basit doku özellikleri
            texture_features = {
                'smoothness': float(np.var(gray_image)),
                'uniformity': float(np.sum(gray_image ** 2)),
                'entropy': float(-np.sum(gray_image * np.log2(gray_image + 1e-10))),
                'energy': float(np.sum(gray_image ** 2))
            }
            
            return texture_features
            
        except Exception:
            return {}
    
    def _extract_color_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Renk özelliklerini çıkarır"""
        
        try:
            # BGR'den HSV'ye çevir
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Renk kanalları
            h, s, v = cv2.split(hsv)
            
            color_features = {
                'hue_mean': float(np.mean(h)),
                'saturation_mean': float(np.mean(s)),
                'value_mean': float(np.mean(v)),
                'hue_std': float(np.std(h)),
                'saturation_std': float(np.std(s)),
                'value_std': float(np.std(v)),
                'dominant_colors': self._find_dominant_colors(image)
            }
            
            return color_features
            
        except Exception:
            return {}
    
    def _find_dominant_colors(self, image: np.ndarray, num_colors: int = 5) -> List[Tuple]:
        """Baskın renkleri bulur"""
        
        try:
            # Görüntüyü yeniden boyutlandır
            small_image = cv2.resize(image, (150, 150))
            
            # Piksel verilerini düzleştir
            pixels = small_image.reshape(-1, 3)
            
            # K-means ile renk kümeleme
            from sklearn.cluster import KMeans
            
            kmeans = KMeans(n_clusters=num_colors, random_state=42)
            kmeans.fit(pixels)
            
            # Renk merkezleri
            colors = kmeans.cluster_centers_.astype(int)
            
            # Her rengin piksel sayısı
            labels = kmeans.labels_
            color_counts = np.bincount(labels)
            
            # Renk ve sayıları birleştir
            dominant_colors = [(tuple(color), int(count)) for color, count in zip(colors, color_counts)]
            
            # Sayıya göre sırala
            dominant_colors.sort(key=lambda x: x[1], reverse=True)
            
            return dominant_colors
            
        except Exception:
            return []
    
    def _extract_edge_features(self, gray_image: np.ndarray) -> Dict[str, float]:
        """Kenar özelliklerini çıkarır"""
        
        try:
            # Canny kenar tespiti
            edges = cv2.Canny(gray_image, 50, 150)
            
            # Kenar yoğunluğu
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Kenar yönleri
            sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            
            # Gradient büyüklüğü ve yönü
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
            gradient_direction = np.arctan2(sobely, sobelx)
            
            edge_features = {
                'edge_density': float(edge_density),
                'gradient_magnitude_mean': float(np.mean(gradient_magnitude)),
                'gradient_direction_mean': float(np.mean(gradient_direction)),
                'edge_complexity': float(np.std(gradient_magnitude))
            }
            
            return edge_features
            
        except Exception:
            return {}
    
    def _analyze_leaf_characteristics(self, gray_image: np.ndarray) -> Dict[str, Any]:
        """Yaprak özelliklerini analiz eder"""
        
        try:
            # Yaprak şekli analizi
            # Basit kontur tespiti
            _, binary = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # En büyük konturu bul
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Alan ve çevre
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                
                # Dairesellik
                circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                
                # Sınırlayıcı dikdörtgen
                x, y, w, h = cv2.boundingRect(largest_contour)
                aspect_ratio = w / h if h > 0 else 0
                
                leaf_analysis = {
                    'leaf_area': float(area),
                    'leaf_perimeter': float(perimeter),
                    'circularity': float(circularity),
                    'aspect_ratio': float(aspect_ratio),
                    'bounding_box': (int(x), int(y), int(w), int(h))
                }
            else:
                leaf_analysis = {}
            
            return leaf_analysis
            
        except Exception:
            return {}
    
    def create_analysis_visualization(self, image_path: str, output_path: str = None) -> Dict[str, Any]:
        """Analiz görselleştirmesi oluşturur"""
        
        try:
            # Görüntüyü yükle
            image = cv2.imread(image_path)
            if image is None:
                return {'success': False, 'error': 'Görüntü yüklenemedi'}
            
            # Gri tonlamaya çevir
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Kenar tespiti
            edges = cv2.Canny(gray, 50, 150)
            
            # Kontur tespiti
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Görselleştirme
            visualization = image.copy()
            
            # Konturları çiz
            cv2.drawContours(visualization, contours, -1, (0, 255, 0), 2)
            
            # Kenarları çiz
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            edges_colored[edges > 0] = [0, 0, 255]  # Kırmızı kenarlar
            
            # Görselleştirmeleri birleştir
            combined = np.hstack([image, visualization, edges_colored])
            
            # Çıktı dosyası kaydet
            if output_path:
                cv2.imwrite(output_path, combined)
            
            return {
                'success': True,
                'visualization': combined,
                'output_path': output_path,
                'contours_found': len(contours),
                'edges_detected': np.sum(edges > 0)
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Görselleştirme hatası: {str(e)}'}
    
    def image_to_base64(self, image_path: str) -> str:
        """Görüntüyü base64 formatına çevirir"""
        
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Base64 dönüştürme hatası: {e}")
            return ""

if __name__ == "__main__":
    # Test
    processor = ImageProcessor()
    
    # Örnek görüntü işleme
    test_image = "test_image.jpg"
    
    if os.path.exists(test_image):
        print("🔍 Görüntü analizi başlıyor...")
        
        # Özellik çıkarma
        features = processor.extract_image_features(test_image)
        if features['success']:
            print("✅ Özellikler çıkarıldı:")
            print(f"Görüntü boyutu: {features['basic_stats']['image_size']}")
            print(f"Ortalama yoğunluk: {features['basic_stats']['mean_intensity']:.2f}")
            print(f"Kenar yoğunluğu: {features['edge_features']['edge_density']:.4f}")
        
        # Görselleştirme
        viz_result = processor.create_analysis_visualization(test_image, "analysis_viz.jpg")
        if viz_result['success']:
            print(f"✅ Görselleştirme oluşturuldu: {viz_result['output_path']}")
    
    else:
        print("❌ Test görüntüsü bulunamadı. Lütfen bir görüntü dosyası ekleyin.")
