"""
Stunting Prediction Pipeline — End-to-End Inference
====================================================

Module ini mengemas seluruh logika preprocessing + prediksi dari notebook
training ke dalam satu function yang dapat dipanggil dari website/API.

Author: Marsel (Skripsi 2024-2025)
Model: XGBoost (Tuned) - SSGI 2024 Banten

USAGE:
    from predict_pipeline import StuntingPredictor
    
    predictor = StuntingPredictor()
    
    # Input data balita
    input_data = {
        'umur_bulan': 18,
        'jenis_kelamin': 'Perempuan',
        'pendidikan_ibu': 'SMA',
        'sanitasi': 'Leher Angsa',
        'kualitas_air': 'Air Kemasan/PDAM',
        'berat_badan_kg': 9.2,
        'mpasi_karbo': True,
        'mpasi_umbi': False,
        # ... (lihat dokumentasi field di bawah)
        'demam': False,
        'batuk': True,
        'diare': False,
        'ispa': False,
    }
    
    result = predictor.predict(input_data)
    print(result)
    # {
    #   'prediction': 'Stunting',
    #   'probability': 0.78,
    #   'risk_level': 'TINGGI',
    #   'top_factors': [...],
    #   'recommendations': [...]
    # }
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional


# =============================================================================
# KONSTANTA — MAPPING INPUT USER KE KODE SSGI
# =============================================================================

# Mapping dari label user-friendly ke kode SSGI numerik
JENIS_KELAMIN_MAP = {
    'Laki-laki': 1,
    'Perempuan': 2,
}

PENDIDIKAN_IBU_MAP = {
    'Tidak Sekolah/SD': 1,
    'SMP': 2,
    'SMA': 3,
    'D1/D2/D3': 4,
    'D4/S1': 5,
    'S2/S3': 6,
}

SANITASI_MAP = {
    'Leher Angsa (jamban sehat)': 1,
    'Plengsengan': 2,
    'Cemplung': 3,
    'Tidak punya/MCK umum': 4,
    'Lainnya': 5,
}

KUALITAS_AIR_MAP = {
    'Air Kemasan': 1,
    'Air Isi Ulang': 2,
    'Ledeng/PDAM': 3,
    'Sumur Bor/Pompa': 4,
    'Sumur Gali': 5,
    'Mata Air Terlindung': 6,
    'Mata Air Tidak Terlindung': 7,
    'Air Permukaan (sungai/danau)': 8,
}


# =============================================================================
# CORE PREDICTOR CLASS
# =============================================================================

class StuntingPredictor:
    """
    End-to-end stunting prediction pipeline.
    
    Loads model + preprocessing artifacts, validates input, applies feature
    engineering, and returns prediction with interpretation.
    """
    
    def __init__(self,
                 artifacts_dir: str = './artifacts',
                 model_filename: str = 'model_xgb_full.pkl',
                 threshold: float = 0.604):
        """
        Args:
            artifacts_dir: Folder berisi model.pkl, imputation_values.json,
                           feature_order.json
            model_filename: Nama file model (default: model_xgb_full.pkl)
            threshold: Probability threshold untuk klasifikasi 1/0
                       (default 0.604 dari hasil optimasi notebook)
        """
        self.artifacts_dir = artifacts_dir
        self.threshold = threshold
        
        # Load model
        model_path = os.path.join(artifacts_dir, model_filename)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model tidak ditemukan: {model_path}\n"
                f"Pastikan file .pkl ada di folder artifacts/"
            )
        self.model = joblib.load(model_path)
        
        # Load imputation values
        imp_path = os.path.join(artifacts_dir, 'imputation_values.json')
        if os.path.exists(imp_path):
            with open(imp_path, 'r') as f:
                self.imputation_values = json.load(f)
        else:
            # Fallback defaults (dari training data SSGI Banten 2024)
            self.imputation_values = {
                'P1509': 11.6,    # median berat badan training
                'P512': 1,        # modus sanitasi (Leher Angsa)
                'P505': 4,        # modus kualitas air (PDAM)
                'P409_IBU': 3,    # modus pendidikan ibu (SMA)
            }
        
        # Load feature order
        order_path = os.path.join(artifacts_dir, 'feature_order.json')
        if os.path.exists(order_path):
            with open(order_path, 'r') as f:
                self.feature_order = json.load(f)
        else:
            # Fallback: urutan default dari notebook
            self.feature_order = [
                'P1509', 'P409_IBU', 'P505', 'P512', 'P404_anak', 'P4072_anak',
                'FG1_GRAINS_ROOTS', 'FG2_LEGUMES', 'FG3_NUTS_SEEDS', 'FG4_DAIRY',
                'FG5_FLESH_FOODS', 'FG6_EGGS', 'FG7_VIT_A_RICH', 'FG8_OTHER_VEG_FRUIT',
                'MPASI_DIVERSITY', 'MDD_TERPENUHI', 'MPASI_FREQUENCY', 'MPASI_MAD',
                'MORBIDITY_INDEX', 'AGE_GROUP', 'SES_INDEX'
            ]
        
        # SHAP explainer (lazy load)
        self._explainer = None
    
    # =========================================================================
    # INPUT VALIDATION
    # =========================================================================
    
    def _validate_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validasi input user. Raises ValueError jika invalid."""
        
        # Field wajib
        required = ['umur_bulan', 'jenis_kelamin', 'berat_badan_kg']
        for field in required:
            if field not in data:
                raise ValueError(f"Field wajib hilang: '{field}'")
        
        # Validasi umur
        umur = data['umur_bulan']
        if not isinstance(umur, (int, float)) or umur < 0 or umur > 59:
            raise ValueError(
                f"Umur tidak valid: {umur}. Harus integer 0-59 bulan."
            )
        
        # Validasi jenis kelamin
        jk = data['jenis_kelamin']
        if jk not in JENIS_KELAMIN_MAP:
            raise ValueError(
                f"Jenis kelamin tidak valid: '{jk}'. "
                f"Harus salah satu: {list(JENIS_KELAMIN_MAP.keys())}"
            )
        
        # Validasi berat badan
        bb = data['berat_badan_kg']
        if not isinstance(bb, (int, float)) or bb < 1.5 or bb > 35:
            raise ValueError(
                f"Berat badan tidak valid: {bb} kg. Harus 1.5-35 kg untuk balita."
            )
        
        return data
    
    # =========================================================================
    # FEATURE ENGINEERING (HARUS SAMA DENGAN TRAINING!)
    # =========================================================================
    
    def _build_features(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Konstruksi 20+ fitur dari input user (sesuai notebook training).
        
        Args:
            data: dict input dari user (sudah di-validate)
        
        Returns:
            DataFrame 1 baris dengan kolom sesuai feature_order
        """
        features = {}
        
        # === DEMOGRAFI & ANTROPOMETRI ===
        features['P404_anak'] = JENIS_KELAMIN_MAP[data['jenis_kelamin']]
        features['P4072_anak'] = int(data['umur_bulan'])
        features['P1509'] = float(data['berat_badan_kg'])
        
        # === SOSIAL EKONOMI ===
        features['P409_IBU'] = PENDIDIKAN_IBU_MAP.get(
            data.get('pendidikan_ibu', 'SMA'),
            self.imputation_values['P409_IBU']
        )
        features['P512'] = SANITASI_MAP.get(
            data.get('sanitasi', 'Leher Angsa (jamban sehat)'),
            self.imputation_values['P512']
        )
        features['P505'] = KUALITAS_AIR_MAP.get(
            data.get('kualitas_air', 'Ledeng/PDAM'),
            self.imputation_values['P505']
        )
        
        # === MPASI: rekoding 12 jenis makanan ===
        # Konvensi: True/1 = dikonsumsi, False/0 = tidak
        mpasi_inputs = {
            'P1108KK2': data.get('mpasi_karbo', False),           # Karbohidrat
            'P1108LK2': data.get('mpasi_umbi', False),             # Umbi-umbian
            'P1108WK2': data.get('mpasi_protein_nabati', False),  # Protein Nabati
            'P1108QK2': data.get('mpasi_jeroan', False),           # Jeroan
            'P1108RK2': data.get('mpasi_daging', False),           # Daging
            'P1108TK2': data.get('mpasi_telur', False),            # Telur
            'P1108UK2': data.get('mpasi_ikan_segar', False),       # Ikan segar
            'P1108VK2': data.get('mpasi_ikan_awetan', False),      # Ikan awetan
            'P1108MK2': data.get('mpasi_sayur_hijau', False),      # Sayur hijau
            'P1108NK2': data.get('mpasi_sayur_oranye', False),     # Sayur oranye
            'P1108OK2': data.get('mpasi_buah_vita', False),        # Buah Vit A
            'P1108PK2': data.get('mpasi_buah_lain', False),        # Buah lainnya
        }
        # Konversi ke biner 0/1
        mpasi_inputs = {k: int(bool(v)) for k, v in mpasi_inputs.items()}
        
        # === ELIGIBILITY MPASI (WHO 2021: 6-23 bulan) ===
        umur = features['P4072_anak']
        eligible = (6 <= umur <= 23)
        
        # === 8 FOOD GROUPS WHO 2021 ===
        fg_map = {
            'FG1_GRAINS_ROOTS':    ['P1108KK2', 'P1108LK2'],
            'FG2_LEGUMES':         ['P1108WK2'],
            'FG3_NUTS_SEEDS':      [],  # Tidak ada di SSGI
            'FG4_DAIRY':           [],  # Tidak ada di SSGI
            'FG5_FLESH_FOODS':     ['P1108QK2', 'P1108RK2', 'P1108UK2', 'P1108VK2'],
            'FG6_EGGS':            ['P1108TK2'],
            'FG7_VIT_A_RICH':      ['P1108MK2', 'P1108NK2'],
            'FG8_OTHER_VEG_FRUIT': ['P1108OK2', 'P1108PK2'],
        }
        for fg_name, fg_cols in fg_map.items():
            if not fg_cols:
                features[fg_name] = 0
            else:
                val = 1 if any(mpasi_inputs[c] == 1 for c in fg_cols) else 0
                features[fg_name] = val if eligible else 0
        
        # === MPASI_DIVERSITY (MDD - WHO 2021 Indicator 5) ===
        fg_cols_all = list(fg_map.keys())
        features['MPASI_DIVERSITY'] = (
            sum(features[fg] for fg in fg_cols_all) if eligible else 0
        )
        
        # === MDD_TERPENUHI (≥5 food groups) ===
        features['MDD_TERPENUHI'] = int(features['MPASI_DIVERSITY'] >= 5) if eligible else 0
        
        # === MPASI_FREQUENCY (proxy MMF) ===
        features['MPASI_FREQUENCY'] = (
            sum(mpasi_inputs.values()) if eligible else 0
        )
        
        # === MPASI_MAD (Minimum Acceptable Diet) ===
        features['MPASI_MAD'] = int(
            features['MDD_TERPENUHI'] == 1 and
            features['MPASI_FREQUENCY'] >= 4
        ) if eligible else 0
        
        # === RIWAYAT PENYAKIT (MORBIDITY_INDEX) ===
        morbidity_inputs = {
            'demam':  int(bool(data.get('demam', False))),
            'batuk':  int(bool(data.get('batuk', False))),
            'diare':  int(bool(data.get('diare', False))),
            'ispa':   int(bool(data.get('ispa', False))),
        }
        features['MORBIDITY_INDEX'] = sum(morbidity_inputs.values())
        
        # === AGE_GROUP ===
        if umur <= 5:
            features['AGE_GROUP'] = 0   # 0-5 bln (belum MPASI)
        elif umur <= 11:
            features['AGE_GROUP'] = 1   # 6-11 bln (early MPASI)
        elif umur <= 23:
            features['AGE_GROUP'] = 2   # 12-23 bln (window kritis)
        else:
            features['AGE_GROUP'] = 3   # 24-59 bln (sudah lewat)
        
        # === SES_INDEX ===
        features['SES_INDEX'] = features['P409_IBU'] + features['P512']
        
        # === ARRANGE KE FEATURE ORDER YANG SAMA DENGAN TRAINING ===
        ordered = {col: features.get(col, 0) for col in self.feature_order}
        df = pd.DataFrame([ordered])
        
        return df
    
    # =========================================================================
    # PREDICTION
    # =========================================================================
    
    def predict(self, data: Dict[str, Any],
                return_explanation: bool = True) -> Dict[str, Any]:
        """
        Prediksi status stunting dari input user.
        
        Args:
            data: dict input dari user (lihat docstring class)
            return_explanation: jika True, sertakan SHAP top factors
        
        Returns:
            dict berisi:
                - prediction: 'Stunting' atau 'Tidak Stunting'
                - probability: float 0-1
                - risk_level: 'RENDAH' / 'SEDANG' / 'TINGGI' / 'SANGAT TINGGI'
                - threshold_used: float
                - top_factors: list of dict (jika return_explanation)
                - recommendations: list of str
                - input_summary: dict ringkasan input
        """
        # 1. Validasi
        data = self._validate_input(data)
        
        # 2. Feature engineering
        X_input = self._build_features(data)
        
        # 3. Predict probability
        prob = float(self.model.predict_proba(X_input)[0, 1])
        
        # 4. Klasifikasi dengan threshold optimal
        is_stunting = prob >= self.threshold
        
        # 5. Risk level
        if prob < 0.30:
            risk_level = 'RENDAH'
            risk_color = '#4CAF50'
        elif prob < 0.55:
            risk_level = 'SEDANG'
            risk_color = '#FFC107'
        elif prob < 0.75:
            risk_level = 'TINGGI'
            risk_color = '#FF9800'
        else:
            risk_level = 'SANGAT TINGGI'
            risk_color = '#D32F2F'
        
        # 6. Build result
        result = {
            'prediction': 'Stunting' if is_stunting else 'Tidak Stunting',
            'probability': round(prob, 4),
            'probability_percent': f"{prob*100:.1f}%",
            'risk_level': risk_level,
            'risk_color': risk_color,
            'threshold_used': self.threshold,
            'input_summary': {
                'umur_bulan': data['umur_bulan'],
                'jenis_kelamin': data['jenis_kelamin'],
                'berat_badan_kg': data['berat_badan_kg'],
                'pendidikan_ibu': data.get('pendidikan_ibu', 'N/A'),
                'food_groups_consumed': int(X_input['MPASI_DIVERSITY'].iloc[0]),
                'morbidity_count': int(X_input['MORBIDITY_INDEX'].iloc[0]),
            }
        }
        
        # 7. SHAP explanation (top factors)
        if return_explanation:
            try:
                result['top_factors'] = self._explain(X_input)
            except Exception as e:
                result['top_factors'] = []
                result['explanation_error'] = str(e)
        
        # 8. Recommendations berdasarkan risk + input
        result['recommendations'] = self._generate_recommendations(data, X_input, prob)
        
        return result
    
    # =========================================================================
    # EXPLANATION (SHAP)
    # =========================================================================
    
    def _get_explainer(self):
        """Lazy-load SHAP explainer."""
        if self._explainer is None:
            try:
                import shap
                # Ambil XGBClassifier dari pipeline jika perlu
                if hasattr(self.model, 'named_steps'):
                    clf = self.model.named_steps.get('clf', self.model)
                else:
                    clf = self.model
                self._explainer = shap.TreeExplainer(clf)
            except ImportError:
                raise ImportError(
                    "Library 'shap' tidak terinstall. Run: pip install shap"
                )
        return self._explainer
    
    def _explain(self, X_input: pd.DataFrame, top_k: int = 5) -> List[Dict]:
        """
        Hitung SHAP values dan return top K factors paling berpengaruh.
        
        Returns:
            list of dict dengan: feature, value, shap_value, direction
        """
        explainer = self._get_explainer()
        shap_values = explainer.shap_values(X_input)
        
        # Untuk binary classification, shap_values bisa 2D atau 3D
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # class 1 (stunting)
        if shap_values.ndim == 3:
            shap_values = shap_values[0, :, 1]
        else:
            shap_values = shap_values[0]
        
        # Friendly names
        feature_labels = {
            'P1509': 'Berat Badan',
            'P404_anak': 'Jenis Kelamin',
            'P4072_anak': 'Umur (bulan)',
            'P409_IBU': 'Pendidikan Ibu',
            'P512': 'Jenis Sanitasi',
            'P505': 'Kualitas Air',
            'FG1_GRAINS_ROOTS': 'Konsumsi Karbohidrat',
            'FG2_LEGUMES': 'Konsumsi Kacang-kacangan',
            'FG5_FLESH_FOODS': 'Konsumsi Daging/Ikan',
            'FG6_EGGS': 'Konsumsi Telur',
            'FG7_VIT_A_RICH': 'Konsumsi Sayur Vit A',
            'FG8_OTHER_VEG_FRUIT': 'Konsumsi Sayur/Buah lain',
            'MPASI_DIVERSITY': 'Keragaman MPASI',
            'MDD_TERPENUHI': 'Standar MDD WHO',
            'MPASI_FREQUENCY': 'Frekuensi MPASI',
            'MPASI_MAD': 'Standar MAD WHO',
            'MORBIDITY_INDEX': 'Jumlah Penyakit Terkini',
            'AGE_GROUP': 'Kelompok Umur',
            'SES_INDEX': 'Indeks Sosial Ekonomi',
        }
        
        factors = []
        for i, col in enumerate(self.feature_order):
            if i < len(shap_values):
                factors.append({
                    'feature': feature_labels.get(col, col),
                    'feature_code': col,
                    'value': float(X_input[col].iloc[0]),
                    'shap_value': float(shap_values[i]),
                    'direction': 'meningkatkan' if shap_values[i] > 0 else 'menurunkan',
                    'abs_impact': abs(float(shap_values[i])),
                })
        
        # Sort by absolute impact
        factors.sort(key=lambda x: x['abs_impact'], reverse=True)
        
        return factors[:top_k]
    
    # =========================================================================
    # RECOMMENDATIONS
    # =========================================================================
    
    def _generate_recommendations(self, data: Dict, X_input: pd.DataFrame,
                                   prob: float) -> List[str]:
        """Generate rekomendasi berdasarkan input dan risk level."""
        recs = []
        umur = int(X_input['P4072_anak'].iloc[0])
        
        # Rekomendasi umum berdasarkan risk
        if prob >= 0.75:
            recs.append("⚠️ Segera konsultasikan dengan tenaga kesehatan (Posyandu/Puskesmas)")
            recs.append("📋 Lakukan pemantauan pertumbuhan bulanan dengan ketat")
        elif prob >= 0.55:
            recs.append("📋 Konsultasi rutin dengan kader Posyandu setiap bulan")
            recs.append("🍽️ Tingkatkan kualitas dan keragaman MPASI")
        elif prob >= 0.30:
            recs.append("✅ Lanjutkan pemantauan pertumbuhan rutin di Posyandu")
        else:
            recs.append("✅ Pertahankan pola asuh dan pemberian makan yang baik")
        
        # Rekomendasi MPASI (jika 6-23 bulan)
        if 6 <= umur <= 23:
            diversity = int(X_input['MPASI_DIVERSITY'].iloc[0])
            if diversity < 5:
                recs.append(
                    f"🥗 Keragaman MPASI baru {diversity}/6 food groups. "
                    f"Target WHO: minimal 5. Tambah variasi protein hewani, sayur, buah."
                )
            
            if X_input['FG6_EGGS'].iloc[0] == 0:
                recs.append("🥚 Pertimbangkan menambahkan telur sebagai sumber protein hewani")
            
            if X_input['FG7_VIT_A_RICH'].iloc[0] == 0:
                recs.append("🥬 Tambahkan sayur hijau atau oranye (kaya Vitamin A)")
        
        # Rekomendasi sanitasi
        sanitasi_val = X_input['P512'].iloc[0]
        if sanitasi_val > 2:  # Bukan jamban sehat
            recs.append("🚽 Tingkatkan fasilitas sanitasi ke jamban leher angsa (jika memungkinkan)")
        
        # Rekomendasi air
        air_val = X_input['P505'].iloc[0]
        if air_val >= 5:  # Sumur gali atau lebih buruk
            recs.append("💧 Pastikan air minum dimasak sampai mendidih atau gunakan air kemasan/PDAM")
        
        # Rekomendasi morbiditas
        morb = int(X_input['MORBIDITY_INDEX'].iloc[0])
        if morb >= 2:
            recs.append(
                f"🏥 Anak memiliki {morb} penyakit dalam 2 minggu terakhir. "
                f"Konsultasikan ke Puskesmas untuk pemeriksaan lebih lanjut."
            )
        
        return recs


# =============================================================================
# QUICK TEST (jika dijalankan langsung)
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Quick Test — StuntingPredictor")
    print("=" * 60)
    
    predictor = StuntingPredictor(artifacts_dir='./artifacts')
    
    # Contoh input balita
    sample_input = {
        'umur_bulan': 18,
        'jenis_kelamin': 'Perempuan',
        'pendidikan_ibu': 'SMA',
        'sanitasi': 'Leher Angsa (jamban sehat)',
        'kualitas_air': 'Ledeng/PDAM',
        'berat_badan_kg': 9.2,
        'mpasi_karbo': True,
        'mpasi_umbi': False,
        'mpasi_protein_nabati': True,
        'mpasi_telur': True,
        'mpasi_daging': False,
        'mpasi_jeroan': False,
        'mpasi_ikan_segar': True,
        'mpasi_ikan_awetan': False,
        'mpasi_sayur_hijau': True,
        'mpasi_sayur_oranye': False,
        'mpasi_buah_vita': True,
        'mpasi_buah_lain': False,
        'demam': False,
        'batuk': True,
        'diare': False,
        'ispa': False,
    }
    
    result = predictor.predict(sample_input)
    
    print(f"\nPrediksi      : {result['prediction']}")
    print(f"Probabilitas  : {result['probability_percent']}")
    print(f"Risk Level    : {result['risk_level']}")
    print(f"\nFood Groups dikonsumsi: {result['input_summary']['food_groups_consumed']}/6")
    print(f"Penyakit recent: {result['input_summary']['morbidity_count']}")
    
    if result.get('top_factors'):
        print(f"\n5 Faktor Utama:")
        for i, f in enumerate(result['top_factors'], 1):
            print(f"  {i}. {f['feature']:<28} → {f['direction']} risiko (SHAP: {f['shap_value']:+.3f})")
    
    print(f"\nRekomendasi:")
    for r in result['recommendations']:
        print(f"  {r}")
