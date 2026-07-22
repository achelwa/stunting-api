"""
Stunting Prediction Pipeline — End-to-End Inference
====================================================
"""
import os
import joblib

class StuntingPredictor:
    def __init__(self,
                 artifacts_dir: str = 'artifacts',
                 model_filename: str = 'model_xgb_full.pkl',
                 threshold: float = 0.604):
        
        # 1. KUNCI JAWABAN: Dapatkan path absolut dari folder di mana file predict_pipeline.py berada
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        
        # 2. Gabungkan BASE_DIR dengan folder artifacts
        if not os.path.isabs(artifacts_dir):
            self.artifacts_dir = os.path.join(BASE_DIR, artifacts_dir)
        else:
            self.artifacts_dir = artifacts_dir

        self.threshold = threshold
        model_path = os.path.join(self.artifacts_dir, model_filename)

        # Print log untuk memastikan path yang dibaca di server Cloud
        print(f"[LOG SERVER] Membaca model dari path absolut: {model_path}")

        # 3. Cek keberadaan file
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model tidak ditemukan di: {model_path}\n"
                f"Isi dari folder BASE_DIR ({BASE_DIR}): {os.listdir(BASE_DIR)}"
            )

        # 4. Load Model
        self.model = joblib.load(model_path)
# =============================================================================
# KONSTANTA — MAPPING INPUT USER KE KODE SSGI
# =============================================================================

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
    def __init__(self,
                 artifacts_dir: str = './artifacts',
                 model_filename: str = 'model_xgb_full.pkl',
                 threshold: float = 0.604):
        
        # Samakan / absolutkan path ke folder artifacts
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.isabs(artifacts_dir):
            self.artifacts_dir = os.path.join(base_dir, artifacts_dir)
        else:
            self.artifacts_dir = artifacts_dir

        self.threshold = threshold
        model_path = os.path.join(self.artifacts_dir, model_filename)

        # 1. Cek keberadaan file model
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"File model tidak ditemukan di path: '{model_path}'. "
                f"Pastikan folder 'artifacts' sejajar dengan file python dan berisi '{model_filename}'."
            )

        # 2. Muat model (dengan fallback jika joblib standar gagal)
        try:
            self.model = joblib.load(model_path)
        except Exception as primary_err:
            # Fallback jika joblib gagal (misal masalah versi sklearn/pickle)
            if xgb is not None:
                try:
                    self.model = xgb.XGBClassifier()
                    self.model.load_model(model_path)
                except Exception as secondary_err:
                    raise RuntimeError(
                        f"Gagal memuat model via joblib maupun xgboost!\n"
                        f"Error Joblib: {primary_err}\n"
                        f"Error XGBoost: {secondary_err}"
                    )
            else:
                raise RuntimeError(
                    f"Gagal memuat model via joblib: {primary_err}. "
                    f"Pastikan xgboost terinstall (pip install xgboost)."
                )

        # 3. Load imputation values
        imp_path = os.path.join(self.artifacts_dir, 'imputation_values.json')
        if os.path.exists(imp_path):
            with open(imp_path, 'r', encoding='utf-8') as f:
                self.imputation_values = json.load(f)
        else:
            self.imputation_values = {
                'P1509': 11.6,
                'P512': 1,
                'P505': 4,
                'P409_IBU': 3,
            }

        # 4. Load feature order
        order_path = os.path.join(self.artifacts_dir, 'feature_order.json')
        if os.path.exists(order_path):
            with open(order_path, 'r', encoding='utf-8') as f:
                self.feature_order = json.load(f)
        else:
            self.feature_order = [
                'P1509', 'P409_IBU', 'P505', 'P512', 'P404_anak', 'P4072_anak',
                'FG1_GRAINS_ROOTS', 'FG2_LEGUMES', 'FG3_NUTS_SEEDS', 'FG4_DAIRY',
                'FG5_FLESH_FOODS', 'FG6_EGGS', 'FG7_VIT_A_RICH', 'FG8_OTHER_VEG_FRUIT',
                'MPASI_DIVERSITY', 'MDD_TERPENUHI', 'MPASI_FREQUENCY', 'MPASI_MAD',
                'MORBIDITY_INDEX', 'AGE_GROUP', 'SES_INDEX'
            ]

        self._explainer = None

    def _validate_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        required = ['umur_bulan', 'jenis_kelamin', 'berat_badan_kg']
        for field in required:
            if field not in data:
                raise ValueError(f"Field wajib hilang: '{field}'")

        umur = data['umur_bulan']
        if not isinstance(umur, (int, float)) or umur < 0 or umur > 59:
            raise ValueError(f"Umur tidak valid: {umur}. Harus 0-59 bulan.")

        jk = data['jenis_kelamin']
        if jk not in JENIS_KELAMIN_MAP:
            raise ValueError(
                f"Jenis kelamin tidak valid: '{jk}'. Harus salah satu: {list(JENIS_KELAMIN_MAP.keys())}"
            )

        bb = data['berat_badan_kg']
        if not isinstance(bb, (int, float)) or bb < 1.5 or bb > 35:
            raise ValueError(f"Berat badan tidak valid: {bb} kg. Harus 1.5-35 kg.")

        return data

    def _build_features(self, data: Dict[str, Any]) -> pd.DataFrame:
        features = {}
        features['P404_anak'] = JENIS_KELAMIN_MAP[data['jenis_kelamin']]
        features['P4072_anak'] = int(data['umur_bulan'])
        features['P1509'] = float(data['berat_badan_kg'])

        features['P409_IBU'] = PENDIDIKAN_IBU_MAP.get(
            data.get('pendidikan_ibu', 'SMA'), self.imputation_values.get('P409_IBU', 3)
        )
        features['P512'] = SANITASI_MAP.get(
            data.get('sanitasi', 'Leher Angsa (jamban sehat)'), self.imputation_values.get('P512', 1)
        )
        features['P505'] = KUALITAS_AIR_MAP.get(
            data.get('kualitas_air', 'Ledeng/PDAM'), self.imputation_values.get('P505', 4)
        )

        mpasi_inputs = {
            'P1108KK2': data.get('mpasi_karbo', False),
            'P1108LK2': data.get('mpasi_umbi', False),
            'P1108WK2': data.get('mpasi_protein_nabati', False),
            'P1108QK2': data.get('mpasi_jeroan', False),
            'P1108RK2': data.get('mpasi_daging', False),
            'P1108TK2': data.get('mpasi_telur', False),
            'P1108UK2': data.get('mpasi_ikan_segar', False),
            'P1108VK2': data.get('mpasi_ikan_awetan', False),
            'P1108MK2': data.get('mpasi_sayur_hijau', False),
            'P1108NK2': data.get('mpasi_sayur_oranye', False),
            'P1108OK2': data.get('mpasi_buah_vita', False),
            'P1108PK2': data.get('mpasi_buah_lain', False),
        }
        mpasi_inputs = {k: int(bool(v)) for k, v in mpasi_inputs.items()}

        umur = features['P4072_anak']
        eligible = (6 <= umur <= 23)

        fg_map = {
            'FG1_GRAINS_ROOTS': ['P1108KK2', 'P1108LK2'],
            'FG2_LEGUMES': ['P1108WK2'],
            'FG3_NUTS_SEEDS': [],
            'FG4_DAIRY': [],
            'FG5_FLESH_FOODS': ['P1108QK2', 'P1108RK2', 'P1108UK2', 'P1108VK2'],
            'FG6_EGGS': ['P1108TK2'],
            'FG7_VIT_A_RICH': ['P1108MK2', 'P1108NK2'],
            'FG8_OTHER_VEG_FRUIT': ['P1108OK2', 'P1108PK2'],
        }
        for fg_name, fg_cols in fg_map.items():
            if not fg_cols:
                features[fg_name] = 0
            else:
                val = 1 if any(mpasi_inputs[c] == 1 for c in fg_cols) else 0
                features[fg_name] = val if eligible else 0

        fg_cols_all = list(fg_map.keys())
        features['MPASI_DIVERSITY'] = sum(features[fg] for fg in fg_cols_all) if eligible else 0
        features['MDD_TERPENUHI'] = int(features['MPASI_DIVERSITY'] >= 5) if eligible else 0
        features['MPASI_FREQUENCY'] = sum(mpasi_inputs.values()) if eligible else 0
        features['MPASI_MAD'] = int(features['MDD_TERPENUHI'] == 1 and features['MPASI_FREQUENCY'] >= 4) if eligible else 0

        morbidity_inputs = {
            'demam': int(bool(data.get('demam', False))),
            'batuk': int(bool(data.get('batuk', False))),
            'diare': int(bool(data.get('diare', False))),
            'ispa': int(bool(data.get('ispa', False))),
        }
        features['MORBIDITY_INDEX'] = sum(morbidity_inputs.values())

        if umur <= 5:
            features['AGE_GROUP'] = 0
        elif umur <= 11:
            features['AGE_GROUP'] = 1
        elif umur <= 23:
            features['AGE_GROUP'] = 2
        else:
            features['AGE_GROUP'] = 3

        features['SES_INDEX'] = features['P409_IBU'] + features['P512']

        ordered = {col: features.get(col, 0) for col in self.feature_order}
        return pd.DataFrame([ordered])

    def predict(self, data: Dict[str, Any], return_explanation: bool = True) -> Dict[str, Any]:
        data = self._validate_input(data)
        X_input = self._build_features(data)

        # Predict probability
        prob = float(self.model.predict_proba(X_input)[0, 1])
        is_stunting = prob >= self.threshold

        if prob < 0.30:
            risk_level, risk_color = 'RENDAH', '#4CAF50'
        elif prob < 0.55:
            risk_level, risk_color = 'SEDANG', '#FFC107'
        elif prob < 0.75:
            risk_level, risk_color = 'TINGGI', '#FF9800'
        else:
            risk_level, risk_color = 'SANGAT TINGGI', '#D32F2F'

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

        if return_explanation:
            try:
                result['top_factors'] = self._explain(X_input)
            except Exception as e:
                result['top_factors'] = []
                result['explanation_error'] = str(e)

        result['recommendations'] = self._generate_recommendations(data, X_input, prob)
        return result

    def _get_explainer(self):
        if self._explainer is None:
            import shap
            clf = getattr(self.model, 'named_steps', {}).get('clf', self.model)
            self._explainer = shap.TreeExplainer(clf)
        return self._explainer

    def _explain(self, X_input: pd.DataFrame, top_k: int = 5) -> List[Dict]:
        explainer = self._get_explainer()
        shap_values = explainer.shap_values(X_input)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        if shap_values.ndim == 3:
            shap_values = shap_values[0, :, 1]
        elif shap_values.ndim == 2:
            shap_values = shap_values[0]

        feature_labels = {
            'P1509': 'Berat Badan', 'P404_anak': 'Jenis Kelamin', 'P4072_anak': 'Umur (bulan)',
            'P409_IBU': 'Pendidikan Ibu', 'P512': 'Jenis Sanitasi', 'P505': 'Kualitas Air',
            'FG1_GRAINS_ROOTS': 'Konsumsi Karbohidrat', 'FG2_LEGUMES': 'Konsumsi Kacang-kacangan',
            'FG5_FLESH_FOODS': 'Konsumsi Daging/Ikan', 'FG6_EGGS': 'Konsumsi Telur',
            'FG7_VIT_A_RICH': 'Konsumsi Sayur Vit A', 'FG8_OTHER_VEG_FRUIT': 'Konsumsi Sayur/Buah lain',
            'MPASI_DIVERSITY': 'Keragaman MPASI', 'MDD_TERPENUHI': 'Standar MDD WHO',
            'MPASI_FREQUENCY': 'Frekuensi MPASI', 'MPASI_MAD': 'Standar MAD WHO',
            'MORBIDITY_INDEX': 'Jumlah Penyakit Terkini', 'AGE_GROUP': 'Kelompok Umur',
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

        factors.sort(key=lambda x: x['abs_impact'], reverse=True)
        return factors[:top_k]

    def _generate_recommendations(self, data: Dict, X_input: pd.DataFrame, prob: float) -> List[str]:
        recs = []
        umur = int(X_input['P4072_anak'].iloc[0])

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

        if 6 <= umur <= 23:
            diversity = int(X_input['MPASI_DIVERSITY'].iloc[0])
            if diversity < 5:
                recs.append(f"🥗 Keragaman MPASI baru {diversity}/6 food groups. Target WHO: minimal 5.")
            if X_input['FG6_EGGS'].iloc[0] == 0:
                recs.append("🥚 Pertimbangkan menambahkan telur sebagai sumber protein hewani")
            if X_input['FG7_VIT_A_RICH'].iloc[0] == 0:
                recs.append("🥬 Tambahkan sayur hijau atau oranye (kaya Vitamin A)")

        if X_input['P512'].iloc[0] > 2:
            recs.append("🚽 Tingkatkan fasilitas sanitasi ke jamban leher angsa")

        if X_input['P505'].iloc[0] >= 5:
            recs.append("💧 Pastikan air minum dimasak sampai mendidih")

        if int(X_input['MORBIDITY_INDEX'].iloc[0]) >= 2:
            recs.append("🏥 Anak memiliki riwayat penyakit baru-baru ini. Konsultasikan ke Puskesmas.")

        return recs
