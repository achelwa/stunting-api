"""
Save Artifacts for Deployment
==============================

Script ini dijalankan SATU KALI di notebook training (setelah model selesai)
untuk menyimpan semua artifacts yang dibutuhkan deployment.

CARA PAKAI:
    1. Buka notebook training Anda (marsel_nitip.ipynb)
    2. Copy isi file ini sebagai cell terakhir
    3. Jalankan cell tersebut
    4. File-file akan tersimpan ke folder artifacts/
    5. Upload folder artifacts/ ke server deployment

OUTPUT yang dihasilkan:
    artifacts/
    ├── model_xgb_full.pkl       (sudah ada dari notebook)
    ├── model_xgb_no_P1509.pkl   (sudah ada dari notebook)
    ├── imputation_values.json   (BARU - generate dari script ini)
    ├── feature_order.json       (BARU - generate dari script ini)
    └── training_metadata.json   (BARU - info training)
"""

import os
import json
import joblib

# =============================================================================
# 1. PASTIKAN FOLDER artifacts/ ADA
# =============================================================================
os.makedirs('artifacts', exist_ok=True)
print("✅ Folder artifacts/ siap")


# =============================================================================
# 2. SIMPAN URUTAN KOLOM FITUR (feature_order.json)
#    Ini WAJIB supaya input ke predict_pipeline punya urutan yang sama
#    dengan saat training
# =============================================================================
feature_order = list(X_train.columns)
with open('artifacts/feature_order.json', 'w') as f:
    json.dump(feature_order, f, indent=2)
print(f"✅ feature_order.json — {len(feature_order)} fitur")
print(f"   Urutan: {feature_order[:5]}... (total {len(feature_order)})")


# =============================================================================
# 3. SIMPAN IMPUTATION VALUES (imputation_values.json)
#    Statistik dari TRAIN data yang dipakai untuk fill NaN saat inference
# =============================================================================
imputation_values = {
    # Median dari training data (untuk numerik)
    'P1509': float(X_train['P1509'].median()),
    
    # Modus dari training data (untuk kategorikal)
    'P512':     int(X_train['P512'].mode().iloc[0]),
    'P505':     int(X_train['P505'].mode().iloc[0]),
    'P409_IBU': int(X_train['P409_IBU'].mode().iloc[0]),
}

with open('artifacts/imputation_values.json', 'w') as f:
    json.dump(imputation_values, f, indent=2)
print(f"✅ imputation_values.json — {imputation_values}")


# =============================================================================
# 4. SIMPAN METADATA TRAINING (training_metadata.json)
#    Info versi model, hyperparameter terbaik, performa, dll.
# =============================================================================
metadata = {
    'model_info': {
        'algorithm': 'XGBoost (with SMOTE in pipeline)',
        'random_state': 42,
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    },
    'dataset_info': {
        'source': 'SSGI 2024 - Provinsi Banten',
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_features': X_train.shape[1],
        'stunting_rate': float(y_train.mean()),
    },
    'best_hyperparameters_xgb': {
        k.replace('clf__', ''): str(v) 
        for k, v in xgb_random.best_params_.items()
    },
    'performance_test_set': {
        'accuracy': float(eval_xgb['metrics']['Accuracy']),
        'precision': float(eval_xgb['metrics']['Precision']),
        'recall': float(eval_xgb['metrics']['Recall']),
        'f1_score': float(eval_xgb['metrics']['F1-Score']),
        'auc_roc': float(eval_xgb['metrics']['AUC-ROC']),
        'auc_pr': float(eval_xgb['metrics']['AUC-PR']),
    },
    'optimal_threshold': 0.604,  # Sesuaikan dengan hasil tuning Anda
    'feature_order': feature_order,
}

with open('artifacts/training_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2, default=str)
print(f"✅ training_metadata.json")


# =============================================================================
# 5. PASTIKAN MODEL TERSIMPAN (jika belum)
# =============================================================================
if not os.path.exists('artifacts/model_xgb_full.pkl'):
    joblib.dump(xgb_best, 'artifacts/model_xgb_full.pkl')
    print(f"✅ model_xgb_full.pkl — di-copy ke artifacts/")
else:
    print(f"✅ model_xgb_full.pkl — sudah ada")


# =============================================================================
# 6. VERIFIKASI SEMUA FILE ADA
# =============================================================================
required_files = [
    'artifacts/model_xgb_full.pkl',
    'artifacts/feature_order.json',
    'artifacts/imputation_values.json',
    'artifacts/training_metadata.json',
]

print('\n' + '=' * 60)
print('VERIFIKASI ARTIFACTS UNTUK DEPLOYMENT')
print('=' * 60)
all_ok = True
for f in required_files:
    exists = os.path.exists(f)
    size_kb = os.path.getsize(f) / 1024 if exists else 0
    status = '✅' if exists else '❌'
    print(f'  {status} {f:<45} ({size_kb:.1f} KB)')
    if not exists:
        all_ok = False

print('=' * 60)
if all_ok:
    print('✅ SEMUA ARTIFACTS SIAP UNTUK DEPLOYMENT')
    print('\nLangkah selanjutnya:')
    print('  1. Download folder artifacts/ ke komputer lokal')
    print('  2. Copy folder artifacts/ ke project deployment')
    print('  3. Jalankan: streamlit run app.py')
else:
    print('❌ Ada artifacts yang belum tersimpan. Periksa lagi.')
