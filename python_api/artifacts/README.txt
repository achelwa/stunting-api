# Folder ini harus diisi dengan artifacts dari notebook training Anda:
#
# 1. model_xgb_full.pkl       — Model XGBoost terbaik
# 2. feature_order.json       — Urutan 20+ fitur sesuai training
# 3. imputation_values.json   — Median/modus untuk imputasi missing values
#
# CARA GENERATE: 
# 1. Jalankan save_artifacts.py di notebook training (marsel_nitip.ipynb)
# 2. Download folder artifacts/ dari Colab
# 3. Replace file README.txt ini dengan isi yang sudah di-download
