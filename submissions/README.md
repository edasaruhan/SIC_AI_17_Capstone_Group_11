# Final Submission

`final_submission.csv`, Kaggle Food Demand Forecasting test kümesindeki haftalar 146-155 için üretilen nihai tahmin dosyasıdır.

## Production Record

- Eğitim dönemi: hafta 1-145
- Eğitim satırı: 456.548
- Test dönemi: hafta 146-155
- Submission satırı: 32.573
- LightGBM boosting round: 1.360
- Ortalama tahmin: 234,45 sipariş
- Minimum tahmin: 12,88 sipariş
- Maksimum tahmin: 7.050,18 sipariş

`final_submission_manifest.json`, üretim zamanı, kullanılan dönemler, satır sayıları ve haftalık tahmin özetlerini kaydeder. Model dosyası `data/final_submission_lgb_model.txt` altında saklanır.

Submission yeniden üretmek için repository kökünden şu komutu çalıştırın:

```powershell
python extensions/forecasting/build_submission.py --data-dir data --output submissions/final_submission.csv --manifest submissions/final_submission_manifest.json --save-model data/final_submission_lgb_model.txt
```

Pipeline, `sample_submission.csv` kimlik sırasını korur ve eksik, tekrarlı veya negatif tahminleri reddeder.
