# AI for Food Waste Reduction

Samsung Innovation Campus AI 17 Capstone Group 11

Bu proje, yemek siparişi talebini tahmin ederek fazla üretim ve gıda israfı riskini azaltmayı amaçlar. Çalışma; veri hazırlama, özellik mühendisliği, LightGBM tabanlı nokta tahmini, quantile tahmin aralıkları, hiperparametre ayarı, sızıntısız zaman serisi doğrulaması ve gerçek Kaggle test kümesi için submission üretimini kapsar.

## Final Results

| Değerlendirme | Sonuç |
| --- | ---: |
| Baseline holdout RMSLE | 0.47628 |
| Optuna tuned holdout RMSLE | 0.47534 |
| Holdout RMSLE iyileşmesi | %0.20 |
| Kalibre quantile band coverage | %80.59 |
| Recursive walk-forward ortalama RMSLE | 0.49833 |
| Persistence baseline ortalama RMSLE | 0.80930 |
| Persistence baseline'a göre iyileşme | %38.32 |
| Final submission satır sayısı | 32.573 |

Holdout sonucu, gerçek hedeflerin yalnızca değerlendirme için bulunduğu tek bir zaman dilimini kullanır. Recursive walk-forward sonucu daha zor bir protokoldür: her doğrulama haftasının lag özellikleri, gelecekteki gerçek hedefler yerine önceki tahminlerden oluşturulur. Bu nedenle iki RMSLE değeri doğrudan aynı deney gibi karşılaştırılmamalıdır.

## Repository Structure

```text
data/                    Eğitilmiş modeller, encoder'lar ve deney sonuçları
deploy/                  FastAPI, Docker, güvenlik ve monitoring paketi
extensions/forecasting/ Leakage-aware doğrulama ve gerçek submission pipeline'ı
notebooks/               Faz 1-4 notebook'ları ve yeniden üretim betikleri
reports/                 Proje raporları, sonuç JSON'ları ve görseller
submissions/             Final Kaggle submission ve üretim manifesti
```

## Setup

Python 3.11 veya 3.12 önerilir.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Kaggle Food Demand Forecasting veri setindeki aşağıdaki dosyaları `data/` klasörüne kopyalayın:

- `train.csv`
- `test.csv`
- `meal_info.csv`
- `fulfilment_center_info.csv`
- `sample_submission.csv`

CSV dosyaları veri lisansı ve dosya boyutu nedeniyle Git deposuna dahil edilmez.

## Reproduce Model Refinement

Komutlar repository kökünden çalıştırılabilir:

```powershell
python notebooks/repro_01_02.py
python notebooks/repro_03_baseline.py
python notebooks/refine_01_hpo.py
python notebooks/refine_02_final_eval.py
python notebooks/refine_03_quantile_calibration.py
python notebooks/refine_04_final_calibration.py
```

İlk iki komut veri hazırlama ve baseline sonuçlarını üretir. Sonraki komutlar Optuna aramasını, tarafsız holdout değerlendirmesini ve quantile coverage kalibrasyonunu çalıştırır.

## Run Tests

```powershell
python -m unittest discover -s extensions/forecasting/tests -v
```

Test paketi lag zamanlamasını, recursive geçmiş güncellemesini, metrikleri, submission sırasını ve hata analizi yardımcılarını denetler.

## Run Leakage Aware Validation

```powershell
python extensions/forecasting/walk_forward_validation.py --data-dir data --output reports/results/walk_forward_metrics.json
```

Varsayılan çalışma, eğitim bitiş haftaları 115, 125 ve 135 olan üç expanding-window fold kullanır. Her fold sonraki 10 haftayı recursive olarak tahmin eder.

## Build Final Submission

```powershell
python extensions/forecasting/build_submission.py --data-dir data --output submissions/final_submission.csv --manifest submissions/final_submission_manifest.json --save-model data/final_submission_lgb_model.txt
```

Bu komut haftalar 1-145 üzerindeki 456.548 etiketli satırla modeli eğitir ve haftalar 146-155 için 32.573 tahmin üretir. Submission kimlik sırası `sample_submission.csv` ile doğrulanır; eksik, tekrarlı veya negatif tahminler hata oluşturur.

## Deployment

Deployment artefaktlarını hazırlayıp API testlerini çalıştırın:

```powershell
python -m pip install -r deploy/requirements-dev.txt
python deploy/scripts/export_artifacts.py
python -m pytest deploy/tests -q
```

Docker imajı repository kökünden oluşturulur:

```powershell
docker build -f deploy/Dockerfile -t food-demand-api .
docker run --rm -p 8000:8000 -e API_KEYS=local-demo-key food-demand-api
```

Çalışan servisin health endpoint'i `http://localhost:8000/health`, etkileşimli Swagger arayüzü ise `http://localhost:8000/docs` adresindedir. Ayrıntılar için `deploy/README.md` dosyasına bakın.

## Final Deliverables

- `reports/Model Refinement and Test Submission.pdf`
- `reports/Model Refinement and Test Submission.docx`
- `reports/Deployment Documentation TR.pdf`
- `reports/results/walk_forward_metrics.json`
- `submissions/final_submission.csv`
- `submissions/final_submission_manifest.json`
- `data/final_submission_lgb_model.txt`
- `deploy/README.md`
- `deploy/Dockerfile`
- `deploy/app/main.py`

## Methodology Notes

- Rastgele k-fold kullanılmaz; zaman sırası korunur.
- Hiperparametre araması gerçek holdout dönemine bakmadan iç-validasyon üzerinde yapılır.
- Kaggle `test.csv` hedef içermediği için model seçimi veya metrik raporlama amacıyla kullanılmaz.
- Final submission lag özellikleri, gelecekte bulunmayan hedefler yerine önceki haftaların tahminlerinden recursive olarak üretilir.
- Quantile alpha kalibrasyonu ayrı bir kalibrasyon setinde tekrar doğrulanmadığı için coverage sonucu sınırlılık olarak raporlanır.

## Team

- Osman Furkan Erkan
- Nermin Kılıçarslan
- Dilara Kuyar
- Ahsen Nisa Sarıkaya

Repository: https://github.com/edasaruhan/SIC_AI_17_Capstone_Group_11
