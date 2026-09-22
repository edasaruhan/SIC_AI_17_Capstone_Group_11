# Deployment

Bu klasör, eğitilmiş LightGBM modellerini FastAPI üzerinden nokta tahmini ve
kalibre tahmin aralığı üreten bir servise dönüştürür. Yerel demo arayüzü `/`
adresinden, etkileşimli API arayüzü `/docs` adresinden açılır. Saha Ekranı,
mobil tasarımın merkezi mutfaklara uyarlanmış halidir: yemek kodu ve gerçek
haftalık siparişler girildiğinde `/predict` sonucunu gösterir. Kamera/barkod,
stok yönetimi ve elektronik raf etiketi bağlantısı bu projede yoktur.

## Yerel kurulum

Komutları repository kökünden çalıştırın:

```powershell
python -m pip install -r deploy/requirements-dev.txt
python deploy/scripts/export_artifacts.py
python -m pytest deploy/tests -q
$env:API_KEYS = "local-demo-key"
python -m uvicorn deploy.app.main:app --port 8000
```

Kontrol adresleri:

- `http://localhost:8000/health`
- `http://localhost:8000/#saha` (mobil saha demosu)
- `http://localhost:8000/docs`
- `http://localhost:8000/metrics` (`X-API-Key` gerekir)

API anahtarlarını yalnızca ortam değişkeninde veya platformun secret deposunda
tutun. `.env` dosyası Git'e eklenmez.

## Docker

Docker build context repository köküdür:

```powershell
docker build -f deploy/Dockerfile -t food-demand-api .
docker run --rm -p 8000:8000 -e API_KEYS=local-demo-key food-demand-api
```

Compose kullanmak için `deploy/.env.example` dosyasını `deploy/.env` olarak
kopyalayıp güvenli bir anahtar belirleyin, ardından:

```powershell
docker compose -f deploy/compose.yaml up --build
```

Container root olmayan `app` kullanıcısıyla çalışır, health check içerir ve
model dosyalarının SHA-256 özetlerini açılışta doğrular.

## API

| Metot | Yol | Açıklama | Kimlik doğrulama |
| --- | --- | --- | --- |
| GET | `/health` | Servis ve model durumu | Hayır |
| GET | `/model-info` | Özellikler, kategoriler ve model bilgisi | API key |
| POST | `/predict` | Tek tahmin ve P8/P50/P92 aralığı | API key |
| POST | `/predict/batch` | En fazla 500 tahmin | API key |
| GET | `/metrics` | Prometheus metrikleri | API key |

İstek, statik ürün/merkez alanlarının yanında aynı merkez-yemek çifti için
eskiden yeniye sıralanmış en fazla 13 `recent_orders` değeri alır. API,
`num_orders_lag_1`, `num_orders_lag_4` ve `num_orders_roll_mean_4`
özelliklerini bu geçmişten üretir. Dört haftadan az geçmiş cold-start uyarısı
oluşturur.

## Monitoring

Tahminler varsayılan olarak `deploy/logs/predictions.jsonl` dosyasına yazılır.
Drift raporu en az 200 tahmin olmadan karar üretmez:

```powershell
python deploy/monitoring/drift_report.py `
  --reference data/test_with_quantile_pred.parquet `
  --log deploy/logs/predictions.jsonl `
  --out deploy/logs/drift_report.json
```

Gerçek siparişler elde edildiğinde performans raporu RMSLE, MAE, WAPE, bias,
fazla/eksik tahmin oranı ve tahmin aralığı kapsamasını hesaplar:

```powershell
python deploy/monitoring/performance_report.py `
  --log deploy/logs/predictions.jsonl `
  --actuals deploy/monitoring/sample_actuals.csv `
  --out deploy/logs/performance_report.json
```

`sample_actuals.csv` yalnızca beklenen dosya şemasını gösterir. Gerçek
monitoring değerlendirmesi için aynı hafta, merkez ve yemeklere ait POS/satış
sonuçları kullanılmalıdır.

## Sınırlamalar

- Hız sınırı tek process belleğindedir; çoklu instance için API gateway veya
  Redis gerekir.
- API tek haftalık tahmin verir. Çok haftalı tahminler istemci tarafından
  recursive çağrılarla oluşturulur.
- Quantile bandı tek adımlı validasyonda kalibre edilmiştir; uzun recursive
  ufuklarda yeniden kalibrasyon gerekir.
- Üretim ortamında model dosyaları Git yerine nesne depolama, DVC veya MLflow
  üzerinden sürümlenebilir.
