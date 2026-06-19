# Xedoc Hands FaceSwap Bridge

Локальный bridge для режима FaceSwap в Xedoc Hands. Сайт отправляет текущий кадр камеры и загруженную картинку-источник на `http://127.0.0.1:8790/swap`, а bridge возвращает обработанный JPEG.

Это не DeepFace в браузере: DeepFace чаще используют для распознавания и верификации лиц. Для свапа здесь подготовлен каркас под InsightFace `inswapper_128.onnx`.

## Запуск

```powershell
cd bridge\faceswap-bridge
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn server:app --host 127.0.0.1 --port 8790
```

Модель свапа положи в `bridge/faceswap-bridge/models/inswapper_128.onnx` или укажи путь:

```powershell
$env:INSWAPPER_MODEL="C:\models\inswapper_128.onnx"
uvicorn server:app --host 127.0.0.1 --port 8790
```

В сайте выбери режим `FaceSwap`, загрузи изображение маски и включи тумблер маски. Endpoint по умолчанию уже стоит `http://127.0.0.1:8790/swap`.

## Проверка

```powershell
curl http://127.0.0.1:8790/health
```

Если сайт открыт по HTTPS и браузер заблокирует запрос к локальному HTTP, запусти сайт локально через Vite или подними bridge с локальным HTTPS-сертификатом.
