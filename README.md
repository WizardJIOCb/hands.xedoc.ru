<img width="1280" height="1226" alt="image" src="https://github.com/user-attachments/assets/5c2cd8a4-6f47-4a86-b3e4-326e34aea475" />
<img width="1280" height="1174" alt="image" src="https://github.com/user-attachments/assets/feb69c84-03a7-440a-9b01-a6bb31374399" />

# Xedoc Hands

Веб-прототип управления жестами и лицевыми сигналами через камеру.

Сайт: https://hands.xedoc.ru

## Что умеет

- Трекинг рук через MediaPipe Gesture Recognizer.
- Отдельные переключатели для трекинга рук и отрисовки линий/точек рук.
- Жесты рук: указатель, щипок, ладонь, кулак, два/три/четыре пальца, палец вверх/вниз, OK, свайпы, зум двумя руками и другие.
- Трекинг лица через MediaPipe Face Landmarker.
- Лицевые сигналы: улыбка, открытый рот, моргание, брови вверх, повороты головы, кивок и покачивание.
- Режим нескольких лиц: до 4 лиц в кадре.
- Пресеты действий: ПК, Стрим, AI, Xedoc, Дом.
- Webhook для отправки событий жестов на локальный или внешний endpoint.
- Режимы производительности: 640x480 для скорости и 1280x720 для качества.
- Автоматический performance-mode на мобильных устройствах.

## Маски

Есть два режима маски:

- `Mesh` - браузерная маска поверх лица по landmarks.
- `FaceSwap` - отправка кадра в локальный Python bridge для нейросетевого свапа.

В режиме `Mesh` можно загрузить основное изображение маски. Оно применяется ко всем найденным лицам. Для нескольких лиц есть дополнительные слоты:

- `Лицо 2`
- `Лицо 3`
- `Лицо 4`

Лица сортируются по близости к камере: `Лицо 1` - самое крупное, `Лицо 2` - следующее и так далее. Если отдельный слот пустой, лицо использует основную маску.

Настройки маски сейчас общие для всех лиц:

- стабилизация;
- мягкий край;
- цвет лица;
- яркость;
- насыщенность.

Отдельно по лицам сейчас выбирается только изображение маски.

## Webhook

При включенном webhook каждое событие отправляется `POST`-запросом в JSON-формате на указанный endpoint.

Пример endpoint по умолчанию:

```text
http://127.0.0.1:8787/gesture
```

Пример события:

```json
{
  "id": 154,
  "gesture": "Head_Right",
  "gestureTitle": "Голова вправо",
  "preset": "browser",
  "action": "Вкладка вперед",
  "confidence": 0.156,
  "source": "face",
  "at": "2026-06-20T08:09:26.827Z",
  "details": {
    "yaw": 0.204
  }
}
```

## Локальный запуск

```powershell
npm install
npm run dev
```

Сборка:

```powershell
npm run build
```

Локальный preview production-сборки:

```powershell
npm run preview
```

## FaceSwap bridge

Bridge находится в `bridge/faceswap-bridge`.

Запуск:

```powershell
cd bridge\faceswap-bridge
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn server:app --host 127.0.0.1 --port 8790
```

Модель `inswapper_128.onnx` нужно положить в:

```text
bridge/faceswap-bridge/models/inswapper_128.onnx
```

Или указать путь через переменную окружения:

```powershell
$env:INSWAPPER_MODEL="C:\models\inswapper_128.onnx"
uvicorn server:app --host 127.0.0.1 --port 8790
```

Проверка:

```powershell
curl http://127.0.0.1:8790/health
```

В сайте выбери режим `FaceSwap`, загрузи изображение и включи маску. Endpoint по умолчанию:

```text
http://127.0.0.1:8790/swap
```

Если страница открыта по HTTPS, браузер может блокировать запросы к локальному HTTP bridge. В таком случае удобнее тестировать сайт локально через Vite или поднимать bridge с локальным HTTPS.

## Деплой

Production-сборка лежит в `dist`.

```powershell
npm run build
scp -r dist\index.html dist\assets root@132.243.118.214:/var/www/hands.xedoc.ru/
```

После копирования на сервер нужно выставить права на assets:

```bash
find /var/www/hands.xedoc.ru/assets -type d -exec chmod 755 {} \;
find /var/www/hands.xedoc.ru/assets -type f -exec chmod 644 {} \;
```

Nginx-конфиг лежит в:

```text
deploy/nginx/hands.xedoc.ru.conf
```

## Стек

- Vite
- TypeScript
- MediaPipe Tasks Vision
- Lucide icons
- FastAPI bridge для FaceSwap
