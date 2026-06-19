import './style.css'
import {
  DrawingUtils,
  FilesetResolver,
  GestureRecognizer,
  type GestureRecognizerResult,
  type NormalizedLandmark,
} from '@mediapipe/tasks-vision'
import {
  BadgeCheck,
  Bot,
  Copy,
  Crosshair,
  createIcons,
  FlipHorizontal2,
  Hand,
  HandFist,
  HandMetal,
  House,
  MonitorDot,
  MousePointer2,
  Radio,
  Redo2,
  ScanLine,
  ThumbsDown,
  ThumbsUp,
  Undo2,
  Waypoints,
  Webcam,
  Zap,
  ZoomIn,
  ZoomOut,
} from 'lucide'

type PresetId = 'browser' | 'stream' | 'agent' | 'graph' | 'home'

type GestureKey =
  | 'Pointing_Up'
  | 'Pinch'
  | 'Open_Palm'
  | 'Closed_Fist'
  | 'Victory'
  | 'Thumb_Down'
  | 'Thumb_Up'
  | 'ILoveYou'
  | 'Swipe_Left'
  | 'Swipe_Right'
  | 'Zoom_In'
  | 'Zoom_Out'

type GestureDefinition = {
  key: GestureKey
  title: string
  signal: string
  icon: string
}

type ActionMap = Record<GestureKey, string>

type GestureEvent = {
  id: number
  gesture: GestureKey
  gestureTitle: string
  preset: PresetId
  action: string
  confidence: number
  source: 'model' | 'landmarks' | 'motion' | 'manual'
  at: string
  details?: Record<string, number | string | boolean>
}

type MotionSample = {
  x: number
  y: number
  t: number
}

const tasksVersion = '0.10.35'
const wasmPath = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${tasksVersion}/wasm`
const gestureModelPath =
  'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task'

const usedIcons = {
  BadgeCheck,
  Bot,
  Copy,
  Crosshair,
  FlipHorizontal2,
  Hand,
  HandFist,
  HandMetal,
  House,
  MonitorDot,
  MousePointer2,
  Radio,
  Redo2,
  ScanLine,
  ThumbsDown,
  ThumbsUp,
  Undo2,
  Waypoints,
  Webcam,
  Zap,
  ZoomIn,
  ZoomOut,
}

const gestureDefinitions: GestureDefinition[] = [
  { key: 'Pointing_Up', title: 'Указатель', signal: 'MediaPipe', icon: 'mouse-pointer-2' },
  { key: 'Pinch', title: 'Щипок', signal: 'Landmarks', icon: 'crosshair' },
  { key: 'Open_Palm', title: 'Ладонь', signal: 'MediaPipe', icon: 'hand' },
  { key: 'Closed_Fist', title: 'Кулак', signal: 'MediaPipe', icon: 'hand-fist' },
  { key: 'Victory', title: 'Два пальца', signal: 'MediaPipe', icon: 'badge-check' },
  { key: 'Thumb_Down', title: 'Палец вниз', signal: 'MediaPipe', icon: 'thumbs-down' },
  { key: 'Thumb_Up', title: 'Палец вверх', signal: 'MediaPipe', icon: 'thumbs-up' },
  { key: 'ILoveYou', title: 'Рок-жест', signal: 'MediaPipe', icon: 'hand-metal' },
  { key: 'Swipe_Left', title: 'Свайп влево', signal: 'Motion', icon: 'undo-2' },
  { key: 'Swipe_Right', title: 'Свайп вправо', signal: 'Motion', icon: 'redo-2' },
  { key: 'Zoom_In', title: 'Развести руки', signal: 'Two hands', icon: 'zoom-in' },
  { key: 'Zoom_Out', title: 'Свести руки', signal: 'Two hands', icon: 'zoom-out' },
]

const gestureKeys = new Set(gestureDefinitions.map((gesture) => gesture.key))

const presets: Record<
  PresetId,
  {
    title: string
    icon: string
    actions: ActionMap
  }
> = {
  browser: {
    title: 'ПК',
    icon: 'monitor-dot',
    actions: {
      Pointing_Up: 'Воздушный курсор',
      Pinch: 'Левый клик',
      Open_Palm: 'Пауза курсора',
      Closed_Fist: 'Захват окна',
      Victory: 'Правый клик',
      Thumb_Down: 'Esc',
      Thumb_Up: 'Enter',
      ILoveYou: 'Скриншот области',
      Swipe_Left: 'Назад',
      Swipe_Right: 'Вперед',
      Zoom_In: 'Увеличить',
      Zoom_Out: 'Уменьшить',
    },
  },
  stream: {
    title: 'Стрим',
    icon: 'radio',
    actions: {
      Pointing_Up: 'Показать курсор',
      Pinch: 'Поставить маркер',
      Open_Palm: 'Показать баннер',
      Closed_Fist: 'Микрофон mute',
      Victory: 'Музыка play/pause',
      Thumb_Down: 'Скрыть баннер',
      Thumb_Up: 'Следующая сцена OBS',
      ILoveYou: 'Клип-анимация',
      Swipe_Left: 'Предыдущая сцена',
      Swipe_Right: 'Следующая сцена',
      Zoom_In: 'Увеличить камеру',
      Zoom_Out: 'Свернуть камеру',
    },
  },
  agent: {
    title: 'AI',
    icon: 'bot',
    actions: {
      Pointing_Up: 'Анализ окна',
      Pinch: 'Скриншот области',
      Open_Palm: 'Вызвать агента',
      Closed_Fist: 'Диктовка',
      Victory: 'Палитра инструментов',
      Thumb_Down: 'Отменить',
      Thumb_Up: 'Подтвердить',
      ILoveYou: 'Голосовой диалог',
      Swipe_Left: 'Предыдущий ответ',
      Swipe_Right: 'Следующий шаг',
      Zoom_In: 'Расширить контекст',
      Zoom_Out: 'Сжать контекст',
    },
  },
  graph: {
    title: 'Xedoc',
    icon: 'waypoints',
    actions: {
      Pointing_Up: 'Фокус на узел',
      Pinch: 'Создать связь',
      Open_Palm: 'Панорамирование',
      Closed_Fist: 'Перетащить узел',
      Victory: 'Открыть чат узла',
      Thumb_Down: 'Остановить агента',
      Thumb_Up: 'Запустить агента',
      ILoveYou: 'Раскрыть соседей',
      Swipe_Left: 'К прошлому узлу',
      Swipe_Right: 'К следующему узлу',
      Zoom_In: 'Зум графа плюс',
      Zoom_Out: 'Зум графа минус',
    },
  },
  home: {
    title: 'Дом',
    icon: 'house',
    actions: {
      Pointing_Up: 'Выбрать зону',
      Pinch: 'Точечный сценарий',
      Open_Palm: 'Выключить все',
      Closed_Fist: 'Сценарий стрим',
      Victory: 'Музыка в комнате',
      Thumb_Down: 'Свет выкл.',
      Thumb_Up: 'Свет toggle',
      ILoveYou: 'Кино-режим',
      Swipe_Left: 'Предыдущая сцена',
      Swipe_Right: 'Следующая сцена',
      Zoom_In: 'Яркость выше',
      Zoom_Out: 'Яркость ниже',
    },
  },
}

const app = document.querySelector<HTMLDivElement>('#app')

if (!app) {
  throw new Error('App root was not found')
}

app.innerHTML = `
  <div class="app-shell">
    <header class="topbar">
      <div class="brand">
        <span class="brand-mark"><i data-lucide="scan-line"></i></span>
        <div>
          <p class="eyebrow">hands.xedoc.ru</p>
          <h1>Xedoc Hands</h1>
        </div>
      </div>
      <div class="top-actions">
        <button class="button primary" id="cameraButton" type="button">
          <i data-lucide="webcam"></i>
          <span>Камера</span>
        </button>
        <button class="button" id="mirrorButton" type="button" title="Зеркальное отображение">
          <i data-lucide="flip-horizontal-2"></i>
          <span>Зеркало</span>
        </button>
        <button class="button" id="testButton" type="button">
          <i data-lucide="zap"></i>
          <span>Тест</span>
        </button>
      </div>
    </header>

    <main class="workspace">
      <section class="stage-area" aria-label="Камера и жесты">
        <div class="stage" id="stage">
          <video id="cameraVideo" playsinline muted></video>
          <canvas id="overlayCanvas"></canvas>
          <div class="cursor-dot" id="cursorDot"></div>
          <div class="stage-empty" id="stageEmpty">
            <i data-lucide="webcam"></i>
            <span>Камера выключена</span>
          </div>
          <div class="hud">
            <div>
              <span class="hud-label">Жест</span>
              <strong id="currentGesture">Нет руки</strong>
            </div>
            <div>
              <span class="hud-label">FPS</span>
              <strong id="fpsValue">0</strong>
            </div>
            <div>
              <span class="hud-label">Руки</span>
              <strong id="handCount">0</strong>
            </div>
          </div>
        </div>

        <div class="signal-row">
          <div class="meter">
            <div class="meter-top">
              <span>Confidence</span>
              <strong id="confidenceValue">0%</strong>
            </div>
            <span class="meter-track"><span id="confidenceMeter"></span></span>
          </div>
          <div class="meter">
            <div class="meter-top">
              <span>Pinch</span>
              <strong id="pinchValue">0%</strong>
            </div>
            <span class="meter-track"><span id="pinchMeter"></span></span>
          </div>
          <div class="meter">
            <div class="meter-top">
              <span>Motion</span>
              <strong id="motionValue">0%</strong>
            </div>
            <span class="meter-track"><span id="motionMeter"></span></span>
          </div>
        </div>
      </section>

      <aside class="control-area" aria-label="Настройки">
        <section class="panel status-panel">
          <div class="status-line">
            <span class="status-dot" id="modelDot"></span>
            <div>
              <span class="label">Модель</span>
              <strong id="modelStatus">Загрузка</strong>
            </div>
          </div>
          <div class="status-line">
            <span class="status-dot" id="cameraDot"></span>
            <div>
              <span class="label">Камера</span>
              <strong id="cameraStatus">Ожидание</strong>
            </div>
          </div>
        </section>

        <section class="panel">
          <div class="panel-head">
            <h2>Режим</h2>
            <span id="presetTitle">ПК</span>
          </div>
          <div class="preset-tabs" id="presetTabs"></div>
        </section>

        <section class="panel">
          <div class="panel-head">
            <h2>Webhook</h2>
            <label class="switch">
              <input id="webhookToggle" type="checkbox" />
              <span></span>
            </label>
          </div>
          <label class="input-label" for="webhookUrl">Endpoint</label>
          <input id="webhookUrl" class="text-input" type="url" value="http://127.0.0.1:8787/gesture" />
          <p class="webhook-state" id="webhookState">Локальный лог активен</p>
        </section>

        <section class="panel">
          <div class="panel-head">
            <h2>Последнее событие</h2>
            <button class="icon-button" id="copyButton" type="button" title="Скопировать JSON">
              <i data-lucide="copy"></i>
            </button>
          </div>
          <pre class="event-json" id="eventJson">{}</pre>
        </section>
      </aside>
    </main>

    <section class="gesture-strip" aria-label="Жесты и действия">
      <div class="strip-head">
        <h2>Жесты</h2>
        <p id="actionHint">Готово к маппингу команд</p>
      </div>
      <div class="gesture-grid" id="gestureGrid"></div>
    </section>

    <section class="event-log-band" aria-label="События">
      <div class="strip-head">
        <h2>События</h2>
        <p id="eventCount">0 команд</p>
      </div>
      <div class="event-log" id="eventLog"></div>
    </section>
  </div>
`

const stage = getElement<HTMLDivElement>('stage')
const video = getElement<HTMLVideoElement>('cameraVideo')
const canvas = getElement<HTMLCanvasElement>('overlayCanvas')
const canvasContext = canvas.getContext('2d')
const cursorDot = getElement<HTMLDivElement>('cursorDot')
const stageEmpty = getElement<HTMLDivElement>('stageEmpty')
const cameraButton = getElement<HTMLButtonElement>('cameraButton')
const mirrorButton = getElement<HTMLButtonElement>('mirrorButton')
const testButton = getElement<HTMLButtonElement>('testButton')
const modelDot = getElement<HTMLSpanElement>('modelDot')
const modelStatus = getElement<HTMLElement>('modelStatus')
const cameraDot = getElement<HTMLSpanElement>('cameraDot')
const cameraStatus = getElement<HTMLElement>('cameraStatus')
const currentGesture = getElement<HTMLElement>('currentGesture')
const fpsValue = getElement<HTMLElement>('fpsValue')
const handCount = getElement<HTMLElement>('handCount')
const confidenceValue = getElement<HTMLElement>('confidenceValue')
const confidenceMeter = getElement<HTMLSpanElement>('confidenceMeter')
const pinchValue = getElement<HTMLElement>('pinchValue')
const pinchMeter = getElement<HTMLSpanElement>('pinchMeter')
const motionValue = getElement<HTMLElement>('motionValue')
const motionMeter = getElement<HTMLSpanElement>('motionMeter')
const presetTitle = getElement<HTMLElement>('presetTitle')
const presetTabs = getElement<HTMLDivElement>('presetTabs')
const webhookToggle = getElement<HTMLInputElement>('webhookToggle')
const webhookUrl = getElement<HTMLInputElement>('webhookUrl')
const webhookState = getElement<HTMLElement>('webhookState')
const eventJson = getElement<HTMLPreElement>('eventJson')
const copyButton = getElement<HTMLButtonElement>('copyButton')
const gestureGrid = getElement<HTMLDivElement>('gestureGrid')
const actionHint = getElement<HTMLElement>('actionHint')
const eventLog = getElement<HTMLDivElement>('eventLog')
const eventCount = getElement<HTMLElement>('eventCount')

if (!canvasContext) {
  throw new Error('Canvas 2D context is not available')
}

const context: CanvasRenderingContext2D = canvasContext
const drawingUtils = new DrawingUtils(context)
let recognizer: GestureRecognizer | null = null
let stream: MediaStream | null = null
let isRunning = false
let lastVideoTime = -1
let currentPreset: PresetId = readPreset()
let mirrorMode = localStorage.getItem('xedoc-hands-mirror') !== 'off'
let lastEvent: GestureEvent | null = null
let eventSequence = 0
let frameCount = 0
let fpsStartedAt = performance.now()
let pinchDown = false
let motionSamples: MotionSample[] = []
let zoomSamples: MotionSample[] = []
const eventHistory: GestureEvent[] = []
const cooldowns = new Map<GestureKey, number>()

renderPresetTabs()
renderGestureGrid()
refreshIcons()
setMirrorMode(mirrorMode)
setModelState('loading', 'Загрузка')
setCameraState('idle', 'Ожидание')
void bootModel()

cameraButton.addEventListener('click', () => {
  if (isRunning) {
    stopCamera()
    return
  }

  void startCamera()
})

mirrorButton.addEventListener('click', () => {
  setMirrorMode(!mirrorMode)
})

testButton.addEventListener('click', () => {
  fireGesture('Thumb_Up', 1, 'manual', { test: true })
})

copyButton.addEventListener('click', () => {
  void copyLastEvent()
})

webhookToggle.addEventListener('change', () => {
  localStorage.setItem('xedoc-hands-webhook-enabled', String(webhookToggle.checked))
  webhookState.textContent = webhookToggle.checked ? 'Webhook включен' : 'Локальный лог активен'
})

webhookUrl.addEventListener('change', () => {
  localStorage.setItem('xedoc-hands-webhook-url', webhookUrl.value)
})

window.addEventListener('resize', syncCanvasSize)

const savedWebhook = localStorage.getItem('xedoc-hands-webhook-url')
const savedWebhookEnabled = localStorage.getItem('xedoc-hands-webhook-enabled') === 'true'

if (savedWebhook) {
  webhookUrl.value = savedWebhook
}

webhookToggle.checked = savedWebhookEnabled
webhookState.textContent = savedWebhookEnabled ? 'Webhook включен' : 'Локальный лог активен'

async function bootModel() {
  try {
    const vision = await FilesetResolver.forVisionTasks(wasmPath)

    recognizer = await GestureRecognizer.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: gestureModelPath,
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numHands: 2,
      minHandDetectionConfidence: 0.55,
      minHandPresenceConfidence: 0.55,
      minTrackingConfidence: 0.55,
    })

    setModelState('ready', 'Готова')
  } catch (error) {
    console.error(error)
    setModelState('error', 'Ошибка')
  }
}

async function startCamera() {
  if (!recognizer) {
    setCameraState('loading', 'Ждем модель')
    await bootModel()
  }

  if (!recognizer) {
    setCameraState('error', 'Модель недоступна')
    return
  }

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: 'user',
      },
    })

    video.srcObject = stream
    await video.play()
    isRunning = true
    cameraButton.classList.add('is-active')
    cameraButton.querySelector('span')!.textContent = 'Стоп'
    stageEmpty.classList.add('is-hidden')
    setCameraState('ready', 'Работает')
    syncCanvasSize()
    requestAnimationFrame(predictFrame)
  } catch (error) {
    console.error(error)
    setCameraState('error', 'Нет доступа')
  }
}

function stopCamera() {
  isRunning = false
  lastVideoTime = -1
  motionSamples = []
  zoomSamples = []
  pinchDown = false
  stream?.getTracks().forEach((track) => track.stop())
  stream = null
  video.srcObject = null
  context.clearRect(0, 0, canvas.width, canvas.height)
  cursorDot.classList.remove('is-visible')
  stageEmpty.classList.remove('is-hidden')
  cameraButton.classList.remove('is-active')
  cameraButton.querySelector('span')!.textContent = 'Камера'
  setCameraState('idle', 'Ожидание')
  setReadout('Нет руки', 0, 0, 0, 0, 0)
  setActiveGestures(new Set())
}

function predictFrame(now: number) {
  if (!isRunning || !recognizer) {
    return
  }

  syncCanvasSize()

  if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime
    const result = recognizer.recognizeForVideo(video, now)
    processResult(result, now)
    updateFps(now)
  }

  requestAnimationFrame(predictFrame)
}

function processResult(result: GestureRecognizerResult, now: number) {
  context.clearRect(0, 0, canvas.width, canvas.height)

  const detected = new Set<GestureKey>()
  const topGesture = result.gestures[0]?.[0]
  const topScore = topGesture?.score ?? 0
  const modelGesture = topGesture?.categoryName ?? 'None'
  const landmarks = result.landmarks[0]

  for (const hand of result.landmarks) {
    drawingUtils.drawConnectors(hand, GestureRecognizer.HAND_CONNECTIONS, {
      color: 'rgba(74, 222, 128, 0.9)',
      lineWidth: 4,
    })
    drawingUtils.drawLandmarks(hand, {
      color: 'rgba(255, 255, 255, 0.95)',
      fillColor: 'rgba(12, 20, 24, 0.9)',
      lineWidth: 2,
      radius: 4,
    })
  }

  if (isGestureKey(modelGesture) && topScore > 0.58) {
    detected.add(modelGesture)
    fireGesture(modelGesture, topScore, 'model')
  }

  let pinchStrength = 0
  let motionStrength = 0

  if (landmarks) {
    const pinch = updatePinch(landmarks)
    pinchStrength = pinch.strength

    if (pinch.active) {
      detected.add('Pinch')
    }

    motionStrength = updateMotion(landmarks, now)
    updateCursor(landmarks)
  } else {
    pinchDown = false
    motionSamples = []
    cursorDot.classList.remove('is-visible')
  }

  updateZoom(result.landmarks, now, detected)
  setActiveGestures(detected)

  const gestureTitle =
    detected.size > 0
      ? [...detected].map((gesture) => gestureTitleFor(gesture)).join(' + ')
      : 'Нет руки'

  setReadout(gestureTitle, topScore, result.landmarks.length, pinchStrength, motionStrength, detected.size)
}

function updatePinch(landmarks: NormalizedLandmark[]) {
  const thumbTip = landmarks[4]
  const indexTip = landmarks[8]
  const indexMcp = landmarks[5]
  const pinkyMcp = landmarks[17]
  const palmWidth = Math.max(distance(indexMcp, pinkyMcp), 0.001)
  const ratio = distance(thumbTip, indexTip) / palmWidth
  const strength = clamp(1 - (ratio - 0.25) / 0.55, 0, 1)
  const active = ratio < 0.5

  if (ratio < 0.42 && !pinchDown) {
    pinchDown = true
    fireGesture('Pinch', strength, 'landmarks', { ratio: round(ratio), strength: round(strength) })
  }

  if (ratio > 0.62) {
    pinchDown = false
  }

  return { active, strength }
}

function updateMotion(landmarks: NormalizedLandmark[], now: number) {
  const pointer = landmarks[8]
  const sample = {
    x: mirrorMode ? 1 - pointer.x : pointer.x,
    y: pointer.y,
    t: now,
  }

  motionSamples.push(sample)
  motionSamples = motionSamples.filter((item) => now - item.t <= 520)

  const first = motionSamples[0]

  if (!first) {
    return 0
  }

  const dx = sample.x - first.x
  const dy = sample.y - first.y
  const strength = clamp(Math.abs(dx) / 0.26, 0, 1)

  if (Math.abs(dx) > 0.22 && Math.abs(dx) > Math.abs(dy) * 1.45) {
    const gesture: GestureKey = dx > 0 ? 'Swipe_Right' : 'Swipe_Left'
    fireGesture(gesture, strength, 'motion', { dx: round(dx), dy: round(dy) })
    motionSamples = []
  }

  return strength
}

function updateZoom(hands: NormalizedLandmark[][], now: number, detected: Set<GestureKey>) {
  if (hands.length < 2) {
    zoomSamples = []
    return
  }

  const first = handCenter(hands[0])
  const second = handCenter(hands[1])
  const span = distance(first, second)
  zoomSamples.push({ x: span, y: 0, t: now })
  zoomSamples = zoomSamples.filter((item) => now - item.t <= 700)

  const start = zoomSamples[0]

  if (!start) {
    return
  }

  const delta = span - start.x
  const strength = clamp(Math.abs(delta) / 0.16, 0, 1)

  if (Math.abs(delta) > 0.12) {
    const gesture: GestureKey = delta > 0 ? 'Zoom_In' : 'Zoom_Out'
    detected.add(gesture)
    fireGesture(gesture, strength, 'motion', { delta: round(delta), span: round(span) })
    zoomSamples = []
  }
}

function updateCursor(landmarks: NormalizedLandmark[]) {
  const pointer = landmarks[8]
  const x = mirrorMode ? 1 - pointer.x : pointer.x
  const y = pointer.y

  cursorDot.style.left = `${clamp(x, 0, 1) * 100}%`
  cursorDot.style.top = `${clamp(y, 0, 1) * 100}%`
  cursorDot.classList.add('is-visible')
}

function fireGesture(
  gesture: GestureKey,
  confidence: number,
  source: GestureEvent['source'],
  details?: GestureEvent['details'],
) {
  const now = performance.now()
  const cooldown = source === 'model' ? 1150 : 760
  const lastFired = cooldowns.get(gesture) ?? 0

  if (now - lastFired < cooldown) {
    return
  }

  cooldowns.set(gesture, now)

  const event: GestureEvent = {
    id: ++eventSequence,
    gesture,
    gestureTitle: gestureTitleFor(gesture),
    preset: currentPreset,
    action: presets[currentPreset].actions[gesture],
    confidence: round(confidence),
    source,
    at: new Date().toISOString(),
    details,
  }

  lastEvent = event
  eventHistory.unshift(event)
  eventHistory.splice(12)
  renderLastEvent()
  renderLog()

  if (webhookToggle.checked) {
    void sendWebhook(event)
  }
}

async function sendWebhook(event: GestureEvent) {
  const url = webhookUrl.value.trim()

  if (!url) {
    webhookState.textContent = 'Webhook: нет endpoint'
    return
  }

  try {
    await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(event),
    })
    webhookState.textContent = `Webhook: ${gestureTitleFor(event.gesture)}`
  } catch (error) {
    console.error(error)
    webhookState.textContent = 'Webhook: ошибка CORS/сети'
  }
}

async function copyLastEvent() {
  if (!lastEvent) {
    return
  }

  try {
    await navigator.clipboard.writeText(JSON.stringify(lastEvent, null, 2))
    copyButton.classList.add('is-done')
    setTimeout(() => copyButton.classList.remove('is-done'), 900)
  } catch (error) {
    console.error(error)
  }
}

function renderPresetTabs() {
  presetTabs.innerHTML = Object.entries(presets)
    .map(
      ([id, preset]) => `
        <button class="preset-tab${id === currentPreset ? ' is-active' : ''}" data-preset="${id}" type="button">
          <i data-lucide="${preset.icon}"></i>
          <span>${preset.title}</span>
        </button>
      `,
    )
    .join('')

  presetTitle.textContent = presets[currentPreset].title

  presetTabs.querySelectorAll<HTMLButtonElement>('.preset-tab').forEach((button) => {
    button.addEventListener('click', () => {
      currentPreset = button.dataset.preset as PresetId
      localStorage.setItem('xedoc-hands-preset', currentPreset)
      renderPresetTabs()
      renderGestureGrid()
      refreshIcons()
    })
  })
}

function renderGestureGrid() {
  const actions = presets[currentPreset].actions

  gestureGrid.innerHTML = gestureDefinitions
    .map(
      (gesture) => `
        <article class="gesture-tile" data-gesture="${gesture.key}">
          <div class="tile-icon"><i data-lucide="${gesture.icon}"></i></div>
          <div>
            <span class="tile-title">${gesture.title}</span>
            <strong>${actions[gesture.key]}</strong>
            <em>${gesture.signal}</em>
          </div>
        </article>
      `,
    )
    .join('')

  actionHint.textContent = `${presets[currentPreset].title}: ${gestureDefinitions.length} команд`
}

function renderLastEvent() {
  eventJson.textContent = lastEvent ? JSON.stringify(lastEvent, null, 2) : '{}'
}

function renderLog() {
  eventCount.textContent = `${eventHistory.length} команд`
  eventLog.innerHTML =
    eventHistory
      .map(
        (event) => `
          <article class="event-row">
            <span>${String(event.id).padStart(2, '0')}</span>
            <strong>${escapeHtml(event.gestureTitle)}</strong>
            <em>${escapeHtml(event.action)}</em>
            <time>${new Date(event.at).toLocaleTimeString('ru-RU')}</time>
          </article>
        `,
      )
      .join('') || '<p class="empty-log">Команд пока нет</p>'
}

function setActiveGestures(active: Set<GestureKey>) {
  gestureGrid.querySelectorAll<HTMLElement>('.gesture-tile').forEach((tile) => {
    const key = tile.dataset.gesture as GestureKey
    tile.classList.toggle('is-live', active.has(key))
  })
}

function setReadout(
  gestureTitle: string,
  confidence: number,
  hands: number,
  pinchStrength: number,
  motionStrength: number,
  activeCount: number,
) {
  currentGesture.textContent = gestureTitle
  handCount.textContent = String(hands)
  setMeter(confidenceMeter, confidenceValue, confidence)
  setMeter(pinchMeter, pinchValue, pinchStrength)
  setMeter(motionMeter, motionValue, motionStrength)
  stage.dataset.active = activeCount > 0 ? 'true' : 'false'
}

function setMeter(track: HTMLElement, label: HTMLElement, value: number) {
  const percent = Math.round(clamp(value, 0, 1) * 100)
  track.style.width = `${percent}%`
  label.textContent = `${percent}%`
}

function setModelState(state: 'loading' | 'ready' | 'error', text: string) {
  modelDot.dataset.state = state
  modelStatus.textContent = text
}

function setCameraState(state: 'idle' | 'loading' | 'ready' | 'error', text: string) {
  cameraDot.dataset.state = state
  cameraStatus.textContent = text
}

function setMirrorMode(next: boolean) {
  mirrorMode = next
  localStorage.setItem('xedoc-hands-mirror', next ? 'on' : 'off')
  stage.classList.toggle('is-mirrored', next)
  mirrorButton.classList.toggle('is-active', next)
}

function syncCanvasSize() {
  if (!video.videoWidth || !video.videoHeight) {
    return
  }

  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    stage.style.setProperty('--camera-aspect', `${video.videoWidth} / ${video.videoHeight}`)
  }
}

function updateFps(now: number) {
  frameCount += 1

  if (now - fpsStartedAt < 1000) {
    return
  }

  fpsValue.textContent = String(Math.round((frameCount * 1000) / (now - fpsStartedAt)))
  frameCount = 0
  fpsStartedAt = now
}

function readPreset(): PresetId {
  const saved = localStorage.getItem('xedoc-hands-preset')
  return saved && saved in presets ? (saved as PresetId) : 'browser'
}

function getElement<T extends HTMLElement>(id: string) {
  const element = document.getElementById(id)

  if (!element) {
    throw new Error(`Element #${id} was not found`)
  }

  return element as T
}

function refreshIcons() {
  createIcons({ icons: usedIcons })
}

function isGestureKey(value: string): value is GestureKey {
  return gestureKeys.has(value as GestureKey)
}

function gestureTitleFor(gesture: GestureKey) {
  return gestureDefinitions.find((definition) => definition.key === gesture)?.title ?? gesture
}

function distance(a: NormalizedLandmark, b: NormalizedLandmark) {
  return Math.hypot(a.x - b.x, a.y - b.y)
}

function handCenter(landmarks: NormalizedLandmark[]) {
  const wrist = landmarks[0]
  const middleBase = landmarks[9]
  return {
    x: (wrist.x + middleBase.x) / 2,
    y: (wrist.y + middleBase.y) / 2,
    z: (wrist.z + middleBase.z) / 2,
    visibility: Math.min(wrist.visibility ?? 1, middleBase.visibility ?? 1),
  }
}

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max)
}

function round(value: number) {
  return Math.round(value * 1000) / 1000
}

function escapeHtml(value: string) {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;')
}
