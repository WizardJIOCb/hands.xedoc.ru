import './style.css'
import {
  DrawingUtils,
  FaceLandmarker,
  FilesetResolver,
  GestureRecognizer,
  type Category,
  type FaceLandmarkerResult,
  type GestureRecognizerResult,
  type NormalizedLandmark,
} from '@mediapipe/tasks-vision'
import {
  ArrowDown,
  ArrowUp,
  BadgeCheck,
  Bot,
  ChevronsUp,
  CircleCheck,
  CircleDot,
  Copy,
  Crosshair,
  createIcons,
  EyeClosed,
  FlipHorizontal2,
  Hand,
  HandFist,
  HandMetal,
  House,
  MessageCircle,
  MonitorDot,
  MoveDown,
  MoveLeft,
  MoveRight,
  MoveUp,
  MousePointer2,
  Radio,
  Redo2,
  ScanEye,
  ScanLine,
  Smile,
  Target,
  ThumbsDown,
  ThumbsUp,
  Timer,
  Undo2,
  Waypoints,
  Webcam,
  ZoomIn,
  ZoomOut,
} from 'lucide'

type PresetId = 'browser' | 'stream' | 'agent' | 'graph' | 'home'
type MaskMode = 'mesh' | 'faceswap'
type PerformanceMode = 'performance' | 'quality'

type GestureKey =
  | 'Pointing_Up'
  | 'Pinch'
  | 'Open_Palm'
  | 'Closed_Fist'
  | 'Victory'
  | 'Thumb_Down'
  | 'Thumb_Up'
  | 'ILoveYou'
  | 'OK_Gesture'
  | 'Finger_Gun'
  | 'Three_Fingers'
  | 'Four_Fingers'
  | 'Pinch_Hold'
  | 'Swipe_Left'
  | 'Swipe_Right'
  | 'Swipe_Up'
  | 'Swipe_Down'
  | 'Palm_Up'
  | 'Palm_Down'
  | 'Zoom_In'
  | 'Zoom_Out'
  | 'Smile'
  | 'Mouth_Open'
  | 'Blink_Left'
  | 'Blink_Right'
  | 'Brows_Up'
  | 'Head_Left'
  | 'Head_Right'
  | 'Head_Up'
  | 'Head_Down'
  | 'Nod'
  | 'Shake'

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
  source: 'model' | 'landmarks' | 'motion' | 'face' | 'manual'
  at: string
  details?: Record<string, number | string | boolean>
}

type MotionSample = {
  x: number
  y: number
  t: number
}

type HeadSample = {
  yaw: number
  pitch: number
  t: number
}

type FaceTriangle = readonly [number, number, number]

type Point2D = {
  x: number
  y: number
}

type FaceMaskLayer = {
  image: HTMLImageElement
  landmarks: NormalizedLandmark[]
  triangles: FaceTriangle[]
  width: number
  height: number
}

type FaceMaskSlot = {
  layer: FaceMaskLayer | null
  imageUrl: string | null
}

type MaskMotionState = {
  previousLandmarks: NormalizedLandmark[] | null
  displayedLandmarks: NormalizedLandmark[] | null
  previousAt: number
}

type MaskTriangleRender = {
  source: Point2D[]
  target: Point2D[]
  depth: number
}

type MetrikaParams = Record<string, string | number | boolean>

declare global {
  interface Window {
    ym?: (...args: unknown[]) => void
  }
}

type VisionFileset = Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>>

const metrikaCounterId = 110018332
const tasksVersion = '0.10.35'
const wasmPath = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${tasksVersion}/wasm`
const gestureModelPath =
  'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task'
const faceModelPath =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task'

const usedIcons = {
  ArrowDown,
  ArrowUp,
  BadgeCheck,
  Bot,
  ChevronsUp,
  CircleCheck,
  CircleDot,
  Copy,
  Crosshair,
  FlipHorizontal2,
  EyeClosed,
  Hand,
  HandFist,
  HandMetal,
  House,
  MessageCircle,
  MonitorDot,
  MoveDown,
  MoveLeft,
  MoveRight,
  MoveUp,
  MousePointer2,
  Radio,
  Redo2,
  ScanEye,
  ScanLine,
  Smile,
  Target,
  ThumbsDown,
  ThumbsUp,
  Timer,
  Undo2,
  Waypoints,
  Webcam,
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
  { key: 'OK_Gesture', title: 'OK', signal: 'Landmarks', icon: 'circle-check' },
  { key: 'Finger_Gun', title: 'Пистолет', signal: 'Landmarks', icon: 'target' },
  { key: 'Three_Fingers', title: 'Три пальца', signal: 'Landmarks', icon: 'badge-check' },
  { key: 'Four_Fingers', title: 'Четыре пальца', signal: 'Landmarks', icon: 'hand' },
  { key: 'Pinch_Hold', title: 'Щипок удержан', signal: 'Landmarks', icon: 'timer' },
  { key: 'Swipe_Left', title: 'Свайп влево', signal: 'Motion', icon: 'undo-2' },
  { key: 'Swipe_Right', title: 'Свайп вправо', signal: 'Motion', icon: 'redo-2' },
  { key: 'Swipe_Up', title: 'Свайп вверх', signal: 'Motion', icon: 'arrow-up' },
  { key: 'Swipe_Down', title: 'Свайп вниз', signal: 'Motion', icon: 'arrow-down' },
  { key: 'Palm_Up', title: 'Пальцы вверх', signal: 'Landmarks', icon: 'arrow-up' },
  { key: 'Palm_Down', title: 'Пальцы вниз', signal: 'Landmarks', icon: 'arrow-down' },
  { key: 'Zoom_In', title: 'Развести руки', signal: 'Two hands', icon: 'zoom-in' },
  { key: 'Zoom_Out', title: 'Свести руки', signal: 'Two hands', icon: 'zoom-out' },
  { key: 'Smile', title: 'Улыбка', signal: 'Face', icon: 'smile' },
  { key: 'Mouth_Open', title: 'Открыть рот', signal: 'Face', icon: 'message-circle' },
  { key: 'Blink_Left', title: 'Левый морг', signal: 'Face', icon: 'eye-closed' },
  { key: 'Blink_Right', title: 'Правый морг', signal: 'Face', icon: 'eye-closed' },
  { key: 'Brows_Up', title: 'Брови вверх', signal: 'Face', icon: 'chevrons-up' },
  { key: 'Head_Left', title: 'Голова влево', signal: 'Face pose', icon: 'move-left' },
  { key: 'Head_Right', title: 'Голова вправо', signal: 'Face pose', icon: 'move-right' },
  { key: 'Head_Up', title: 'Голова вверх', signal: 'Face pose', icon: 'move-up' },
  { key: 'Head_Down', title: 'Голова вниз', signal: 'Face pose', icon: 'move-down' },
  { key: 'Nod', title: 'Кивок', signal: 'Face motion', icon: 'circle-dot' },
  { key: 'Shake', title: 'Покачать головой', signal: 'Face motion', icon: 'scan-eye' },
]

const gestureKeys = new Set(gestureDefinitions.map((gesture) => gesture.key))
const faceTriangles = buildFaceTriangles(FaceLandmarker.FACE_LANDMARKS_TESSELATION)
const faceMaskSlotCount = 4
const extraMaskSlotNumbers = [2, 3, 4] as const
const faceOvalIndices = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176,
  149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

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
      OK_Gesture: 'Подтвердить',
      Finger_Gun: 'Точный выбор',
      Three_Fingers: 'Task view',
      Four_Fingers: 'Все окна',
      Pinch_Hold: 'Долгий клик',
      Swipe_Left: 'Назад',
      Swipe_Right: 'Вперед',
      Swipe_Up: 'Прокрутка вверх',
      Swipe_Down: 'Прокрутка вниз',
      Palm_Up: 'Показать меню',
      Palm_Down: 'Скрыть меню',
      Zoom_In: 'Увеличить',
      Zoom_Out: 'Уменьшить',
      Smile: 'Подтвердить',
      Mouth_Open: 'Голосовой ввод',
      Blink_Left: 'Назад',
      Blink_Right: 'Вперед',
      Brows_Up: 'Показать меню',
      Head_Left: 'Вкладка назад',
      Head_Right: 'Вкладка вперед',
      Head_Up: 'Прокрутка вверх',
      Head_Down: 'Прокрутка вниз',
      Nod: 'Да',
      Shake: 'Нет',
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
      OK_Gesture: 'Показать OK',
      Finger_Gun: 'Запустить эффект',
      Three_Fingers: 'Сцена 3',
      Four_Fingers: 'Сцена 4',
      Pinch_Hold: 'Удержать маркер',
      Swipe_Left: 'Предыдущая сцена',
      Swipe_Right: 'Следующая сцена',
      Swipe_Up: 'Громкость выше',
      Swipe_Down: 'Громкость ниже',
      Palm_Up: 'Поднять оверлей',
      Palm_Down: 'Опустить оверлей',
      Zoom_In: 'Увеличить камеру',
      Zoom_Out: 'Свернуть камеру',
      Smile: 'Показать реакцию',
      Mouth_Open: 'Включить голос',
      Blink_Left: 'Сцена назад',
      Blink_Right: 'Сцена вперед',
      Brows_Up: 'Показать чат',
      Head_Left: 'Предыдущий клип',
      Head_Right: 'Следующий клип',
      Head_Up: 'Громкость выше',
      Head_Down: 'Громкость ниже',
      Nod: 'Принять',
      Shake: 'Отклонить',
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
      OK_Gesture: 'Принять план',
      Finger_Gun: 'Выбрать цель',
      Three_Fingers: 'Три варианта',
      Four_Fingers: 'Все инструменты',
      Pinch_Hold: 'Глубокий анализ',
      Swipe_Left: 'Предыдущий ответ',
      Swipe_Right: 'Следующий шаг',
      Swipe_Up: 'Ответ выше',
      Swipe_Down: 'Ответ ниже',
      Palm_Up: 'Открыть контекст',
      Palm_Down: 'Свернуть контекст',
      Zoom_In: 'Расширить контекст',
      Zoom_Out: 'Сжать контекст',
      Smile: 'Одобрить',
      Mouth_Open: 'Голосовой агент',
      Blink_Left: 'Шаг назад',
      Blink_Right: 'Шаг вперед',
      Brows_Up: 'Инструменты',
      Head_Left: 'Предыдущий вариант',
      Head_Right: 'Следующий вариант',
      Head_Up: 'Развернуть',
      Head_Down: 'Свернуть',
      Nod: 'Да',
      Shake: 'Нет',
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
      OK_Gesture: 'Закрепить узел',
      Finger_Gun: 'Связать цель',
      Three_Fingers: 'Показать кластер',
      Four_Fingers: 'Развернуть уровень',
      Pinch_Hold: 'Зафиксировать связь',
      Swipe_Left: 'К прошлому узлу',
      Swipe_Right: 'К следующему узлу',
      Swipe_Up: 'Слой выше',
      Swipe_Down: 'Слой ниже',
      Palm_Up: 'Поднять узел',
      Palm_Down: 'Опустить узел',
      Zoom_In: 'Зум графа плюс',
      Zoom_Out: 'Зум графа минус',
      Smile: 'Закрепить',
      Mouth_Open: 'Чат узла',
      Blink_Left: 'Узел назад',
      Blink_Right: 'Узел вперед',
      Brows_Up: 'Меню графа',
      Head_Left: 'Панорама влево',
      Head_Right: 'Панорама вправо',
      Head_Up: 'Панорама вверх',
      Head_Down: 'Панорама вниз',
      Nod: 'Создать связь',
      Shake: 'Удалить связь',
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
      OK_Gesture: 'Сценарий OK',
      Finger_Gun: 'Выбрать устройство',
      Three_Fingers: 'Сцена 3',
      Four_Fingers: 'Сцена 4',
      Pinch_Hold: 'Диммер удержание',
      Swipe_Left: 'Предыдущая сцена',
      Swipe_Right: 'Следующая сцена',
      Swipe_Up: 'Яркость выше',
      Swipe_Down: 'Яркость ниже',
      Palm_Up: 'Шторы вверх',
      Palm_Down: 'Шторы вниз',
      Zoom_In: 'Яркость выше',
      Zoom_Out: 'Яркость ниже',
      Smile: 'Сцена уют',
      Mouth_Open: 'Голос дома',
      Blink_Left: 'Сцена назад',
      Blink_Right: 'Сцена вперед',
      Brows_Up: 'Показать панели',
      Head_Left: 'Зона влево',
      Head_Right: 'Зона вправо',
      Head_Up: 'Яркость выше',
      Head_Down: 'Яркость ниже',
      Nod: 'Включить',
      Shake: 'Выключить',
    },
  },
}

const app = document.querySelector<HTMLDivElement>('#app')
const urlParams = new URLSearchParams(window.location.search)
const isStreamlabsOutput = urlParams.get('output') === 'streamlabs' || urlParams.has('streamlabs')
const shouldAutostartCamera =
  isStreamlabsOutput ||
  urlParams.get('autostart') === '1' ||
  urlParams.has('autostart') ||
  urlParams.get('camera') === '1'

if (!app) {
  throw new Error('App root was not found')
}

document.body.classList.toggle('streamlabs-output', isStreamlabsOutput)

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
        <button class="button" id="cameraButton" type="button">
          <i data-lucide="webcam"></i>
          <span>Камера</span>
        </button>
        <button class="button" id="mirrorButton" type="button" title="Зеркальное отображение">
          <i data-lucide="flip-horizontal-2"></i>
          <span>Зеркало</span>
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
            <span id="stageEmptyText">Камера выключена</span>
          </div>
          <div class="hud" id="hud">
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
            <div>
              <span class="hud-label">Лицо</span>
              <strong id="faceCount">0</strong>
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

        <section class="gesture-strip" aria-label="Жесты и действия">
          <div class="strip-head">
            <h2>Жесты</h2>
            <p id="actionHint">Готово к маппингу команд</p>
          </div>
          <div class="gesture-grid" id="gestureGrid"></div>
        </section>
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
          <div class="status-line">
            <span class="status-dot" id="faceDot"></span>
            <div>
              <span class="label">Лицо</span>
              <strong id="faceStatus">Ожидание</strong>
            </div>
          </div>
        </section>

        <section class="panel">
          <div class="panel-head">
            <h2>Маска</h2>
            <label class="switch">
              <input id="maskToggle" type="checkbox" />
              <span></span>
            </label>
          </div>
          <div class="mask-control">
            <label class="file-button" for="maskFile">
              <i data-lucide="scan-eye"></i>
              <span>Изображение</span>
            </label>
            <input id="maskFile" class="file-input" type="file" accept="image/*" />
            <p class="mask-state" id="maskState">Не выбрана</p>
          </div>
          <div class="mask-mode" id="maskModeTabs" aria-label="Режим маски">
            <button class="mask-mode-button" data-mask-mode="mesh" type="button">Mesh</button>
            <button class="mask-mode-button" data-mask-mode="faceswap" type="button">FaceSwap</button>
          </div>
          <div class="mask-stability" id="maskStabilityBlock">
            <div class="mask-stability-head">
              <label class="input-label" for="maskStability">Стабилизация</label>
              <strong id="maskStabilityValue">25%</strong>
            </div>
            <input id="maskStability" class="range-input" type="range" min="0" max="100" step="1" value="25" />
            <div class="range-captions">
              <span>Быстрее</span>
              <span>Плавнее</span>
            </div>
          </div>
          <div class="mask-stability" id="maskEdgeFeatherBlock">
            <div class="mask-stability-head">
              <div class="setting-title">
                <label class="setting-toggle" title="Включить мягкий край">
                  <input id="maskEdgeFeatherEnabled" type="checkbox" checked />
                  <span></span>
                </label>
                <label class="input-label" for="maskEdgeFeather">Мягкий край</label>
              </div>
              <strong id="maskEdgeFeatherValue">18px</strong>
            </div>
            <input id="maskEdgeFeather" class="range-input" type="range" min="0" max="80" step="1" value="18" />
            <div class="range-captions">
              <span>Жёстче</span>
              <span>Мягче</span>
            </div>
          </div>
          <div class="mask-color" id="maskColorBlock">
            <div class="mask-stability-head">
              <div class="setting-title">
                <label class="setting-toggle" title="Включить цвет лица">
                  <input id="maskColorStrengthEnabled" type="checkbox" checked />
                  <span></span>
                </label>
                <label class="input-label" for="maskColorStrength">Цвет лица</label>
              </div>
              <strong id="maskColorStrengthValue">0%</strong>
            </div>
            <div class="mask-color-row">
              <input id="maskSkinColor" class="color-input" type="color" value="#f2c7ad" />
              <button class="mask-mode-button mask-sample-button" id="maskSampleSkinButton" type="button">С лица</button>
            </div>
            <input id="maskColorStrength" class="range-input" type="range" min="0" max="100" step="1" value="0" />
            <div class="range-captions">
              <span>Оригинал</span>
              <span>Цвет лица</span>
            </div>
            <div class="mask-adjust-grid">
              <div>
                <div class="mask-stability-head">
                  <div class="setting-title">
                    <label class="setting-toggle" title="Включить яркость">
                      <input id="maskBrightnessEnabled" type="checkbox" checked />
                      <span></span>
                    </label>
                    <label class="input-label" for="maskBrightness">Яркость</label>
                  </div>
                  <strong id="maskBrightnessValue">0</strong>
                </div>
                <input id="maskBrightness" class="range-input" type="range" min="-40" max="40" step="1" value="0" />
              </div>
              <div>
                <div class="mask-stability-head">
                  <div class="setting-title">
                    <label class="setting-toggle" title="Включить насыщенность">
                      <input id="maskSaturationEnabled" type="checkbox" checked />
                      <span></span>
                    </label>
                    <label class="input-label" for="maskSaturation">Насыщенность</label>
                  </div>
                  <strong id="maskSaturationValue">0</strong>
                </div>
                <input id="maskSaturation" class="range-input" type="range" min="-50" max="60" step="1" value="0" />
              </div>
              <div>
                <div class="mask-stability-head">
                  <div class="setting-title">
                    <label class="setting-toggle" title="Включить контраст">
                      <input id="maskContrastEnabled" type="checkbox" checked />
                      <span></span>
                    </label>
                    <label class="input-label" for="maskContrast">Контраст</label>
                  </div>
                  <strong id="maskContrastValue">0</strong>
                </div>
                <input id="maskContrast" class="range-input" type="range" min="-40" max="60" step="1" value="0" />
              </div>
              <div>
                <div class="mask-stability-head">
                  <div class="setting-title">
                    <label class="setting-toggle" title="Включить температуру">
                      <input id="maskTemperatureEnabled" type="checkbox" checked />
                      <span></span>
                    </label>
                    <label class="input-label" for="maskTemperature">Температура</label>
                  </div>
                  <strong id="maskTemperatureValue">0</strong>
                </div>
                <input id="maskTemperature" class="range-input" type="range" min="-50" max="50" step="1" value="0" />
                <div class="range-captions">
                  <span>Холоднее</span>
                  <span>Теплее</span>
                </div>
              </div>
              <div>
                <div class="mask-stability-head">
                  <div class="setting-title">
                    <label class="setting-toggle" title="Включить оттенок">
                      <input id="maskTintEnabled" type="checkbox" checked />
                      <span></span>
                    </label>
                    <label class="input-label" for="maskTint">Оттенок</label>
                  </div>
                  <strong id="maskTintValue">0</strong>
                </div>
                <input id="maskTint" class="range-input" type="range" min="-50" max="50" step="1" value="0" />
                <div class="range-captions">
                  <span>Зеленее</span>
                  <span>Розовее</span>
                </div>
              </div>
            </div>
          </div>
          <div class="mask-slots" id="maskSlotsBlock">
            <div class="mask-stability-head">
              <span class="input-label">Маски лиц</span>
              <strong>по близости</strong>
            </div>
            <div class="mask-slot-list">
              <div class="mask-slot">
                <label class="file-button" for="maskSlotFile2">
                  <i data-lucide="scan-eye"></i>
                  <span>Лицо 2</span>
                </label>
                <input id="maskSlotFile2" class="file-input" type="file" accept="image/*" />
                <span id="maskSlotState2">Основная</span>
              </div>
              <div class="mask-slot">
                <label class="file-button" for="maskSlotFile3">
                  <i data-lucide="scan-eye"></i>
                  <span>Лицо 3</span>
                </label>
                <input id="maskSlotFile3" class="file-input" type="file" accept="image/*" />
                <span id="maskSlotState3">Основная</span>
              </div>
              <div class="mask-slot">
                <label class="file-button" for="maskSlotFile4">
                  <i data-lucide="scan-eye"></i>
                  <span>Лицо 4</span>
                </label>
                <input id="maskSlotFile4" class="file-input" type="file" accept="image/*" />
                <span id="maskSlotState4">Основная</span>
              </div>
            </div>
          </div>
          <div class="faceswap-options" id="faceSwapOptions">
            <label class="input-label" for="faceSwapEndpoint">FaceSwap bridge</label>
            <input id="faceSwapEndpoint" class="text-input" type="url" value="http://127.0.0.1:8790/swap" />
            <p class="panel-state" id="faceSwapState">Ожидает локальный bridge</p>
          </div>
          <img class="mask-preview" id="maskPreview" alt="" />
        </section>

        <section class="panel">
          <div class="panel-head">
            <h2>Экран</h2>
            <label class="switch">
              <input id="hudToggle" type="checkbox" checked />
              <span></span>
            </label>
          </div>
          <p class="panel-state" id="hudState">HUD включен</p>
        </section>

        <section class="panel">
          <div class="panel-head">
            <h2>Руки</h2>
            <label class="switch">
              <input id="handTrackingToggle" type="checkbox" checked />
              <span></span>
            </label>
          </div>
          <p class="panel-state" id="handTrackingState">Трекинг включен</p>
          <div class="option-row">
            <span>Линии рук</span>
            <label class="switch">
              <input id="handMarkersToggle" type="checkbox" checked />
              <span></span>
            </label>
          </div>
          <p class="panel-state mode-state" id="handMarkersState">Линии и точки включены</p>
        </section>

        <section class="panel">
          <div class="panel-head">
            <h2>Лицо</h2>
            <label class="switch">
              <input id="faceTrackingToggle" type="checkbox" checked />
              <span></span>
            </label>
          </div>
          <p class="panel-state" id="faceTrackingState">Трекинг и маркеры включены</p>
          <div class="option-row">
            <span>Несколько лиц</span>
            <label class="switch">
              <input id="multiFaceToggle" type="checkbox" />
              <span></span>
            </label>
          </div>
          <p class="panel-state mode-state" id="multiFaceState">1 лицо</p>
        </section>

        <section class="panel">
          <div class="panel-head">
            <h2>Производительность</h2>
            <span id="performanceTitle">Авто</span>
          </div>
          <div class="mask-mode" id="performanceModeTabs" aria-label="Режим производительности">
            <button class="mask-mode-button" data-performance-mode="performance" type="button">Быстрее</button>
            <button class="mask-mode-button" data-performance-mode="quality" type="button">Качество</button>
          </div>
          <p class="panel-state mode-state" id="performanceModeState">640x480</p>
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
const hud = getElement<HTMLDivElement>('hud')
const stageEmpty = getElement<HTMLDivElement>('stageEmpty')
const stageEmptyText = getElement<HTMLSpanElement>('stageEmptyText')
const cameraButton = getElement<HTMLButtonElement>('cameraButton')
const mirrorButton = getElement<HTMLButtonElement>('mirrorButton')
const modelDot = getElement<HTMLSpanElement>('modelDot')
const modelStatus = getElement<HTMLElement>('modelStatus')
const cameraDot = getElement<HTMLSpanElement>('cameraDot')
const cameraStatus = getElement<HTMLElement>('cameraStatus')
const faceDot = getElement<HTMLSpanElement>('faceDot')
const faceStatus = getElement<HTMLElement>('faceStatus')
const handTrackingToggle = getElement<HTMLInputElement>('handTrackingToggle')
const handTrackingState = getElement<HTMLElement>('handTrackingState')
const handMarkersToggle = getElement<HTMLInputElement>('handMarkersToggle')
const handMarkersState = getElement<HTMLElement>('handMarkersState')
const faceTrackingToggle = getElement<HTMLInputElement>('faceTrackingToggle')
const faceTrackingState = getElement<HTMLElement>('faceTrackingState')
const multiFaceToggle = getElement<HTMLInputElement>('multiFaceToggle')
const multiFaceState = getElement<HTMLElement>('multiFaceState')
const hudToggle = getElement<HTMLInputElement>('hudToggle')
const hudState = getElement<HTMLElement>('hudState')
const performanceTitle = getElement<HTMLElement>('performanceTitle')
const performanceModeTabs = getElement<HTMLDivElement>('performanceModeTabs')
const performanceModeState = getElement<HTMLElement>('performanceModeState')
const currentGesture = getElement<HTMLElement>('currentGesture')
const fpsValue = getElement<HTMLElement>('fpsValue')
const handCount = getElement<HTMLElement>('handCount')
const faceCount = getElement<HTMLElement>('faceCount')
const confidenceValue = getElement<HTMLElement>('confidenceValue')
const confidenceMeter = getElement<HTMLSpanElement>('confidenceMeter')
const pinchValue = getElement<HTMLElement>('pinchValue')
const pinchMeter = getElement<HTMLSpanElement>('pinchMeter')
const motionValue = getElement<HTMLElement>('motionValue')
const motionMeter = getElement<HTMLSpanElement>('motionMeter')
const presetTitle = getElement<HTMLElement>('presetTitle')
const presetTabs = getElement<HTMLDivElement>('presetTabs')
const maskToggle = getElement<HTMLInputElement>('maskToggle')
const maskFile = getElement<HTMLInputElement>('maskFile')
const maskState = getElement<HTMLElement>('maskState')
const maskPreview = getElement<HTMLImageElement>('maskPreview')
const maskSlotsBlock = getElement<HTMLDivElement>('maskSlotsBlock')
const maskSlotFiles = extraMaskSlotNumbers.map((slotNumber) =>
  getElement<HTMLInputElement>(`maskSlotFile${slotNumber}`),
)
const maskSlotStates = extraMaskSlotNumbers.map((slotNumber) =>
  getElement<HTMLElement>(`maskSlotState${slotNumber}`),
)
const maskStabilitySlider = getElement<HTMLInputElement>('maskStability')
const maskStabilityValue = getElement<HTMLElement>('maskStabilityValue')
const maskStabilityBlock = getElement<HTMLDivElement>('maskStabilityBlock')
const maskEdgeFeatherSlider = getElement<HTMLInputElement>('maskEdgeFeather')
const maskEdgeFeatherValue = getElement<HTMLElement>('maskEdgeFeatherValue')
const maskEdgeFeatherEnabledToggle = getElement<HTMLInputElement>('maskEdgeFeatherEnabled')
const maskEdgeFeatherBlock = getElement<HTMLDivElement>('maskEdgeFeatherBlock')
const maskColorBlock = getElement<HTMLDivElement>('maskColorBlock')
const maskSkinColorInput = getElement<HTMLInputElement>('maskSkinColor')
const maskSampleSkinButton = getElement<HTMLButtonElement>('maskSampleSkinButton')
const maskColorStrengthSlider = getElement<HTMLInputElement>('maskColorStrength')
const maskColorStrengthValue = getElement<HTMLElement>('maskColorStrengthValue')
const maskColorStrengthEnabledToggle = getElement<HTMLInputElement>('maskColorStrengthEnabled')
const maskBrightnessSlider = getElement<HTMLInputElement>('maskBrightness')
const maskBrightnessValue = getElement<HTMLElement>('maskBrightnessValue')
const maskBrightnessEnabledToggle = getElement<HTMLInputElement>('maskBrightnessEnabled')
const maskSaturationSlider = getElement<HTMLInputElement>('maskSaturation')
const maskSaturationValue = getElement<HTMLElement>('maskSaturationValue')
const maskSaturationEnabledToggle = getElement<HTMLInputElement>('maskSaturationEnabled')
const maskContrastSlider = getElement<HTMLInputElement>('maskContrast')
const maskContrastValue = getElement<HTMLElement>('maskContrastValue')
const maskContrastEnabledToggle = getElement<HTMLInputElement>('maskContrastEnabled')
const maskTemperatureSlider = getElement<HTMLInputElement>('maskTemperature')
const maskTemperatureValue = getElement<HTMLElement>('maskTemperatureValue')
const maskTemperatureEnabledToggle = getElement<HTMLInputElement>('maskTemperatureEnabled')
const maskTintSlider = getElement<HTMLInputElement>('maskTint')
const maskTintValue = getElement<HTMLElement>('maskTintValue')
const maskTintEnabledToggle = getElement<HTMLInputElement>('maskTintEnabled')
const maskModeTabs = getElement<HTMLDivElement>('maskModeTabs')
const faceSwapOptions = getElement<HTMLDivElement>('faceSwapOptions')
const faceSwapEndpoint = getElement<HTMLInputElement>('faceSwapEndpoint')
const faceSwapState = getElement<HTMLElement>('faceSwapState')
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
const maskRenderCanvas = document.createElement('canvas')
const maskRenderContext = getCanvasContext(maskRenderCanvas)
const maskAlphaCanvas = document.createElement('canvas')
const maskAlphaContext = getCanvasContext(maskAlphaCanvas)
const maskFeatherCanvas = document.createElement('canvas')
const maskFeatherContext = getCanvasContext(maskFeatherCanvas)
const faceSwapCaptureCanvas = document.createElement('canvas')
const faceSwapCaptureContext = getCanvasContext(faceSwapCaptureCanvas)
const faceSwapRequestIntervalMs = 180

let recognizer: GestureRecognizer | null = null
let faceLandmarker: FaceLandmarker | null = null
let maskFaceLandmarker: FaceLandmarker | null = null
let visionFileset: VisionFileset | null = null
let modelBootPromise: Promise<void> | null = null
let stream: MediaStream | null = null
let isRunning = false
let lastVideoTime = -1
let currentPreset: PresetId = readPreset()
let mirrorMode = localStorage.getItem('xedoc-hands-mirror') !== 'off'
let handTrackingEnabled = localStorage.getItem('xedoc-hands-hand-tracking') !== 'off'
let handMarkersEnabled = localStorage.getItem('xedoc-hands-hand-markers') !== 'off'
let faceTrackingEnabled = localStorage.getItem('xedoc-hands-face-tracking') !== 'off'
let multiFaceTrackingEnabled = localStorage.getItem('xedoc-hands-multi-face') === 'on'
let hudVisible = localStorage.getItem('xedoc-hands-hud') !== 'off'
let performanceMode: PerformanceMode = readPerformanceMode()
let maskEnabled = localStorage.getItem('xedoc-hands-mask-enabled') === 'true'
let maskMode: MaskMode = readMaskMode()
let maskStability = readMaskStability()
let maskEdgeFeather = readMaskEdgeFeather()
let maskEdgeFeatherEnabled = readMaskSettingEnabled('edge-feather')
let maskSkinColor = readMaskSkinColor()
let maskColorStrength = readMaskColorStrength()
let maskColorStrengthEnabled = readMaskSettingEnabled('color-strength')
let maskBrightness = readMaskBrightness()
let maskBrightnessEnabled = readMaskSettingEnabled('brightness')
let maskSaturation = readMaskSaturation()
let maskSaturationEnabled = readMaskSettingEnabled('saturation')
let maskContrast = readMaskContrast()
let maskContrastEnabled = readMaskSettingEnabled('contrast')
let maskTemperature = readMaskTemperature()
let maskTemperatureEnabled = readMaskSettingEnabled('temperature')
let maskTint = readMaskTint()
let maskTintEnabled = readMaskSettingEnabled('tint')
let maskLayer: FaceMaskLayer | null = null
let maskImageUrl: string | null = null
let maskSourceBlob: Blob | null = null
const extraMaskSlots: FaceMaskSlot[] = Array.from({ length: faceMaskSlotCount - 1 }, () => ({
  layer: null,
  imageUrl: null,
}))
let maskMotionStates = createMaskMotionStates()
let faceSwapFrameImage: HTMLImageElement | null = null
let faceSwapFrameUrl: string | null = null
let faceSwapInFlight = false
let faceSwapLastRequestAt = 0
let faceSwapBridgeStatus: 'idle' | 'ok' | 'error' = 'idle'
let lastEvent: GestureEvent | null = null
let eventSequence = 0
let lastTrackedFaceCount = -1
let maxTrackedFaceCount = 0
let frameCount = 0
let fpsStartedAt = performance.now()
let pinchDown = false
let pinchStartedAt: number | null = null
let pinchHoldDown = false
let motionSamples: MotionSample[] = []
let zoomSamples: MotionSample[] = []
let headSamples: HeadSample[] = []
const eventHistory: GestureEvent[] = []
const cooldowns = new Map<GestureKey, number>()

renderPresetTabs()
renderGestureGrid()
renderMaskModeTabs()
renderPerformanceModeTabs()
refreshIcons()
setMirrorMode(mirrorMode)
setHandTrackingEnabled(handTrackingEnabled)
setHandMarkersEnabled(handMarkersEnabled)
setFaceTrackingEnabled(faceTrackingEnabled)
setMultiFaceTrackingEnabled(multiFaceTrackingEnabled, false)
setHudVisible(hudVisible)
setPerformanceMode(performanceMode)
setMaskMode(maskMode)
setMaskStability(maskStability)
setMaskEdgeFeather(maskEdgeFeather)
setMaskEdgeFeatherEnabled(maskEdgeFeatherEnabled)
setMaskSkinColor(maskSkinColor)
setMaskColorStrength(maskColorStrength)
setMaskColorStrengthEnabled(maskColorStrengthEnabled)
setMaskBrightness(maskBrightness)
setMaskBrightnessEnabled(maskBrightnessEnabled)
setMaskSaturation(maskSaturation)
setMaskSaturationEnabled(maskSaturationEnabled)
setMaskContrast(maskContrast)
setMaskContrastEnabled(maskContrastEnabled)
setMaskTemperature(maskTemperature)
setMaskTemperatureEnabled(maskTemperatureEnabled)
setMaskTint(maskTint)
setMaskTintEnabled(maskTintEnabled)
setMaskEnabled(maskEnabled)
updateMaskState()
setModelState('loading', 'Загрузка')
setCameraState('idle', 'Ожидание')
setFaceState('idle', faceTrackingEnabled ? 'Ожидание' : 'Откл.')
const bootPromise = bootModel()
void bootPromise

if (shouldAutostartCamera) {
  void bootPromise.then(() => startCamera())
}

cameraButton.addEventListener('click', () => {
  if (isRunning) {
    stopCamera()
    return
  }

  void startCamera()
})

mirrorButton.addEventListener('click', () => {
  setMirrorMode(!mirrorMode)
  trackToggle('mirror_toggled', mirrorMode)
})

maskToggle.addEventListener('change', () => {
  setMaskEnabled(maskToggle.checked)
  trackToggle('mask_toggled', maskToggle.checked, {
    maskMode,
    maskCount: getLoadedMaskCount(),
  })
})

handTrackingToggle.addEventListener('change', () => {
  setHandTrackingEnabled(handTrackingToggle.checked)
  trackToggle('hand_tracking_toggled', handTrackingToggle.checked)
})

handMarkersToggle.addEventListener('change', () => {
  setHandMarkersEnabled(handMarkersToggle.checked)
  trackToggle('hand_markers_toggled', handMarkersToggle.checked)
})

faceTrackingToggle.addEventListener('change', () => {
  setFaceTrackingEnabled(faceTrackingToggle.checked)
  trackToggle('face_tracking_toggled', faceTrackingToggle.checked)
})

multiFaceToggle.addEventListener('change', () => {
  setMultiFaceTrackingEnabled(multiFaceToggle.checked)
  trackToggle('multi_face_toggled', multiFaceToggle.checked, {
    faceLimit: getFaceCountLimit(),
  })
})

hudToggle.addEventListener('change', () => {
  setHudVisible(hudToggle.checked)
  trackToggle('hud_toggled', hudToggle.checked)
})

performanceModeTabs.addEventListener('click', (event) => {
  const button = (event.target as HTMLElement).closest<HTMLButtonElement>('.mask-mode-button')
  const mode = button?.dataset.performanceMode

  if (mode !== 'performance' && mode !== 'quality') {
    return
  }

  const shouldRestartCamera = isRunning && mode !== performanceMode
  const changed = mode !== performanceMode
  setPerformanceMode(mode)

  if (changed) {
    trackMetrika('performance_mode_changed', {
      mode,
      camera: mode === 'performance' ? '640x480' : '1280x720',
    })
  }

  if (shouldRestartCamera) {
    stopCamera()
    void startCamera()
  }
})

maskModeTabs.addEventListener('click', (event) => {
  const button = (event.target as HTMLElement).closest<HTMLButtonElement>('.mask-mode-button')
  const mode = button?.dataset.maskMode

  if (mode === 'mesh' || mode === 'faceswap') {
    const changed = mode !== maskMode
    setMaskMode(mode)

    if (changed) {
      trackMetrika('mask_mode_changed', {
        mode,
        maskCount: getLoadedMaskCount(),
      })
    }
  }
})

maskFile.addEventListener('change', () => {
  const file = maskFile.files?.[0]

  if (file) {
    loadMaskFile(file)
  }
})

maskSlotFiles.forEach((input, index) => {
  input.addEventListener('change', () => {
    const file = input.files?.[0]

    if (file) {
      loadMaskFile(file, index + 1)
    }

    input.value = ''
  })
})

maskStabilitySlider.addEventListener('input', () => {
  setMaskStability(Number(maskStabilitySlider.value))
})

maskStabilitySlider.addEventListener('change', () => {
  trackMaskSetting('stability', maskStability)
})

maskEdgeFeatherSlider.addEventListener('input', () => {
  setMaskEdgeFeather(Number(maskEdgeFeatherSlider.value))
})

maskEdgeFeatherSlider.addEventListener('change', () => {
  trackMaskSetting('edgeFeather', maskEdgeFeather)
})

maskEdgeFeatherEnabledToggle.addEventListener('change', () => {
  setMaskEdgeFeatherEnabled(maskEdgeFeatherEnabledToggle.checked)
  trackMaskSettingToggle('edgeFeather', maskEdgeFeatherEnabled)
})

maskSkinColorInput.addEventListener('input', () => {
  setMaskSkinColor(maskSkinColorInput.value)
})

maskSkinColorInput.addEventListener('change', () => {
  trackMaskSetting('skinColor', maskSkinColor)
})

maskColorStrengthSlider.addEventListener('input', () => {
  setMaskColorStrength(Number(maskColorStrengthSlider.value))
})

maskColorStrengthSlider.addEventListener('change', () => {
  trackMaskSetting('skinColorStrength', maskColorStrength)
})

maskColorStrengthEnabledToggle.addEventListener('change', () => {
  setMaskColorStrengthEnabled(maskColorStrengthEnabledToggle.checked)
  trackMaskSettingToggle('skinColorStrength', maskColorStrengthEnabled)
})

maskBrightnessSlider.addEventListener('input', () => {
  setMaskBrightness(Number(maskBrightnessSlider.value))
})

maskBrightnessSlider.addEventListener('change', () => {
  trackMaskSetting('brightness', maskBrightness)
})

maskBrightnessEnabledToggle.addEventListener('change', () => {
  setMaskBrightnessEnabled(maskBrightnessEnabledToggle.checked)
  trackMaskSettingToggle('brightness', maskBrightnessEnabled)
})

maskSaturationSlider.addEventListener('input', () => {
  setMaskSaturation(Number(maskSaturationSlider.value))
})

maskSaturationSlider.addEventListener('change', () => {
  trackMaskSetting('saturation', maskSaturation)
})

maskSaturationEnabledToggle.addEventListener('change', () => {
  setMaskSaturationEnabled(maskSaturationEnabledToggle.checked)
  trackMaskSettingToggle('saturation', maskSaturationEnabled)
})

maskContrastSlider.addEventListener('input', () => {
  setMaskContrast(Number(maskContrastSlider.value))
})

maskContrastSlider.addEventListener('change', () => {
  trackMaskSetting('contrast', maskContrast)
})

maskContrastEnabledToggle.addEventListener('change', () => {
  setMaskContrastEnabled(maskContrastEnabledToggle.checked)
  trackMaskSettingToggle('contrast', maskContrastEnabled)
})

maskTemperatureSlider.addEventListener('input', () => {
  setMaskTemperature(Number(maskTemperatureSlider.value))
})

maskTemperatureSlider.addEventListener('change', () => {
  trackMaskSetting('temperature', maskTemperature)
})

maskTemperatureEnabledToggle.addEventListener('change', () => {
  setMaskTemperatureEnabled(maskTemperatureEnabledToggle.checked)
  trackMaskSettingToggle('temperature', maskTemperatureEnabled)
})

maskTintSlider.addEventListener('input', () => {
  setMaskTint(Number(maskTintSlider.value))
})

maskTintSlider.addEventListener('change', () => {
  trackMaskSetting('tint', maskTint)
})

maskTintEnabledToggle.addEventListener('change', () => {
  setMaskTintEnabled(maskTintEnabledToggle.checked)
  trackMaskSettingToggle('tint', maskTintEnabled)
})

maskSampleSkinButton.addEventListener('click', () => {
  trackMetrika('mask_skin_sample_clicked', {
    maskMode,
  })
  sampleMaskSkinColor()
})

faceSwapEndpoint.addEventListener('change', () => {
  localStorage.setItem('xedoc-hands-faceswap-endpoint', faceSwapEndpoint.value.trim())
  faceSwapState.textContent = 'Endpoint обновлен'
  faceSwapBridgeStatus = 'idle'
  trackMetrika('faceswap_endpoint_changed')
})

copyButton.addEventListener('click', () => {
  void copyLastEvent()
})

webhookToggle.addEventListener('change', () => {
  localStorage.setItem('xedoc-hands-webhook-enabled', String(webhookToggle.checked))
  webhookState.textContent = webhookToggle.checked ? 'Webhook включен' : 'Локальный лог активен'
  trackToggle('webhook_toggled', webhookToggle.checked)
})

webhookUrl.addEventListener('change', () => {
  localStorage.setItem('xedoc-hands-webhook-url', webhookUrl.value)
  trackMetrika('webhook_endpoint_changed')
})

window.addEventListener('resize', syncCanvasSize)
window.addEventListener('beforeunload', () => {
  if (maskImageUrl) {
    URL.revokeObjectURL(maskImageUrl)
  }

  extraMaskSlots.forEach(revokeMaskSlot)
  resetFaceSwapFrame()
})

const savedWebhook = localStorage.getItem('xedoc-hands-webhook-url')
const savedWebhookEnabled = localStorage.getItem('xedoc-hands-webhook-enabled') === 'true'
const savedFaceSwapEndpoint = localStorage.getItem('xedoc-hands-faceswap-endpoint')

if (savedWebhook) {
  webhookUrl.value = savedWebhook
}

if (savedFaceSwapEndpoint) {
  faceSwapEndpoint.value = savedFaceSwapEndpoint
}

webhookToggle.checked = savedWebhookEnabled
webhookState.textContent = savedWebhookEnabled ? 'Webhook включен' : 'Локальный лог активен'

trackMetrika('app_loaded', {
  preset: currentPreset,
  performanceMode,
  handTracking: handTrackingEnabled,
  faceTracking: faceTrackingEnabled,
  multiFace: multiFaceTrackingEnabled,
  hudVisible,
  maskMode,
  maskEnabled,
  streamlabsOutput: isStreamlabsOutput,
  autostart: shouldAutostartCamera,
})

async function bootModel() {
  if (modelBootPromise) {
    await modelBootPromise
    return
  }

  modelBootPromise = bootModels()
  await modelBootPromise
}

async function bootModels() {
  try {
    const vision = await getVisionFileset()

    const [handTask, faceTask, maskFaceTask] = await Promise.all([
      GestureRecognizer.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: gestureModelPath,
          delegate: 'GPU',
        },
        runningMode: 'VIDEO',
        numHands: 2,
        minHandDetectionConfidence: 0.55,
        minHandPresenceConfidence: 0.55,
        minTrackingConfidence: 0.55,
      }),
      createVideoFaceLandmarker(vision),
      createMaskFaceLandmarker(vision),
    ])

    recognizer = handTask
    faceLandmarker = faceTask
    maskFaceLandmarker = maskFaceTask

    setModelState('ready', 'Готовы')
    setFaceState('idle', faceTrackingEnabled ? 'Нет лица' : 'Откл.')
  } catch (error) {
    console.error(error)
    modelBootPromise = null
    setModelState('error', 'Ошибка')
    setFaceState('error', 'Ошибка')
  }
}

async function getVisionFileset() {
  if (!visionFileset) {
    visionFileset = await FilesetResolver.forVisionTasks(wasmPath)
  }

  return visionFileset
}

function getFaceCountLimit() {
  return multiFaceTrackingEnabled ? 4 : 1
}

async function createVideoFaceLandmarker(vision: VisionFileset) {
  return FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: faceModelPath,
      delegate: 'GPU',
    },
    runningMode: 'VIDEO',
    numFaces: getFaceCountLimit(),
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: true,
    minFaceDetectionConfidence: 0.55,
    minFacePresenceConfidence: 0.55,
    minTrackingConfidence: 0.55,
  })
}

async function createMaskFaceLandmarker(vision: VisionFileset) {
  return FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: faceModelPath,
      delegate: 'GPU',
    },
    runningMode: 'IMAGE',
    numFaces: 1,
    minFaceDetectionConfidence: 0.55,
    minFacePresenceConfidence: 0.55,
    minTrackingConfidence: 0.55,
  })
}

async function rebuildFaceLandmarker() {
  try {
    if (modelBootPromise) {
      await modelBootPromise
    }

    const vision = await getVisionFileset()
    setFaceState('loading', 'Перекл.')
    const nextFaceLandmarker = await createVideoFaceLandmarker(vision)
    faceLandmarker?.close()
    faceLandmarker = nextFaceLandmarker
    headSamples = []
    resetMaskMotion()
    setFaceState('idle', faceTrackingEnabled ? (isRunning ? 'Нет лица' : 'Ожидание') : 'Откл.')
  } catch (error) {
    console.error(error)
    setFaceState('error', 'Ошибка')
  }
}

async function startCamera() {
  if (isRunning) {
    return
  }

  stageEmptyText.textContent = 'Запускаем камеру'
  stageEmpty.classList.remove('is-hidden')

  if (!recognizer || !faceLandmarker) {
    setCameraState('loading', 'Ждем модель')
    stageEmptyText.textContent = 'Загружаем модели'
    await bootModel()
  }

  if (!recognizer || !faceLandmarker) {
    setCameraState('error', 'Модель недоступна')
    stageEmptyText.textContent = 'Модель недоступна'
    stageEmpty.classList.remove('is-hidden')
    return
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    setCameraState('error', 'Нет getUserMedia')
    stageEmptyText.textContent = 'Камера недоступна в этом браузере'
    stageEmpty.classList.remove('is-hidden')
    trackMetrika('camera_error', {
      reason: 'getUserMedia_unavailable',
    })
    return
  }

  try {
    stageEmptyText.textContent = 'Запрашиваем доступ к камере'
    stream = await navigator.mediaDevices.getUserMedia(getCameraConstraints())

    video.srcObject = stream
    await video.play()
    isRunning = true
    cameraButton.classList.add('is-active')
    cameraButton.querySelector('span')!.textContent = 'Стоп'
    stageEmpty.classList.add('is-hidden')
    setCameraState('ready', 'Работает')
    syncCanvasSize()
    requestAnimationFrame(predictFrame)
    trackMetrika('camera_started', {
      performanceMode,
      handTracking: handTrackingEnabled,
      faceTracking: faceTrackingEnabled,
      multiFace: multiFaceTrackingEnabled,
      streamlabsOutput: isStreamlabsOutput,
      autostart: shouldAutostartCamera,
    })
  } catch (error) {
    console.error(error)
    setCameraState('error', 'Нет доступа')
    stageEmptyText.textContent = getCameraErrorMessage(error)
    stageEmpty.classList.remove('is-hidden')
    trackMetrika('camera_error', {
      reason: error instanceof Error ? error.name : 'unknown',
      streamlabsOutput: isStreamlabsOutput,
      autostart: shouldAutostartCamera,
    })
  }
}

function stopCamera() {
  isRunning = false
  lastVideoTime = -1
  motionSamples = []
  zoomSamples = []
  headSamples = []
  pinchDown = false
  pinchStartedAt = null
  pinchHoldDown = false
  stream?.getTracks().forEach((track) => track.stop())
  stream = null
  video.srcObject = null
  context.clearRect(0, 0, canvas.width, canvas.height)
  cursorDot.classList.remove('is-visible')
  stageEmpty.classList.remove('is-hidden')
  stageEmptyText.textContent = 'Камера выключена'
  cameraButton.classList.remove('is-active')
  cameraButton.querySelector('span')!.textContent = 'Камера'
  setCameraState('idle', 'Ожидание')
  setFaceState('idle', faceTrackingEnabled ? 'Ожидание' : 'Откл.')
  resetMaskMotion()
  resetFaceSwapFrame()
  faceCount.textContent = '0'
  setReadout(handTrackingEnabled ? 'Нет руки' : 'Руки выкл.', 0, 0, 0, 0, 0)
  setActiveGestures(new Set())
  trackMetrika('camera_stopped')
}

function predictFrame(now: number) {
  if (!isRunning || !recognizer || !faceLandmarker) {
    return
  }

  syncCanvasSize()

  if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime
    const handResult = handTrackingEnabled ? recognizer.recognizeForVideo(video, now) : null
    const faceResult = faceTrackingEnabled ? faceLandmarker.detectForVideo(video, now) : null
    processResult(handResult, faceResult, now)
    updateFps(now)
  }

  requestAnimationFrame(predictFrame)
}

function processResult(result: GestureRecognizerResult | null, faceResult: FaceLandmarkerResult | null, now: number) {
  context.clearRect(0, 0, canvas.width, canvas.height)
  updateFaceSwapBridge(now)
  drawFaceSwapFrame()

  const detected = new Set<GestureKey>()
  const handLandmarks = handTrackingEnabled ? (result?.landmarks ?? []) : []
  const topGesture = handTrackingEnabled ? result?.gestures[0]?.[0] : undefined
  const topScore = topGesture?.score ?? 0
  const modelGesture = topGesture?.categoryName ?? 'None'
  const landmarks = handLandmarks[0]

  if (faceTrackingEnabled && faceResult) {
    processFaceResult(faceResult, now, detected)
  } else {
    faceCount.textContent = '0'
    trackFaceCount(0)
    headSamples = []
    resetMaskMotion()
    setFaceState('idle', faceTrackingEnabled ? 'Нет лица' : 'Откл.')
  }

  if (handMarkersEnabled) {
    for (const hand of handLandmarks) {
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
  }

  if (handTrackingEnabled && isGestureKey(modelGesture) && topScore > 0.58) {
    detected.add(modelGesture)
    fireGesture(modelGesture, topScore, 'model')
  }

  let pinchStrength = 0
  let motionStrength = 0

  if (handTrackingEnabled && landmarks) {
    const pinch = updatePinch(landmarks, now)
    pinchStrength = pinch.strength

    if (pinch.active) {
      detected.add('Pinch')
    }

    if (pinch.holdActive) {
      detected.add('Pinch_Hold')
    }

    updateShapeGestures(landmarks, pinch.ratio, detected)
    motionStrength = updateMotion(landmarks, now)
    updateCursor(landmarks)
  } else {
    resetHandTracking()
  }

  updateZoom(handLandmarks, now, detected)
  setActiveGestures(detected)

  const gestureTitle =
    detected.size > 0
      ? [...detected].map((gesture) => gestureTitleFor(gesture)).join(' + ')
      : handTrackingEnabled
        ? 'Нет руки'
        : 'Руки выкл.'

  setReadout(gestureTitle, topScore, handLandmarks.length, pinchStrength, motionStrength, detected.size)
}

function updatePinch(landmarks: NormalizedLandmark[], now: number) {
  const thumbTip = landmarks[4]
  const indexTip = landmarks[8]
  const indexMcp = landmarks[5]
  const pinkyMcp = landmarks[17]
  const palmWidth = Math.max(distance(indexMcp, pinkyMcp), 0.001)
  const ratio = distance(thumbTip, indexTip) / palmWidth
  const strength = clamp(1 - (ratio - 0.25) / 0.55, 0, 1)
  const active = ratio < 0.5
  let holdActive = false

  if (active) {
    pinchStartedAt ??= now
    holdActive = now - pinchStartedAt > 700

    if (holdActive && !pinchHoldDown) {
      pinchHoldDown = true
      fireGesture('Pinch_Hold', strength, 'landmarks', {
        holdMs: Math.round(now - pinchStartedAt),
        ratio: round(ratio),
        strength: round(strength),
      })
    }
  } else {
    pinchStartedAt = null
    pinchHoldDown = false
  }

  if (ratio < 0.42 && !pinchDown) {
    pinchDown = true
    fireGesture('Pinch', strength, 'landmarks', { ratio: round(ratio), strength: round(strength) })
  }

  if (ratio > 0.62) {
    pinchDown = false
  }

  return { active, holdActive, ratio, strength }
}

function updateShapeGestures(
  landmarks: NormalizedLandmark[],
  pinchRatio: number,
  detected: Set<GestureKey>,
) {
  const hand = analyzeHand(landmarks)
  const pinchConfidence = clamp(1 - pinchRatio, 0, 1)
  const directionY = hand.averageLongTipY - landmarks[0].y
  const directionConfidence = clamp(Math.abs(directionY) / (hand.palmWidth * 1.8), 0, 1)

  if (pinchRatio < 0.47 && hand.middle && hand.ring && hand.pinky) {
    detected.add('OK_Gesture')
    fireGesture('OK_Gesture', pinchConfidence, 'landmarks', {
      ratio: round(pinchRatio),
      longFingers: hand.longCount,
    })
    return
  }

  if (hand.thumb && hand.index && !hand.middle && !hand.ring && !hand.pinky) {
    detected.add('Finger_Gun')
    fireGesture('Finger_Gun', 0.86, 'landmarks', { longFingers: hand.longCount })
    return
  }

  if (hand.index && hand.middle && hand.ring && !hand.pinky) {
    detected.add('Three_Fingers')
    fireGesture('Three_Fingers', 0.82, 'landmarks', { longFingers: hand.longCount })
  } else if (hand.longCount === 4 && !hand.thumb) {
    detected.add('Four_Fingers')
    fireGesture('Four_Fingers', 0.82, 'landmarks', { longFingers: hand.longCount })
  }

  if (hand.longCount >= 4 && directionY < -hand.palmWidth * 0.7) {
    detected.add('Palm_Up')
    fireGesture('Palm_Up', directionConfidence, 'landmarks', { directionY: round(directionY) })
  }

  if (hand.longCount >= 4 && directionY > hand.palmWidth * 0.7) {
    detected.add('Palm_Down')
    fireGesture('Palm_Down', directionConfidence, 'landmarks', { directionY: round(directionY) })
  }
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
  const strength = clamp(Math.max(Math.abs(dx), Math.abs(dy)) / 0.26, 0, 1)

  if (Math.abs(dx) > 0.22 && Math.abs(dx) > Math.abs(dy) * 1.45) {
    const gesture: GestureKey = dx > 0 ? 'Swipe_Right' : 'Swipe_Left'
    fireGesture(gesture, strength, 'motion', { dx: round(dx), dy: round(dy) })
    motionSamples = []
  }

  if (Math.abs(dy) > 0.2 && Math.abs(dy) > Math.abs(dx) * 1.45) {
    const gesture: GestureKey = dy < 0 ? 'Swipe_Up' : 'Swipe_Down'
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

function processFaceResult(result: FaceLandmarkerResult, now: number, detected: Set<GestureKey>) {
  const faces = result.faceLandmarks
  const rankedFaces = getRankedFaces(faces)
  const primaryFace = rankedFaces[0]
  const primaryFaceIndex = primaryFace?.index ?? -1
  const face = primaryFace?.landmarks
  faceCount.textContent = String(faces.length)
  trackFaceCount(faces.length)

  if (!face) {
    setFaceState('idle', 'Нет лица')
    headSamples = []
    resetMaskMotion()
    return
  }

  setFaceState('ready', 'В кадре')

  if (maskEnabled && maskMode === 'mesh' && hasMeshMaskLayer()) {
    drawMasksForFaces(rankedFaces, now)
  } else if (maskEnabled && maskMode === 'faceswap') {
    rememberMaskFace(face, now)
    rankedFaces.filter((ranked) => ranked.index !== primaryFaceIndex).forEach((ranked) => drawFace(ranked.landmarks))
  } else {
    rememberMaskFace(face, now)
    if (multiFaceTrackingEnabled) {
      faces.forEach(drawFace)
    } else {
      drawFace(face)
    }
  }

  const categories = result.faceBlendshapes[primaryFaceIndex]?.categories ?? []
  const smile = (faceScore(categories, 'mouthSmileLeft') + faceScore(categories, 'mouthSmileRight')) / 2
  const jawOpen = faceScore(categories, 'jawOpen')
  const blinkLeft = faceScore(categories, 'eyeBlinkLeft')
  const blinkRight = faceScore(categories, 'eyeBlinkRight')
  const browsUp = Math.max(
    faceScore(categories, 'browInnerUp'),
    (faceScore(categories, 'browOuterUpLeft') + faceScore(categories, 'browOuterUpRight')) / 2,
  )

  if (smile > 0.35) {
    fireFaceSignal('Smile', smile, detected, { smile: round(smile) })
  }

  if (jawOpen > 0.36) {
    fireFaceSignal('Mouth_Open', jawOpen, detected, { jawOpen: round(jawOpen) })
  }

  if (blinkLeft > 0.55) {
    fireFaceSignal('Blink_Left', blinkLeft, detected, { blinkLeft: round(blinkLeft) })
  }

  if (blinkRight > 0.55) {
    fireFaceSignal('Blink_Right', blinkRight, detected, { blinkRight: round(blinkRight) })
  }

  if (browsUp > 0.34) {
    fireFaceSignal('Brows_Up', browsUp, detected, { browsUp: round(browsUp) })
  }

  const pose = getHeadPose(face)

  if (!pose) {
    return
  }

  const yaw = mirrorMode ? -pose.yaw : pose.yaw
  const pitch = pose.pitch
  const yawStrength = clamp((Math.abs(yaw) - 0.17) / 0.22, 0, 1)
  const pitchStrength = clamp((Math.abs(pitch) - 0.16) / 0.22, 0, 1)

  if (yaw < -0.2) {
    fireFaceSignal('Head_Left', yawStrength, detected, { yaw: round(yaw) })
  } else if (yaw > 0.2) {
    fireFaceSignal('Head_Right', yawStrength, detected, { yaw: round(yaw) })
  }

  if (pitch < -0.19) {
    fireFaceSignal('Head_Up', pitchStrength, detected, { pitch: round(pitch) })
  } else if (pitch > 0.19) {
    fireFaceSignal('Head_Down', pitchStrength, detected, { pitch: round(pitch) })
  }

  headSamples.push({ yaw, pitch, t: now })
  headSamples = headSamples.filter((sample) => now - sample.t <= 900)

  if (headSamples.length < 5) {
    return
  }

  const yawValues = headSamples.map((sample) => sample.yaw)
  const pitchValues = headSamples.map((sample) => sample.pitch)
  const minYaw = Math.min(...yawValues)
  const maxYaw = Math.max(...yawValues)
  const minPitch = Math.min(...pitchValues)
  const maxPitch = Math.max(...pitchValues)
  const yawRange = maxYaw - minYaw
  const pitchRange = maxPitch - minPitch

  if (pitchRange > 0.32 && minPitch < -0.11 && maxPitch > 0.11) {
    fireFaceSignal('Nod', clamp((pitchRange - 0.28) / 0.3, 0, 1), detected, {
      pitchRange: round(pitchRange),
    })
    headSamples = []
  } else if (yawRange > 0.36 && minYaw < -0.13 && maxYaw > 0.13) {
    fireFaceSignal('Shake', clamp((yawRange - 0.32) / 0.34, 0, 1), detected, {
      yawRange: round(yawRange),
    })
    headSamples = []
  }
}

function drawMasksForFaces(rankedFaces: { landmarks: NormalizedLandmark[]; index: number; area: number }[], now: number) {
  const maskTargets = rankedFaces
    .slice(0, faceMaskSlotCount)
    .map((rankedFace, rank) => ({
      landmarks: rankedFace.landmarks,
      layer: getMaskLayerForFaceRank(rank),
      motion: maskMotionStates[rank],
    }))
    .filter((target): target is { landmarks: NormalizedLandmark[]; layer: FaceMaskLayer; motion: MaskMotionState } =>
      Boolean(target.layer && target.motion),
    )

  for (const target of maskTargets.reverse()) {
    drawFaceMask(compensateMaskLag(target.landmarks, now, target.motion), target.layer)
  }
}

function getRankedFaces(faces: NormalizedLandmark[][]) {
  return faces
    .map((landmarks, index) => ({
      landmarks,
      index,
      area: getFaceLandmarkArea(landmarks),
    }))
    .sort((left, right) => right.area - left.area)
}

function getMaskLayerForFaceRank(rank: number) {
  return extraMaskSlots[rank - 1]?.layer ?? maskLayer
}

function getFaceLandmarkArea(face: NormalizedLandmark[]) {
  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity

  for (const landmark of face) {
    minX = Math.min(minX, landmark.x)
    minY = Math.min(minY, landmark.y)
    maxX = Math.max(maxX, landmark.x)
    maxY = Math.max(maxY, landmark.y)
  }

  if (!Number.isFinite(minX) || !Number.isFinite(minY) || !Number.isFinite(maxX) || !Number.isFinite(maxY)) {
    return 0
  }

  return Math.max(maxX - minX, 0) * Math.max(maxY - minY, 0)
}

function drawFace(landmarks: NormalizedLandmark[]) {
  drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, {
    color: 'rgba(47, 110, 211, 0.88)',
    lineWidth: 3,
  })
  drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, {
    color: 'rgba(255, 255, 255, 0.82)',
    lineWidth: 2,
  })
  drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, {
    color: 'rgba(255, 255, 255, 0.82)',
    lineWidth: 2,
  })
  drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW, {
    color: 'rgba(74, 222, 128, 0.9)',
    lineWidth: 2,
  })
  drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW, {
    color: 'rgba(74, 222, 128, 0.9)',
    lineWidth: 2,
  })
  drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LIPS, {
    color: 'rgba(207, 122, 19, 0.9)',
    lineWidth: 2,
  })
  drawingUtils.drawLandmarks([landmarks[1], landmarks[13], landmarks[14]], {
    color: 'rgba(255, 255, 255, 0.95)',
    fillColor: 'rgba(47, 110, 211, 0.72)',
    lineWidth: 2,
    radius: 3,
  })
}

function drawFaceMask(landmarks: NormalizedLandmark[], layer: FaceMaskLayer | null = maskLayer) {
  if (!layer) {
    return
  }

  syncMaskCanvases()
  maskRenderContext.clearRect(0, 0, maskRenderCanvas.width, maskRenderCanvas.height)
  maskRenderContext.save()
  maskRenderContext.imageSmoothingEnabled = true
  maskRenderContext.imageSmoothingQuality = 'high'
  maskRenderContext.globalAlpha = 1
  const renderTriangles = getMaskRenderTriangles(layer, landmarks)

  for (const triangle of renderTriangles) {
    drawWarpedTriangle(
      maskRenderContext,
      layer.image,
      triangle.source,
      triangle.target,
      0.6,
    )
  }

  maskRenderContext.restore()
  applyMaskColorCorrection()
  applyRenderedMaskFeather(landmarks)
  applySoftFaceMask(landmarks)

  context.save()
  context.globalAlpha = 0.96
  context.drawImage(maskRenderCanvas, 0, 0)
  context.restore()
}

function getMaskRenderTriangles(layer: FaceMaskLayer, landmarks: NormalizedLandmark[]) {
  const triangles: MaskTriangleRender[] = []

  for (const triangle of layer.triangles) {
    const source = triangle.map((index) => layer.landmarks[index])
    const target = triangle.map((index) => landmarks[index])

    if (!source.every(Boolean) || !target.every(Boolean)) {
      continue
    }

    const sourcePoints = source.map((landmark) => landmarkToSourcePoint(landmark, layer))
    const targetPoints = target.map(landmarkToCanvasPoint)

    if (Math.abs(triangleArea(sourcePoints)) < 0.05 || Math.abs(triangleArea(targetPoints)) < 0.008) {
      continue
    }

    triangles.push({
      source: sourcePoints,
      target: targetPoints,
      depth: averageTriangleZ(target),
    })
  }

  return triangles.sort((left, right) => right.depth - left.depth)
}

function averageTriangleZ(landmarks: NormalizedLandmark[]) {
  return landmarks.reduce((sum, landmark) => sum + (landmark.z ?? 0), 0) / landmarks.length
}

function applyMaskColorCorrection() {
  const strength = maskColorStrengthEnabled ? maskColorStrength / 100 : 0
  const brightness = maskBrightnessEnabled ? maskBrightness : 0
  const saturation = maskSaturationEnabled ? maskSaturation : 0
  const contrast = maskContrastEnabled ? maskContrast : 0
  const temperature = maskTemperatureEnabled ? maskTemperature : 0
  const tint = maskTintEnabled ? maskTint : 0

  if (strength <= 0 && brightness === 0 && saturation === 0 && contrast === 0 && temperature === 0 && tint === 0) {
    return
  }

  maskFeatherContext.clearRect(0, 0, maskFeatherCanvas.width, maskFeatherCanvas.height)
  maskFeatherContext.drawImage(maskRenderCanvas, 0, 0)
  maskRenderContext.clearRect(0, 0, maskRenderCanvas.width, maskRenderCanvas.height)
  maskRenderContext.save()
  maskRenderContext.filter = `brightness(${100 + brightness}%) contrast(${100 + contrast}%) saturate(${100 + saturation}%)`
  maskRenderContext.drawImage(maskFeatherCanvas, 0, 0)
  maskRenderContext.restore()

  if (temperature !== 0) {
    applyMaskColorOverlay(temperature > 0 ? '#ffb36c' : '#6ca8ff', Math.abs(temperature) / 170)
  }

  if (tint !== 0) {
    applyMaskColorOverlay(tint > 0 ? '#ff7bbd' : '#58d06f', Math.abs(tint) / 190)
  }

  if (strength <= 0) {
    return
  }

  maskFeatherContext.clearRect(0, 0, maskFeatherCanvas.width, maskFeatherCanvas.height)
  maskFeatherContext.save()
  maskFeatherContext.fillStyle = maskSkinColor
  maskFeatherContext.fillRect(0, 0, maskFeatherCanvas.width, maskFeatherCanvas.height)
  maskFeatherContext.globalCompositeOperation = 'destination-in'
  maskFeatherContext.drawImage(maskRenderCanvas, 0, 0)
  maskFeatherContext.restore()

  maskRenderContext.save()
  maskRenderContext.globalAlpha = strength
  maskRenderContext.globalCompositeOperation = 'color'
  maskRenderContext.drawImage(maskFeatherCanvas, 0, 0)
  maskRenderContext.restore()
}

function applyMaskColorOverlay(color: string, alpha: number) {
  maskFeatherContext.clearRect(0, 0, maskFeatherCanvas.width, maskFeatherCanvas.height)
  maskFeatherContext.save()
  maskFeatherContext.fillStyle = color
  maskFeatherContext.fillRect(0, 0, maskFeatherCanvas.width, maskFeatherCanvas.height)
  maskFeatherContext.globalCompositeOperation = 'destination-in'
  maskFeatherContext.drawImage(maskRenderCanvas, 0, 0)
  maskFeatherContext.restore()

  maskRenderContext.save()
  maskRenderContext.globalAlpha = clamp(alpha, 0, 0.35)
  maskRenderContext.globalCompositeOperation = 'soft-light'
  maskRenderContext.drawImage(maskFeatherCanvas, 0, 0)
  maskRenderContext.restore()
}

function applyRenderedMaskFeather(landmarks: NormalizedLandmark[]) {
  if (!maskEdgeFeatherEnabled) {
    return
  }

  const blur = getAdaptiveMaskEdgeFeather(landmarks)

  if (blur <= 0) {
    return
  }

  applyTargetFaceEdgeFeather(landmarks, blur)
}

function applyTargetFaceEdgeFeather(landmarks: NormalizedLandmark[], blur: number) {
  const erosion = Math.max(1, Math.round(blur * 1.05))
  const diagonal = Math.max(1, Math.round(erosion * 0.7))
  const offsets: Point2D[] = [
    { x: erosion, y: 0 },
    { x: -erosion, y: 0 },
    { x: 0, y: erosion },
    { x: 0, y: -erosion },
    { x: diagonal, y: diagonal },
    { x: diagonal, y: -diagonal },
    { x: -diagonal, y: diagonal },
    { x: -diagonal, y: -diagonal },
  ]

  maskAlphaContext.clearRect(0, 0, maskAlphaCanvas.width, maskAlphaCanvas.height)
  maskAlphaContext.save()
  drawFaceOvalPath(maskAlphaContext, landmarks, 1.08)
  maskAlphaContext.fillStyle = '#fff'
  maskAlphaContext.fill()
  maskAlphaContext.restore()

  maskFeatherContext.clearRect(0, 0, maskFeatherCanvas.width, maskFeatherCanvas.height)
  maskFeatherContext.save()
  maskFeatherContext.drawImage(maskAlphaCanvas, 0, 0)
  maskFeatherContext.globalCompositeOperation = 'destination-in'

  for (const offset of offsets) {
    maskFeatherContext.drawImage(maskAlphaCanvas, offset.x, offset.y)
  }

  maskFeatherContext.restore()

  maskAlphaContext.clearRect(0, 0, maskAlphaCanvas.width, maskAlphaCanvas.height)
  maskAlphaContext.save()
  maskAlphaContext.filter = `blur(${blur}px)`
  maskAlphaContext.drawImage(maskFeatherCanvas, 0, 0)
  maskAlphaContext.restore()

  maskRenderContext.save()
  maskRenderContext.globalCompositeOperation = 'destination-in'
  // Feather only the outer face oval, so eyes, mouth and mesh texture details stay solid.
  maskRenderContext.drawImage(maskAlphaCanvas, 0, 0)
  maskRenderContext.drawImage(maskAlphaCanvas, 0, 0)
  maskRenderContext.restore()
}

function getAdaptiveMaskEdgeFeather(landmarks: NormalizedLandmark[]) {
  const pose = getHeadPose(landmarks)
  const turn = clamp(((pose ? Math.abs(pose.yaw) : 0) - 0.16) / 0.32, 0, 1)

  return Math.round(maskEdgeFeather * (1 + turn * 0.85))
}

function sampleMaskSkinColor() {
  const primaryMaskMotion = maskMotionStates[0]
  const landmarks = primaryMaskMotion.displayedLandmarks ?? primaryMaskMotion.previousLandmarks

  if (!isRunning || !video.videoWidth || !video.videoHeight || !landmarks) {
    maskState.textContent = 'Лицо не найдено'
    trackMetrika('mask_skin_sample_error', {
      reason: 'face_not_found',
    })
    return
  }

  syncMaskCanvases()
  maskFeatherContext.clearRect(0, 0, maskFeatherCanvas.width, maskFeatherCanvas.height)
  maskFeatherContext.drawImage(video, 0, 0, maskFeatherCanvas.width, maskFeatherCanvas.height)

  const samples = [50, 280, 205, 425, 151, 200]
    .map((index) => landmarks[index])
    .filter(Boolean)
    .map(landmarkToCanvasPoint)
  const radius = clamp(Math.round(canvas.width * 0.006), 3, 8)
  let red = 0
  let green = 0
  let blue = 0
  let count = 0

  for (const point of samples) {
    const x = clamp(Math.round(point.x - radius), 0, Math.max(maskFeatherCanvas.width - radius * 2 - 1, 0))
    const y = clamp(Math.round(point.y - radius), 0, Math.max(maskFeatherCanvas.height - radius * 2 - 1, 0))
    const size = radius * 2 + 1
    const data = maskFeatherContext.getImageData(x, y, size, size).data

    for (let index = 0; index < data.length; index += 4) {
      const r = data[index]
      const g = data[index + 1]
      const b = data[index + 2]
      const luma = r * 0.2126 + g * 0.7152 + b * 0.0722

      if (luma < 45 || luma > 245 || b > r * 1.25) {
        continue
      }

      red += r
      green += g
      blue += b
      count += 1
    }
  }

  if (!count) {
    maskState.textContent = 'Цвет не взят'
    trackMetrika('mask_skin_sample_error', {
      reason: 'sample_empty',
    })
    return
  }

  setMaskSkinColor(rgbToHex(red / count, green / count, blue / count))

  if (maskColorStrength === 0) {
    setMaskColorStrength(35)
  }

  maskState.textContent = 'Цвет взят с лица'
  trackMetrika('mask_skin_sampled', {
    color: maskSkinColor,
    strength: maskColorStrength,
  })
}

function compensateMaskLag(landmarks: NormalizedLandmark[], now: number, motion: MaskMotionState) {
  const previousRaw = motion.previousLandmarks
  const previousDisplayed = motion.displayedLandmarks
  const previousAt = motion.previousAt
  const current = cloneLandmarks(landmarks)
  motion.previousLandmarks = current
  motion.previousAt = now

  if (!previousRaw || !previousDisplayed || previousRaw.length !== landmarks.length || !previousAt) {
    motion.displayedLandmarks = current
    return current
  }

  const dt = clamp(now - previousAt, 8, 80)
  const stability = maskStability / 100
  const leadFactor = 0.7 - stability * 0.52
  const localLeadFactor = 0.18 - stability * 0.12
  const response = 0.96 - stability * 0.44
  const frameScale = clamp(dt / 33, 0.7, 1.45)
  const maxStep = (0.12 - stability * 0.075) * frameScale
  const globalVelocity = averageLandmarkDelta(landmarks, previousRaw)

  const stabilized = landmarks.map((landmark, index) => {
    const lastRaw = previousRaw[index]
    const lastShown = previousDisplayed[index]
    const localDx = landmark.x - lastRaw.x - globalVelocity.x
    const localDy = landmark.y - lastRaw.y - globalVelocity.y
    const localDz = landmark.z - lastRaw.z - globalVelocity.z
    const predicted = {
      x: landmark.x + globalVelocity.x * leadFactor + localDx * localLeadFactor,
      y: landmark.y + globalVelocity.y * leadFactor + localDy * localLeadFactor,
      z: landmark.z + globalVelocity.z * leadFactor + localDz * localLeadFactor,
    }
    const next = {
      x: lastShown.x + (predicted.x - lastShown.x) * response,
      y: lastShown.y + (predicted.y - lastShown.y) * response,
      z: lastShown.z + (predicted.z - lastShown.z) * response,
    }
    const dx = next.x - lastShown.x
    const dy = next.y - lastShown.y
    const length = Math.hypot(dx, dy)

    if (length > maxStep) {
      const scale = maxStep / length
      next.x = lastShown.x + dx * scale
      next.y = lastShown.y + dy * scale
    }

    return {
      x: clamp(next.x, -0.08, 1.08),
      y: clamp(next.y, -0.08, 1.08),
      z: next.z,
      visibility: landmark.visibility,
    }
  })

  motion.displayedLandmarks = cloneLandmarks(stabilized)
  return stabilized
}

function updateFaceSwapBridge(now: number) {
  if (!maskEnabled || maskMode !== 'faceswap' || !maskSourceBlob || faceSwapInFlight) {
    return
  }

  if (!video.videoWidth || !video.videoHeight || now - faceSwapLastRequestAt < faceSwapRequestIntervalMs) {
    return
  }

  faceSwapLastRequestAt = now
  faceSwapInFlight = true
  void requestFaceSwapFrame()
}

function drawFaceSwapFrame() {
  if (!maskEnabled || maskMode !== 'faceswap' || !faceSwapFrameImage?.complete) {
    return
  }

  context.save()
  context.imageSmoothingEnabled = true
  context.imageSmoothingQuality = 'high'
  context.drawImage(faceSwapFrameImage, 0, 0, canvas.width, canvas.height)
  context.restore()
}

async function requestFaceSwapFrame() {
  const source = maskSourceBlob
  const endpoint = faceSwapEndpoint.value.trim()

  if (!source || !endpoint) {
    faceSwapInFlight = false
    faceSwapState.textContent = 'Нет endpoint'
    return
  }

  try {
    const frame = await captureVideoBlob()
    const body = new FormData()
    body.append('frame', frame, 'frame.jpg')
    body.append('source', source, 'source.png')

    const response = await fetch(endpoint, {
      method: 'POST',
      body,
    })

    if (!response.ok) {
      throw new Error(`FaceSwap bridge ${response.status}`)
    }

    const imageBlob = await response.blob()
    await setFaceSwapFrame(imageBlob)
    faceSwapState.textContent = 'Bridge работает'
    trackFaceSwapBridgeStatus('ok')
  } catch (error) {
    console.error(error)
    faceSwapState.textContent = 'Bridge недоступен'
    trackFaceSwapBridgeStatus('error')
  } finally {
    faceSwapInFlight = false
  }
}

function captureVideoBlob() {
  syncFaceSwapCaptureCanvas()
  faceSwapCaptureContext.drawImage(video, 0, 0, faceSwapCaptureCanvas.width, faceSwapCaptureCanvas.height)

  return new Promise<Blob>((resolve, reject) => {
    faceSwapCaptureCanvas.toBlob(
      (blob) => {
        if (blob) {
          resolve(blob)
        } else {
          reject(new Error('Не удалось снять кадр'))
        }
      },
      'image/jpeg',
      0.82,
    )
  })
}

function syncFaceSwapCaptureCanvas() {
  if (faceSwapCaptureCanvas.width !== video.videoWidth || faceSwapCaptureCanvas.height !== video.videoHeight) {
    faceSwapCaptureCanvas.width = video.videoWidth
    faceSwapCaptureCanvas.height = video.videoHeight
  }
}

function setFaceSwapFrame(blob: Blob) {
  return new Promise<void>((resolve, reject) => {
    const url = URL.createObjectURL(blob)
    const image = new Image()

    image.addEventListener('load', () => {
      resetFaceSwapFrame()
      faceSwapFrameUrl = url
      faceSwapFrameImage = image
      resolve()
    })

    image.addEventListener('error', () => {
      URL.revokeObjectURL(url)
      reject(new Error('Не удалось показать FaceSwap кадр'))
    })

    image.src = url
  })
}

function resetFaceSwapFrame() {
  if (faceSwapFrameUrl) {
    URL.revokeObjectURL(faceSwapFrameUrl)
  }

  faceSwapFrameImage = null
  faceSwapFrameUrl = null
}

function drawWarpedTriangle(
  targetContext: CanvasRenderingContext2D,
  image: CanvasImageSource,
  source: Point2D[],
  target: Point2D[],
  clipBleed = 3.6,
) {
  const [s0, s1, s2] = source
  const [t0, t1, t2] = target
  const clip = expandTriangle(target, clipBleed)
  const determinant = s0.x * (s1.y - s2.y) + s1.x * (s2.y - s0.y) + s2.x * (s0.y - s1.y)

  if (Math.abs(determinant) < 0.001) {
    return
  }

  const a = (t0.x * (s1.y - s2.y) + t1.x * (s2.y - s0.y) + t2.x * (s0.y - s1.y)) / determinant
  const b = (t0.y * (s1.y - s2.y) + t1.y * (s2.y - s0.y) + t2.y * (s0.y - s1.y)) / determinant
  const c = (t0.x * (s2.x - s1.x) + t1.x * (s0.x - s2.x) + t2.x * (s1.x - s0.x)) / determinant
  const d = (t0.y * (s2.x - s1.x) + t1.y * (s0.x - s2.x) + t2.y * (s1.x - s0.x)) / determinant
  const e =
    (t0.x * (s1.x * s2.y - s2.x * s1.y) +
      t1.x * (s2.x * s0.y - s0.x * s2.y) +
      t2.x * (s0.x * s1.y - s1.x * s0.y)) /
    determinant
  const f =
    (t0.y * (s1.x * s2.y - s2.x * s1.y) +
      t1.y * (s2.x * s0.y - s0.x * s2.y) +
      t2.y * (s0.x * s1.y - s1.x * s0.y)) /
    determinant

  targetContext.save()
  targetContext.beginPath()
  targetContext.moveTo(clip[0].x, clip[0].y)
  targetContext.lineTo(clip[1].x, clip[1].y)
  targetContext.lineTo(clip[2].x, clip[2].y)
  targetContext.closePath()
  targetContext.clip()
  targetContext.transform(a, b, c, d, e, f)
  targetContext.drawImage(image, 0, 0)
  targetContext.restore()
}

function applySoftFaceMask(landmarks: NormalizedLandmark[]) {
  maskAlphaContext.clearRect(0, 0, maskAlphaCanvas.width, maskAlphaCanvas.height)
  maskAlphaContext.save()
  maskAlphaContext.filter = `blur(${performanceMode === 'performance' ? 12 : 24}px)`
  drawFaceOvalPath(maskAlphaContext, landmarks, 1.18)
  maskAlphaContext.fillStyle = 'rgba(255, 255, 255, 0.98)'
  maskAlphaContext.fill()
  maskAlphaContext.filter = 'none'
  drawFaceOvalPath(maskAlphaContext, landmarks, 1.05)
  maskAlphaContext.fillStyle = '#fff'
  maskAlphaContext.fill()
  maskAlphaContext.restore()

  maskRenderContext.save()
  maskRenderContext.globalCompositeOperation = 'destination-in'
  maskRenderContext.drawImage(maskAlphaCanvas, 0, 0)
  maskRenderContext.restore()
}

function drawFaceOvalPath(targetContext: CanvasRenderingContext2D, landmarks: NormalizedLandmark[], scale: number) {
  drawFaceOvalPathInSpace(targetContext, landmarks, scale, canvas.width, canvas.height)
}

function drawFaceOvalPathInSpace(
  targetContext: CanvasRenderingContext2D,
  landmarks: NormalizedLandmark[],
  scale: number,
  width: number,
  height: number,
) {
  const points = getFaceMaskBoundaryPointsInSpace(landmarks, width, height)

  if (points.length < 3) {
    return
  }

  const center = polygonCenter(points)
  const expansion = getFaceOvalExpansion(landmarks, width)
  const scaled = points.map((point) => scaleFaceOvalPoint(point, center, scale, expansion))
  const firstPoint = scaled[0]
  const lastPoint = scaled[scaled.length - 1]

  targetContext.beginPath()
  targetContext.moveTo((lastPoint.x + firstPoint.x) / 2, (lastPoint.y + firstPoint.y) / 2)

  for (let index = 0; index < scaled.length; index += 1) {
    const current = scaled[index]
    const next = scaled[(index + 1) % scaled.length]
    const controlX = (current.x + next.x) / 2
    const controlY = (current.y + next.y) / 2
    targetContext.quadraticCurveTo(current.x, current.y, controlX, controlY)
  }

  targetContext.closePath()
}

function getFaceMaskBoundaryPointsInSpace(landmarks: NormalizedLandmark[], width: number, height: number) {
  const points = landmarks
    .map((landmark) => landmarkToPointInSpace(landmark, width, height))
    .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y))

  if (points.length < 3) {
    return points
  }

  return convexHull(points)
}

function getFaceOvalExpansion(landmarks: NormalizedLandmark[], width = canvas.width) {
  const pose = getHeadPose(landmarks)
  const yaw = pose?.yaw ?? 0
  const pitch = pose?.pitch ?? 0
  const turn = clamp((Math.abs(yaw) - 0.1) / 0.34, 0, 1)
  const pitchLift = clamp(Math.abs(pitch) / 0.35, 0, 1)

  return {
    x: 1 + turn * 0.32,
    y: 1 + turn * 0.13 + pitchLift * 0.05,
    biasX: yaw * turn * 0.06 * width,
  }
}

function scaleFaceOvalPoint(
  point: Point2D,
  center: Point2D,
  scale: number,
  expansion: { x: number; y: number; biasX: number },
): Point2D {
  const dx = point.x - center.x
  const dy = point.y - center.y

  return {
    x: center.x + expansion.biasX + dx * scale * expansion.x,
    y: center.y + dy * scale * expansion.y,
  }
}

function syncMaskCanvases() {
  if (maskRenderCanvas.width !== canvas.width || maskRenderCanvas.height !== canvas.height) {
    maskRenderCanvas.width = canvas.width
    maskRenderCanvas.height = canvas.height
    maskAlphaCanvas.width = canvas.width
    maskAlphaCanvas.height = canvas.height
    maskFeatherCanvas.width = canvas.width
    maskFeatherCanvas.height = canvas.height
  }
}

function fireFaceSignal(
  gesture: GestureKey,
  confidence: number,
  detected: Set<GestureKey>,
  details?: GestureEvent['details'],
) {
  detected.add(gesture)
  fireGesture(gesture, confidence, 'face', details)
}

function faceScore(categories: Category[], name: string) {
  return categories.find((category) => category.categoryName === name)?.score ?? 0
}

function getHeadPose(landmarks: NormalizedLandmark[]) {
  const leftEye = landmarks[33]
  const rightEye = landmarks[263]
  const nose = landmarks[1]
  const upperLip = landmarks[13]
  const lowerLip = landmarks[14]

  if (!leftEye || !rightEye || !nose || !upperLip || !lowerLip) {
    return null
  }

  const eyeCenter = midpoint(leftEye, rightEye)
  const mouthCenter = midpoint(upperLip, lowerLip)
  const eyeDistance = Math.max(distance(leftEye, rightEye), 0.001)
  const eyeToMouth = Math.max(distance(eyeCenter, mouthCenter), 0.001)

  return {
    yaw: (nose.x - eyeCenter.x) / eyeDistance,
    pitch: (nose.y - eyeCenter.y) / eyeToMouth - 0.54,
  }
}

function updateCursor(landmarks: NormalizedLandmark[]) {
  if (!handMarkersEnabled) {
    cursorDot.classList.remove('is-visible')
    return
  }

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
  trackMetrika('gesture_used', {
    gesture,
    gestureTitle: event.gestureTitle,
    preset: currentPreset,
    action: event.action,
    source,
    confidence: event.confidence,
  })

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
    trackMetrika('event_json_copied', {
      gesture: lastEvent.gesture,
      source: lastEvent.source,
    })
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
      const nextPreset = button.dataset.preset as PresetId

      if (nextPreset === currentPreset) {
        return
      }

      const previousPreset = currentPreset
      currentPreset = nextPreset
      localStorage.setItem('xedoc-hands-preset', currentPreset)
      renderPresetTabs()
      renderGestureGrid()
      refreshIcons()
      trackMetrika('preset_changed', {
        preset: currentPreset,
        previousPreset,
      })
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

function renderMaskModeTabs() {
  maskModeTabs.querySelectorAll<HTMLButtonElement>('.mask-mode-button').forEach((button) => {
    button.classList.toggle('is-active', button.dataset.maskMode === maskMode)
  })
}

function renderPerformanceModeTabs() {
  performanceModeTabs.querySelectorAll<HTMLButtonElement>('.mask-mode-button').forEach((button) => {
    button.classList.toggle('is-active', button.dataset.performanceMode === performanceMode)
  })
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

function trackMetrika(goal: string, params: MetrikaParams = {}) {
  const ym = window.ym

  if (typeof ym !== 'function') {
    return
  }

  ym(metrikaCounterId, 'reachGoal', goal, params)

  if (Object.keys(params).length > 0) {
    ym(metrikaCounterId, 'params', {
      [goal]: params,
    })
  }
}

function trackToggle(goal: string, enabled: boolean, params: MetrikaParams = {}) {
  trackMetrika(goal, {
    enabled,
    ...params,
  })
}

function trackMaskSetting(setting: string, value: string | number) {
  trackMetrika('mask_setting_changed', {
    setting,
    value,
    maskMode,
    maskEnabled,
    maskCount: getLoadedMaskCount(),
  })
}

function trackMaskSettingToggle(setting: string, enabled: boolean) {
  trackMetrika('mask_setting_toggled', {
    setting,
    enabled,
    maskMode,
    maskEnabled,
    maskCount: getLoadedMaskCount(),
  })
}

function trackFaceCount(count: number) {
  maxTrackedFaceCount = Math.max(maxTrackedFaceCount, count)

  if (count === lastTrackedFaceCount) {
    return
  }

  lastTrackedFaceCount = count
  trackMetrika('face_count_changed', {
    count,
    maxCount: maxTrackedFaceCount,
    multiFace: multiFaceTrackingEnabled,
    faceLimit: getFaceCountLimit(),
    maskEnabled,
    maskCount: getLoadedMaskCount(),
  })
}

function getLoadedMaskCount() {
  return Number(Boolean(maskLayer)) + extraMaskSlots.filter((slot) => Boolean(slot.layer)).length
}

function getMaskSlotAnalyticsName(faceRank: number) {
  return faceRank === 0 ? 'main' : `face_${faceRank + 1}`
}

function trackFaceSwapBridgeStatus(status: 'ok' | 'error') {
  if (faceSwapBridgeStatus === status) {
    return
  }

  faceSwapBridgeStatus = status
  trackMetrika('faceswap_bridge_status', {
    status,
  })
}

function setModelState(state: 'loading' | 'ready' | 'error', text: string) {
  modelDot.dataset.state = state
  modelStatus.textContent = text
}

function setCameraState(state: 'idle' | 'loading' | 'ready' | 'error', text: string) {
  cameraDot.dataset.state = state
  cameraStatus.textContent = text
}

function setFaceState(state: 'idle' | 'loading' | 'ready' | 'error', text: string) {
  faceDot.dataset.state = state
  faceStatus.textContent = text
}

function loadMaskFile(file: File, faceRank = 0) {
  const slot = getMaskSlotAnalyticsName(faceRank)

  if (!file.type.startsWith('image/')) {
    setMaskLoadState(faceRank, 'Не изображение')
    trackMetrika('mask_upload_error', {
      slot,
      reason: 'not_image',
    })
    return
  }

  setMaskLoadState(faceRank, 'Загрузка')
  trackMetrika('mask_upload_started', {
    slot,
    fileType: file.type || 'unknown',
    fileSize: file.size,
  })

  const nextUrl = URL.createObjectURL(file)
  const nextImage = new Image()

  nextImage.addEventListener('load', () => {
    void prepareMaskLayer(nextImage, nextUrl, file, faceRank)
  })

  nextImage.addEventListener('error', () => {
    URL.revokeObjectURL(nextUrl)
    setMaskLoadState(faceRank, 'Не загрузилась')
    trackMetrika('mask_upload_error', {
      slot,
      reason: 'image_load',
    })
  })

  nextImage.src = nextUrl
}

async function prepareMaskLayer(image: HTMLImageElement, imageUrl: string, sourceBlob: Blob, faceRank = 0) {
  const analyticsSlot = getMaskSlotAnalyticsName(faceRank)
  setMaskLoadState(faceRank, 'Ищем лицо')

  if (!maskFaceLandmarker) {
    await bootModel()
  }

  if (!maskFaceLandmarker) {
    URL.revokeObjectURL(imageUrl)
    setMaskLoadState(faceRank, 'Модель не готова')
    trackMetrika('mask_upload_error', {
      slot: analyticsSlot,
      reason: 'model_not_ready',
    })
    return
  }

  const result = maskFaceLandmarker.detect(image)
  const landmarks = result.faceLandmarks[0]
  const layer = landmarks
    ? {
        image,
        landmarks,
        triangles: faceTriangles,
        width: image.naturalWidth,
        height: image.naturalHeight,
      }
    : null

  if (faceRank > 0) {
    const slot = extraMaskSlots[faceRank - 1]

    if (!slot) {
      URL.revokeObjectURL(imageUrl)
      trackMetrika('mask_upload_error', {
        slot: analyticsSlot,
        reason: 'slot_unavailable',
      })
      return
    }

    if (!layer) {
      URL.revokeObjectURL(imageUrl)
      updateMaskState()
      setMaskSlotState(faceRank, 'Лицо не найдено')
      trackMetrika('mask_upload_error', {
        slot: analyticsSlot,
        reason: 'face_not_found',
      })
      return
    }

    revokeMaskSlot(slot)
    slot.layer = layer
    slot.imageUrl = imageUrl
    setMaskSlotState(faceRank, 'Своя маска')
    setMaskEnabled(true)
    trackMetrika('mask_changed', {
      slot: analyticsSlot,
      mode: maskMode,
      width: image.naturalWidth,
      height: image.naturalHeight,
      faceDetected: true,
      maskCount: getLoadedMaskCount(),
    })
    return
  }

  if (maskImageUrl) {
    URL.revokeObjectURL(maskImageUrl)
  }

  resetFaceSwapFrame()

  maskLayer = layer
  maskImageUrl = imageUrl
  maskSourceBlob = sourceBlob
  maskPreview.src = imageUrl
  setMaskEnabled(true)
  trackMetrika('mask_changed', {
    slot: analyticsSlot,
    mode: maskMode,
    width: image.naturalWidth,
    height: image.naturalHeight,
    faceDetected: Boolean(layer),
    maskCount: getLoadedMaskCount(),
  })

  if (!landmarks) {
    maskState.textContent = maskMode === 'mesh' ? 'Лицо не найдено' : 'FaceSwap готов'
  }
}

function setMaskLoadState(faceRank: number, text: string) {
  if (faceRank > 0) {
    setMaskSlotState(faceRank, text)
    return
  }

  maskState.textContent = text
}

function setMaskSlotState(faceRank: number, text: string) {
  const state = maskSlotStates[faceRank - 1]

  if (state) {
    state.textContent = text
  }
}

function updateMaskSlotStates() {
  extraMaskSlots.forEach((slot, index) => {
    setMaskSlotState(index + 1, slot.layer ? 'Своя маска' : 'Основная')
  })
}

function revokeMaskSlot(slot: FaceMaskSlot) {
  if (slot.imageUrl) {
    URL.revokeObjectURL(slot.imageUrl)
  }

  slot.layer = null
  slot.imageUrl = null
}

function setMaskEnabled(next: boolean) {
  maskEnabled = next
  localStorage.setItem('xedoc-hands-mask-enabled', String(next))

  if (!next) {
    resetFaceSwapFrame()
  }

  updateMaskState()
}

function setHandTrackingEnabled(next: boolean) {
  handTrackingEnabled = next
  localStorage.setItem('xedoc-hands-hand-tracking', next ? 'on' : 'off')
  handTrackingToggle.checked = next
  handTrackingState.textContent = next ? 'Трекинг включен' : 'Трекинг выключен'
  updateHandMarkersState()
  resetHandTracking()

  if (!next) {
    setReadout('Руки выкл.', 0, 0, 0, 0, 0)
    setActiveGestures(new Set())
    return
  }

  if (!isRunning) {
    setReadout('Нет руки', 0, 0, 0, 0, 0)
  }
}

function setHandMarkersEnabled(next: boolean) {
  handMarkersEnabled = next
  localStorage.setItem('xedoc-hands-hand-markers', next ? 'on' : 'off')
  handMarkersToggle.checked = next
  if (!next) {
    cursorDot.classList.remove('is-visible')
  }
  updateHandMarkersState()
}

function updateHandMarkersState() {
  handMarkersToggle.disabled = !handTrackingEnabled

  if (!handTrackingEnabled) {
    handMarkersState.textContent = 'Трекинг рук выключен'
    return
  }

  handMarkersState.textContent = handMarkersEnabled ? 'Линии и точки включены' : 'Линии и точки скрыты'
}

function setFaceTrackingEnabled(next: boolean) {
  faceTrackingEnabled = next
  localStorage.setItem('xedoc-hands-face-tracking', next ? 'on' : 'off')
  faceTrackingToggle.checked = next
  faceTrackingState.textContent = next ? 'Трекинг и маркеры включены' : 'Трекинг и маркеры выключены'

  if (!next) {
    faceCount.textContent = '0'
    headSamples = []
    resetMaskMotion()
    setFaceState('idle', 'Откл.')
    updateMaskState()
    return
  }

  setFaceState('idle', isRunning ? 'Нет лица' : 'Ожидание')
  updateMaskState()
}

function setMultiFaceTrackingEnabled(next: boolean, rebuild = true) {
  multiFaceTrackingEnabled = next
  localStorage.setItem('xedoc-hands-multi-face', next ? 'on' : 'off')
  multiFaceToggle.checked = next
  multiFaceState.textContent = next ? 'До 4 лиц' : '1 лицо'

  if (rebuild && faceLandmarker) {
    void rebuildFaceLandmarker()
  }
}

function setHudVisible(next: boolean) {
  hudVisible = next
  localStorage.setItem('xedoc-hands-hud', next ? 'on' : 'off')
  hudToggle.checked = next
  hud.classList.toggle('is-hidden', !next)
  hudState.textContent = next ? 'HUD включен' : 'HUD скрыт'
}

function setPerformanceMode(next: PerformanceMode) {
  performanceMode = next
  localStorage.setItem('xedoc-hands-performance-mode', next)
  renderPerformanceModeTabs()
  performanceTitle.textContent = next === 'performance' ? 'Быстрее' : 'Качество'
  performanceModeState.textContent = next === 'performance' ? 'Камера 640x480' : 'Камера 1280x720'
}

function setMaskMode(next: MaskMode) {
  maskMode = next
  localStorage.setItem('xedoc-hands-mask-mode', next)
  renderMaskModeTabs()
  resetMaskMotion()
  resetFaceSwapFrame()
  updateMaskState()
}

function setMaskStability(next: number) {
  maskStability = clamp(Math.round(next), 0, 100)
  localStorage.setItem('xedoc-hands-mask-stability', String(maskStability))
  maskStabilitySlider.value = String(maskStability)
  maskStabilityValue.textContent = `${maskStability}%`
}

function setMaskEdgeFeather(next: number) {
  maskEdgeFeather = clamp(Math.round(next), 0, 80)
  localStorage.setItem('xedoc-hands-mask-edge-feather', String(maskEdgeFeather))
  maskEdgeFeatherSlider.value = String(maskEdgeFeather)
  maskEdgeFeatherValue.textContent = `${maskEdgeFeather}px`
}

function setMaskEdgeFeatherEnabled(next: boolean) {
  maskEdgeFeatherEnabled = next
  setMaskSettingEnabled('edge-feather', next)
  maskEdgeFeatherEnabledToggle.checked = next
  maskEdgeFeatherSlider.disabled = !next
}

function setMaskSkinColor(next: string) {
  maskSkinColor = normalizeHexColor(next, '#f2c7ad')
  localStorage.setItem('xedoc-hands-mask-skin-color', maskSkinColor)
  maskSkinColorInput.value = maskSkinColor
}

function setMaskColorStrength(next: number) {
  maskColorStrength = clamp(Math.round(next), 0, 100)
  localStorage.setItem('xedoc-hands-mask-color-strength', String(maskColorStrength))
  maskColorStrengthSlider.value = String(maskColorStrength)
  maskColorStrengthValue.textContent = `${maskColorStrength}%`
}

function setMaskColorStrengthEnabled(next: boolean) {
  maskColorStrengthEnabled = next
  setMaskSettingEnabled('color-strength', next)
  maskColorStrengthEnabledToggle.checked = next
  maskSkinColorInput.disabled = !next
  maskSampleSkinButton.disabled = !next
  maskColorStrengthSlider.disabled = !next
}

function setMaskBrightness(next: number) {
  maskBrightness = clamp(Math.round(next), -40, 40)
  localStorage.setItem('xedoc-hands-mask-brightness', String(maskBrightness))
  maskBrightnessSlider.value = String(maskBrightness)
  maskBrightnessValue.textContent = signedValue(maskBrightness)
}

function setMaskBrightnessEnabled(next: boolean) {
  maskBrightnessEnabled = next
  setMaskSettingEnabled('brightness', next)
  maskBrightnessEnabledToggle.checked = next
  maskBrightnessSlider.disabled = !next
}

function setMaskSaturation(next: number) {
  maskSaturation = clamp(Math.round(next), -50, 60)
  localStorage.setItem('xedoc-hands-mask-saturation', String(maskSaturation))
  maskSaturationSlider.value = String(maskSaturation)
  maskSaturationValue.textContent = signedValue(maskSaturation)
}

function setMaskSaturationEnabled(next: boolean) {
  maskSaturationEnabled = next
  setMaskSettingEnabled('saturation', next)
  maskSaturationEnabledToggle.checked = next
  maskSaturationSlider.disabled = !next
}

function setMaskContrast(next: number) {
  maskContrast = clamp(Math.round(next), -40, 60)
  localStorage.setItem('xedoc-hands-mask-contrast', String(maskContrast))
  maskContrastSlider.value = String(maskContrast)
  maskContrastValue.textContent = signedValue(maskContrast)
}

function setMaskContrastEnabled(next: boolean) {
  maskContrastEnabled = next
  setMaskSettingEnabled('contrast', next)
  maskContrastEnabledToggle.checked = next
  maskContrastSlider.disabled = !next
}

function setMaskTemperature(next: number) {
  maskTemperature = clamp(Math.round(next), -50, 50)
  localStorage.setItem('xedoc-hands-mask-temperature', String(maskTemperature))
  maskTemperatureSlider.value = String(maskTemperature)
  maskTemperatureValue.textContent = signedValue(maskTemperature)
}

function setMaskTemperatureEnabled(next: boolean) {
  maskTemperatureEnabled = next
  setMaskSettingEnabled('temperature', next)
  maskTemperatureEnabledToggle.checked = next
  maskTemperatureSlider.disabled = !next
}

function setMaskTint(next: number) {
  maskTint = clamp(Math.round(next), -50, 50)
  localStorage.setItem('xedoc-hands-mask-tint', String(maskTint))
  maskTintSlider.value = String(maskTint)
  maskTintValue.textContent = signedValue(maskTint)
}

function setMaskTintEnabled(next: boolean) {
  maskTintEnabled = next
  setMaskSettingEnabled('tint', next)
  maskTintEnabledToggle.checked = next
  maskTintSlider.disabled = !next
}

function updateMaskState() {
  const hasMainImage = Boolean(maskImageUrl && maskSourceBlob)
  const meshReady = hasMeshMaskLayer()
  const canEnable = maskMode === 'faceswap' ? hasMainImage : meshReady

  maskToggle.disabled = !canEnable
  maskToggle.checked = canEnable && maskEnabled
  maskPreview.classList.toggle('is-visible', hasMainImage)
  maskStabilityBlock.classList.toggle('is-hidden', maskMode !== 'mesh')
  maskEdgeFeatherBlock.classList.toggle('is-hidden', maskMode !== 'mesh')
  maskColorBlock.classList.toggle('is-hidden', maskMode !== 'mesh')
  maskSlotsBlock.classList.toggle('is-hidden', maskMode !== 'mesh')
  faceSwapOptions.classList.toggle('is-hidden', maskMode !== 'faceswap')
  updateMaskSlotStates()

  if (!hasMainImage && !meshReady) {
    maskState.textContent = 'Не выбрана'
    return
  }

  if (maskMode === 'mesh' && !meshReady) {
    maskState.textContent = 'Лицо не найдено'
    return
  }

  if (maskMode === 'mesh' && !faceTrackingEnabled && maskEnabled) {
    maskState.textContent = 'Трекинг лица выключен'
    return
  }

  if (!maskEnabled) {
    maskState.textContent = 'Выключена'
    return
  }

  if (maskMode === 'faceswap') {
    maskState.textContent = 'FaceSwap включен'
    return
  }

  const maskCount = Number(Boolean(maskLayer)) + extraMaskSlots.filter((slot) => Boolean(slot.layer)).length
  maskState.textContent = maskCount > 1 ? `Mesh: ${maskCount} маски` : 'Mesh включена'
}

function hasMeshMaskLayer() {
  return Boolean(maskLayer || extraMaskSlots.some((slot) => slot.layer))
}

function rememberMaskFace(landmarks: NormalizedLandmark[], now: number, motion = maskMotionStates[0]) {
  const current = cloneLandmarks(landmarks)
  motion.previousLandmarks = current
  motion.displayedLandmarks = cloneLandmarks(landmarks)
  motion.previousAt = now
}

function resetMaskMotion() {
  maskMotionStates = createMaskMotionStates()
}

function createMaskMotionStates() {
  return Array.from({ length: faceMaskSlotCount }, (): MaskMotionState => ({
    previousLandmarks: null,
    displayedLandmarks: null,
    previousAt: 0,
  }))
}

function resetHandTracking() {
  pinchDown = false
  pinchStartedAt = null
  pinchHoldDown = false
  motionSamples = []
  zoomSamples = []
  cursorDot.classList.remove('is-visible')
}

function getCameraConstraints(): MediaStreamConstraints {
  const size =
    performanceMode === 'performance'
      ? { width: { ideal: 640 }, height: { ideal: 480 } }
      : { width: { ideal: 1280 }, height: { ideal: 720 } }

  return {
    audio: false,
    video: {
      ...size,
      frameRate: { ideal: 30, max: 30 },
      facingMode: 'user',
    },
  }
}

function getCameraErrorMessage(error: unknown) {
  if (!(error instanceof DOMException)) {
    return 'Камера не запустилась'
  }

  if (error.name === 'NotAllowedError' || error.name === 'SecurityError') {
    return 'Доступ к камере запрещен'
  }

  if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
    return 'Камера не найдена'
  }

  if (error.name === 'NotReadableError' || error.name === 'TrackStartError') {
    return 'Камера занята другой программой'
  }

  if (error.name === 'OverconstrainedError' || error.name === 'ConstraintNotSatisfiedError') {
    return 'Камера не поддерживает выбранный режим'
  }

  return 'Камера не запустилась'
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

function readMaskMode(): MaskMode {
  const saved = localStorage.getItem('xedoc-hands-mask-mode')
  return saved === 'faceswap' ? 'faceswap' : 'mesh'
}

function readPerformanceMode(): PerformanceMode {
  const saved = localStorage.getItem('xedoc-hands-performance-mode')

  if (saved === 'performance' || saved === 'quality') {
    return saved
  }

  return isMobileDevice() ? 'performance' : 'quality'
}

function isMobileDevice() {
  const userAgent = navigator.userAgent
  const isAppleMobile = /iPad|iPhone|iPod/.test(userAgent)
  const isCompactTouch = window.matchMedia('(pointer: coarse)').matches && Math.min(screen.width, screen.height) <= 820

  return isAppleMobile || isCompactTouch
}

function readMaskStability() {
  const saved = Number(localStorage.getItem('xedoc-hands-mask-stability'))
  return Number.isFinite(saved) ? clamp(Math.round(saved), 0, 100) : 25
}

function readMaskEdgeFeather() {
  const raw = localStorage.getItem('xedoc-hands-mask-edge-feather')
  const saved = raw === null ? Number.NaN : Number(raw)
  return Number.isFinite(saved) ? clamp(Math.round(saved), 0, 80) : performanceMode === 'performance' ? 12 : 18
}

function readMaskSettingEnabled(key: string) {
  return localStorage.getItem(`xedoc-hands-mask-${key}-enabled`) !== 'off'
}

function setMaskSettingEnabled(key: string, enabled: boolean) {
  localStorage.setItem(`xedoc-hands-mask-${key}-enabled`, enabled ? 'on' : 'off')
}

function readMaskSkinColor() {
  return normalizeHexColor(localStorage.getItem('xedoc-hands-mask-skin-color') ?? '', '#f2c7ad')
}

function readMaskColorStrength() {
  const raw = localStorage.getItem('xedoc-hands-mask-color-strength')
  const saved = raw === null ? Number.NaN : Number(raw)
  return Number.isFinite(saved) ? clamp(Math.round(saved), 0, 100) : 0
}

function readMaskBrightness() {
  const raw = localStorage.getItem('xedoc-hands-mask-brightness')
  const saved = raw === null ? Number.NaN : Number(raw)
  return Number.isFinite(saved) ? clamp(Math.round(saved), -40, 40) : 0
}

function readMaskSaturation() {
  const raw = localStorage.getItem('xedoc-hands-mask-saturation')
  const saved = raw === null ? Number.NaN : Number(raw)
  return Number.isFinite(saved) ? clamp(Math.round(saved), -50, 60) : 0
}

function readMaskContrast() {
  const raw = localStorage.getItem('xedoc-hands-mask-contrast')
  const saved = raw === null ? Number.NaN : Number(raw)
  return Number.isFinite(saved) ? clamp(Math.round(saved), -40, 60) : 0
}

function readMaskTemperature() {
  const raw = localStorage.getItem('xedoc-hands-mask-temperature')
  const saved = raw === null ? Number.NaN : Number(raw)
  return Number.isFinite(saved) ? clamp(Math.round(saved), -50, 50) : 0
}

function readMaskTint() {
  const raw = localStorage.getItem('xedoc-hands-mask-tint')
  const saved = raw === null ? Number.NaN : Number(raw)
  return Number.isFinite(saved) ? clamp(Math.round(saved), -50, 50) : 0
}

function getElement<T extends HTMLElement>(id: string) {
  const element = document.getElementById(id)

  if (!element) {
    throw new Error(`Element #${id} was not found`)
  }

  return element as T
}

function getCanvasContext(targetCanvas: HTMLCanvasElement) {
  const targetContext = targetCanvas.getContext('2d')

  if (!targetContext) {
    throw new Error('Canvas context was not found')
  }

  return targetContext
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

function buildFaceTriangles(connections: readonly { start: number; end: number }[]): FaceTriangle[] {
  const adjacency = new Map<number, Set<number>>()
  const addEdge = (start: number, end: number) => {
    if (!adjacency.has(start)) {
      adjacency.set(start, new Set())
    }

    adjacency.get(start)!.add(end)
  }

  for (const connection of connections) {
    addEdge(connection.start, connection.end)
    addEdge(connection.end, connection.start)
  }

  const seen = new Set<string>()
  const triangles: FaceTriangle[] = []

  for (const [start, neighbors] of adjacency) {
    const list = [...neighbors]

    for (let firstIndex = 0; firstIndex < list.length; firstIndex += 1) {
      for (let secondIndex = firstIndex + 1; secondIndex < list.length; secondIndex += 1) {
        const first = list[firstIndex]
        const second = list[secondIndex]

        if (!adjacency.get(first)?.has(second)) {
          continue
        }

        const ordered = [start, first, second].sort((left, right) => left - right)
        const key = ordered.join(':')

        if (!seen.has(key)) {
          seen.add(key)
          triangles.push([ordered[0], ordered[1], ordered[2]])
        }
      }
    }
  }

  return triangles
}

function landmarkToCanvasPoint(landmark: NormalizedLandmark): Point2D {
  return landmarkToPointInSpace(landmark, canvas.width, canvas.height)
}

function landmarkToPointInSpace(landmark: NormalizedLandmark, width: number, height: number): Point2D {
  return {
    x: landmark.x * width,
    y: landmark.y * height,
  }
}

function landmarkToSourcePoint(landmark: NormalizedLandmark, layer: FaceMaskLayer): Point2D {
  return {
    x: landmark.x * layer.width,
    y: landmark.y * layer.height,
  }
}

function rgbToHex(red: number, green: number, blue: number) {
  return `#${hexChannel(red)}${hexChannel(green)}${hexChannel(blue)}`
}

function hexChannel(value: number) {
  return clamp(Math.round(value), 0, 255).toString(16).padStart(2, '0')
}

function normalizeHexColor(value: string, fallback: string) {
  return /^#[0-9a-f]{6}$/i.test(value) ? value.toLowerCase() : fallback
}

function signedValue(value: number) {
  return value > 0 ? `+${value}` : String(value)
}

function averageLandmarkDelta(current: NormalizedLandmark[], previous: NormalizedLandmark[]) {
  let x = 0
  let y = 0
  let z = 0
  let count = 0

  for (const index of faceOvalIndices) {
    const next = current[index]
    const last = previous[index]

    if (!next || !last) {
      continue
    }

    x += next.x - last.x
    y += next.y - last.y
    z += next.z - last.z
    count += 1
  }

  if (!count) {
    return { x: 0, y: 0, z: 0 }
  }

  return {
    x: x / count,
    y: y / count,
    z: z / count,
  }
}

function expandTriangle(points: Point2D[], pixels: number) {
  const center = polygonCenter(points)

  return points.map((point) => {
    const dx = point.x - center.x
    const dy = point.y - center.y
    const length = Math.max(Math.hypot(dx, dy), 0.001)

    return {
      x: point.x + (dx / length) * pixels,
      y: point.y + (dy / length) * pixels,
    }
  })
}

function polygonCenter(points: Point2D[]): Point2D {
  const sum = points.reduce(
    (acc, point) => ({
      x: acc.x + point.x,
      y: acc.y + point.y,
    }),
    { x: 0, y: 0 },
  )

  return {
    x: sum.x / points.length,
    y: sum.y / points.length,
  }
}

function triangleArea(points: Point2D[]) {
  const [a, b, c] = points
  return ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)) / 2
}

function convexHull(points: Point2D[]) {
  const sorted = [...points].sort((left, right) => left.x - right.x || left.y - right.y)
  const lower: Point2D[] = []

  for (const point of sorted) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], point) <= 0) {
      lower.pop()
    }

    lower.push(point)
  }

  const upper: Point2D[] = []

  for (let index = sorted.length - 1; index >= 0; index -= 1) {
    const point = sorted[index]

    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], point) <= 0) {
      upper.pop()
    }

    upper.push(point)
  }

  return lower.slice(0, -1).concat(upper.slice(0, -1))
}

function cross(origin: Point2D, first: Point2D, second: Point2D) {
  return (first.x - origin.x) * (second.y - origin.y) - (first.y - origin.y) * (second.x - origin.x)
}

function cloneLandmarks(landmarks: NormalizedLandmark[]) {
  return landmarks.map((landmark) => ({ ...landmark }))
}

function analyzeHand(landmarks: NormalizedLandmark[]) {
  const wrist = landmarks[0]
  const palmWidth = Math.max(distance(landmarks[5], landmarks[17]), 0.001)
  const isExtended = (tip: number, pip: number) =>
    distance(landmarks[tip], wrist) > distance(landmarks[pip], wrist) * 1.08
  const thumb =
    distance(landmarks[4], wrist) > distance(landmarks[3], wrist) * 1.06 &&
    distance(landmarks[4], landmarks[5]) > palmWidth * 0.72
  const index = isExtended(8, 6)
  const middle = isExtended(12, 10)
  const ring = isExtended(16, 14)
  const pinky = isExtended(20, 18)
  const averageLongTipY = (landmarks[8].y + landmarks[12].y + landmarks[16].y + landmarks[20].y) / 4
  const longCount = [index, middle, ring, pinky].filter(Boolean).length

  return {
    thumb,
    index,
    middle,
    ring,
    pinky,
    longCount,
    palmWidth,
    averageLongTipY,
  }
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

function midpoint(a: NormalizedLandmark, b: NormalizedLandmark): NormalizedLandmark {
  return {
    x: (a.x + b.x) / 2,
    y: (a.y + b.y) / 2,
    z: (a.z + b.z) / 2,
    visibility: Math.min(a.visibility ?? 1, b.visibility ?? 1),
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
