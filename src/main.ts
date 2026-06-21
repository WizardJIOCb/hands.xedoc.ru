import './style.css'
import { maskPresets } from './generated/maskPresets'
import {
  DrawingUtils,
  FaceLandmarker,
  FilesetResolver,
  GestureRecognizer,
  PoseLandmarker,
  type Category,
  type FaceLandmarkerResult,
  type GestureRecognizerResult,
  type NormalizedLandmark,
  type PoseLandmarkerResult,
} from '@mediapipe/tasks-vision'
import * as THREE from 'three'
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js'
import { VRM, VRMHumanBoneName, VRMLoaderPlugin } from '@pixiv/three-vrm'
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
  Settings2,
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

type Rect2D = {
  minX: number
  minY: number
  maxX: number
  maxY: number
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

type AvatarRig = {
  scene: THREE.Scene
  camera: THREE.PerspectiveCamera
  renderer: THREE.WebGLRenderer
  loader: GLTFLoader
  lightRig: THREE.Group
  fallback: THREE.Group
  fallbackBones: Record<string, THREE.Object3D>
  currentVrm: VRM | null
  currentObject: THREE.Object3D | null
  expressionState: {
    aa: number
    happy: number
    blink: number
    blinkLeft: number
    blinkRight: number
  }
  lastRenderedAt: number
}

type AvatarSide = 'left' | 'right'

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
const poseModelPath =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task'
const defaultAvatarModelUrl =
  'https://raw.githubusercontent.com/pixiv/three-vrm/master/packages/three-vrm/examples/models/VRM1_Constraint_Twist_Sample.vrm'
const avatarModelPresets = [
  {
    id: 'twist',
    label: 'Текущая',
    name: 'Тестовая VRM',
    url: defaultAvatarModelUrl,
  },
  {
    id: 'alicia',
    label: 'Alicia',
    name: 'AliciaSolid VRM',
    url: 'https://raw.githubusercontent.com/vrm-c/UniVRM/master/Tests/Models/Alicia_vrm-0.51/AliciaSolid_vrm-0.51.vrm',
  },
] as const

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
  Settings2,
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
const defaultGestureKeys = gestureDefinitions.map((gesture) => gesture.key)
const shapeGestureKeys: GestureKey[] = [
  'OK_Gesture',
  'Finger_Gun',
  'Three_Fingers',
  'Four_Fingers',
  'Palm_Up',
  'Palm_Down',
]
const swipeGestureKeys: GestureKey[] = ['Swipe_Left', 'Swipe_Right', 'Swipe_Up', 'Swipe_Down']
const zoomGestureKeys: GestureKey[] = ['Zoom_In', 'Zoom_Out']
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
  <div class="app-shell" id="appShell">
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
        <button class="button" id="settingsButton" type="button" title="Показать или скрыть настройки">
          <i data-lucide="settings-2"></i>
          <span>Настройки</span>
        </button>
      </div>
    </header>

    <main class="workspace">
      <section class="stage-area" aria-label="Камера и жесты">
        <div class="stage" id="stage">
          <video id="cameraVideo" playsinline muted></video>
          <canvas id="avatarCanvas"></canvas>
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

        <section class="panel avatar-panel">
          <div class="panel-head">
            <h2>3D Аватар</h2>
            <label class="switch">
              <input id="avatarToggle" type="checkbox" />
              <span></span>
            </label>
          </div>
          <div class="mask-control avatar-control">
            <label class="file-button" for="avatarFile">
              <i data-lucide="bot"></i>
              <span>VRM</span>
            </label>
            <input id="avatarFile" class="file-input" type="file" accept=".vrm,.glb,model/gltf-binary" />
            <button class="mask-mode-button mask-sample-button" id="avatarSampleButton" type="button">Тестовая</button>
          </div>
          <div class="avatar-model-list" aria-label="Модели аватара">
            ${avatarModelPresets.map((model) => `
              <button
                class="mask-mode-button avatar-model-button"
                type="button"
                data-avatar-model="${model.id}"
              >${model.label}</button>
            `).join('')}
          </div>
          <div class="mask-control avatar-control avatar-bg-control">
            <label class="file-button" for="avatarBackgroundFile">
              <i data-lucide="scan-eye"></i>
              <span>Фон</span>
            </label>
            <input id="avatarBackgroundFile" class="file-input" type="file" accept="image/*" />
            <button class="mask-mode-button mask-sample-button" id="avatarBackgroundClearButton" type="button">Очистить</button>
          </div>
          <p class="panel-state" id="avatarState">Выключен</p>
          <div class="mask-adjust-grid">
            <div class="option-row">
              <span>Голова и мимика</span>
              <label class="switch">
                <input id="avatarFaceToggle" type="checkbox" checked />
                <span></span>
              </label>
            </div>
            <div class="option-row">
              <span>Маска поверх модели</span>
              <label class="switch">
                <input id="avatarFaceOverlayToggle" type="checkbox" checked />
                <span></span>
              </label>
            </div>
            <div class="option-row">
              <span>Руки</span>
              <label class="switch">
                <input id="avatarHandsToggle" type="checkbox" checked />
                <span></span>
              </label>
            </div>
            <div class="option-row">
              <span>Торс</span>
              <label class="switch">
                <input id="avatarTorsoToggle" type="checkbox" checked />
                <span></span>
              </label>
            </div>
            <div class="mask-stability">
              <div class="mask-stability-head">
                <label class="input-label" for="avatarSmoothing">Сглаживание</label>
                <strong id="avatarSmoothingValue">35%</strong>
              </div>
              <input id="avatarSmoothing" class="range-input" type="range" min="0" max="180" step="1" value="35" />
              <div class="range-captions">
                <span>Быстрее</span>
                <span>Плавнее</span>
              </div>
            </div>
            <div class="mask-stability">
              <div class="mask-stability-head">
                <label class="input-label" for="avatarSmileSensitivity">Улыбка</label>
                <strong id="avatarSmileSensitivityValue">100%</strong>
              </div>
              <input id="avatarSmileSensitivity" class="range-input" type="range" min="0" max="160" step="5" value="100" />
              <div class="range-captions">
                <span>Спокойнее</span>
                <span>Ярче</span>
              </div>
            </div>
            <div class="mask-stability">
              <div class="mask-stability-head">
                <label class="input-label" for="avatarScale">Масштаб</label>
                <strong id="avatarScaleValue">100%</strong>
              </div>
            <input id="avatarScale" class="range-input" type="range" min="70" max="220" step="1" value="100" />
            </div>
            <div class="mask-stability">
              <div class="mask-stability-head">
                <label class="input-label" for="avatarHeight">Высота</label>
                <strong id="avatarHeightValue">0</strong>
              </div>
              <input id="avatarHeight" class="range-input" type="range" min="-600" max="600" step="1" value="0" />
            </div>
            <div class="mask-stability">
              <div class="mask-stability-head">
                <label class="input-label" for="avatarHeadRollOffset">Наклон головы</label>
                <strong id="avatarHeadRollOffsetValue">0°</strong>
              </div>
              <input id="avatarHeadRollOffset" class="range-input" type="range" min="-30" max="30" step="1" value="0" />
              <div class="range-captions">
                <span>Влево</span>
                <span>Вправо</span>
              </div>
            </div>
            <div class="mask-stability">
              <div class="mask-stability-head">
                <label class="input-label" for="avatarHeadPitchOffset">Сдвиг вверх/вниз</label>
                <strong id="avatarHeadPitchOffsetValue">0°</strong>
              </div>
              <input id="avatarHeadPitchOffset" class="range-input" type="range" min="-30" max="30" step="1" value="0" />
              <div class="range-captions">
                <span>Вниз</span>
                <span>Вверх</span>
              </div>
            </div>
            <div class="mask-stability">
              <div class="mask-stability-head">
                <label class="input-label" for="avatarHeadPitchScale">Чувствит. вверх/вниз</label>
                <strong id="avatarHeadPitchScaleValue">100%</strong>
              </div>
              <input id="avatarHeadPitchScale" class="range-input" type="range" min="50" max="250" step="5" value="100" />
              <div class="range-captions">
                <span>Меньше</span>
                <span>Больше</span>
              </div>
            </div>
          </div>
        </section>

        <section class="panel face-panel">
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

        <section class="panel mask-panel">
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
          <div class="mask-preset-field" aria-label="Готовые маски">
            <label class="input-label" for="maskPresetSelect">Список масок</label>
            <select class="select-input" id="maskPresetSelect">
              <option value="">Выбрать маску</option>
              <option value="uploaded">Загруженная</option>
              ${maskPresets.map((preset) => `
                <option value="${preset.id}">${preset.label}</option>
              `).join('')}
            </select>
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
          <div class="mask-stability" id="maskHandOcclusionBlock">
            <div class="mask-stability-head">
              <div class="setting-title">
                <label class="setting-toggle" title="Показывать руку поверх маски">
                  <input id="maskHandOcclusionEnabled" type="checkbox" checked />
                  <span></span>
                </label>
                <label class="input-label" for="maskHandOcclusionPadding">Рука поверх</label>
              </div>
              <strong id="maskHandOcclusionPaddingValue">24px</strong>
            </div>
            <input id="maskHandOcclusionPadding" class="range-input" type="range" min="0" max="80" step="1" value="24" />
            <div class="range-captions">
              <span>Точнее</span>
              <span>Шире</span>
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

    <dialog class="gesture-dialog" id="gestureDialog">
      <form method="dialog" class="gesture-dialog-card">
        <div class="gesture-dialog-head">
          <div>
            <span class="label">Жест</span>
            <h2 id="gestureDialogTitle">Настройки жеста</h2>
          </div>
          <button class="icon-button" id="gestureDialogClose" type="submit" aria-label="Закрыть">×</button>
        </div>
        <label class="dialog-option-row">
          <span>
            <strong>Трекать жест</strong>
            <em id="gestureDialogSignal">Источник</em>
          </span>
          <input id="gestureDialogEnabled" type="checkbox" />
        </label>
        <label class="dialog-field">
          <span class="input-label">Действие</span>
          <select class="select-input" id="gestureDialogAction">
            <option value="preset">Действие режима</option>
            <option value="webhook">Webhook событие</option>
            <option value="log">Только лог</option>
            <option value="none">Без действия</option>
          </select>
        </label>
        <p class="panel-state" id="gestureDialogActionHint">Список действий будем расширять.</p>
      </form>
    </dialog>
  </div>
`

const appShell = getElement<HTMLDivElement>('appShell')
const stage = getElement<HTMLDivElement>('stage')
const video = getElement<HTMLVideoElement>('cameraVideo')
const avatarCanvas = getElement<HTMLCanvasElement>('avatarCanvas')
const canvas = getElement<HTMLCanvasElement>('overlayCanvas')
const canvasContext = canvas.getContext('2d')
const cursorDot = getElement<HTMLDivElement>('cursorDot')
const hud = getElement<HTMLDivElement>('hud')
const stageEmpty = getElement<HTMLDivElement>('stageEmpty')
const stageEmptyText = getElement<HTMLSpanElement>('stageEmptyText')
const cameraButton = getElement<HTMLButtonElement>('cameraButton')
const mirrorButton = getElement<HTMLButtonElement>('mirrorButton')
const settingsButton = getElement<HTMLButtonElement>('settingsButton')
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
const avatarToggle = getElement<HTMLInputElement>('avatarToggle')
const avatarFile = getElement<HTMLInputElement>('avatarFile')
const avatarSampleButton = getElement<HTMLButtonElement>('avatarSampleButton')
const avatarModelButtons = Array.from(document.querySelectorAll<HTMLButtonElement>('[data-avatar-model]'))
const avatarBackgroundFile = getElement<HTMLInputElement>('avatarBackgroundFile')
const avatarBackgroundClearButton = getElement<HTMLButtonElement>('avatarBackgroundClearButton')
const avatarState = getElement<HTMLElement>('avatarState')
const avatarFaceToggle = getElement<HTMLInputElement>('avatarFaceToggle')
const avatarFaceOverlayToggle = getElement<HTMLInputElement>('avatarFaceOverlayToggle')
const avatarHandsToggle = getElement<HTMLInputElement>('avatarHandsToggle')
const avatarTorsoToggle = getElement<HTMLInputElement>('avatarTorsoToggle')
const avatarSmoothingSlider = getElement<HTMLInputElement>('avatarSmoothing')
const avatarSmoothingValue = getElement<HTMLElement>('avatarSmoothingValue')
const avatarSmileSensitivitySlider = getElement<HTMLInputElement>('avatarSmileSensitivity')
const avatarSmileSensitivityValue = getElement<HTMLElement>('avatarSmileSensitivityValue')
const avatarScaleSlider = getElement<HTMLInputElement>('avatarScale')
const avatarScaleValue = getElement<HTMLElement>('avatarScaleValue')
const avatarHeightSlider = getElement<HTMLInputElement>('avatarHeight')
const avatarHeightValue = getElement<HTMLElement>('avatarHeightValue')
const avatarHeadRollOffsetSlider = getElement<HTMLInputElement>('avatarHeadRollOffset')
const avatarHeadRollOffsetValue = getElement<HTMLElement>('avatarHeadRollOffsetValue')
const avatarHeadPitchOffsetSlider = getElement<HTMLInputElement>('avatarHeadPitchOffset')
const avatarHeadPitchOffsetValue = getElement<HTMLElement>('avatarHeadPitchOffsetValue')
const avatarHeadPitchScaleSlider = getElement<HTMLInputElement>('avatarHeadPitchScale')
const avatarHeadPitchScaleValue = getElement<HTMLElement>('avatarHeadPitchScaleValue')
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
const maskPresetSelect = getElement<HTMLSelectElement>('maskPresetSelect')
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
const maskHandOcclusionBlock = getElement<HTMLDivElement>('maskHandOcclusionBlock')
const maskHandOcclusionEnabledToggle = getElement<HTMLInputElement>('maskHandOcclusionEnabled')
const maskHandOcclusionPaddingSlider = getElement<HTMLInputElement>('maskHandOcclusionPadding')
const maskHandOcclusionPaddingValue = getElement<HTMLElement>('maskHandOcclusionPaddingValue')
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
const gestureDialog = getElement<HTMLDialogElement>('gestureDialog')
const gestureDialogTitle = getElement<HTMLElement>('gestureDialogTitle')
const gestureDialogSignal = getElement<HTMLElement>('gestureDialogSignal')
const gestureDialogEnabled = getElement<HTMLInputElement>('gestureDialogEnabled')
const gestureDialogAction = getElement<HTMLSelectElement>('gestureDialogAction')
const gestureDialogActionHint = getElement<HTMLElement>('gestureDialogActionHint')
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
let poseLandmarker: PoseLandmarker | null = null
let visionFileset: VisionFileset | null = null
let modelBootPromise: Promise<void> | null = null
let stream: MediaStream | null = null
let isRunning = false
let lastVideoTime = -1
let currentPreset: PresetId = readPreset()
let mirrorMode = localStorage.getItem('xedoc-hands-mirror') !== 'off'
let settingsVisible = localStorage.getItem('xedoc-hands-settings-visible') !== 'off'
let enabledGestures = readEnabledGestures()
let handTrackingEnabled = localStorage.getItem('xedoc-hands-hand-tracking') !== 'off'
let handMarkersEnabled = localStorage.getItem('xedoc-hands-hand-markers') !== 'off'
let faceTrackingEnabled = localStorage.getItem('xedoc-hands-face-tracking') !== 'off'
let multiFaceTrackingEnabled = localStorage.getItem('xedoc-hands-multi-face') === 'on'
let hudVisible = localStorage.getItem('xedoc-hands-hud') !== 'off'
let performanceMode: PerformanceMode = readPerformanceMode()
let avatarEnabled = localStorage.getItem('xedoc-hands-avatar-enabled') === 'true'
let avatarFaceEnabled = localStorage.getItem('xedoc-hands-avatar-face') !== 'off'
let avatarFaceOverlayEnabled = localStorage.getItem('xedoc-hands-avatar-face-overlay') !== 'off'
let avatarHandsEnabled = localStorage.getItem('xedoc-hands-avatar-hands') !== 'off'
let avatarTorsoEnabled = localStorage.getItem('xedoc-hands-avatar-torso') !== 'off'
let avatarSmoothing = readAvatarSmoothing()
let avatarSmileSensitivity = readAvatarSmileSensitivity()
let avatarScale = readAvatarScale()
let avatarHeight = readAvatarHeight()
let avatarHeadRollOffset = readAvatarHeadRollOffset()
let avatarHeadPitchOffset = readAvatarHeadPitchOffset()
let avatarHeadPitchScale = readAvatarHeadPitchScale()
let avatarBackgroundImage = readAvatarBackgroundImage()
let maskEnabled = localStorage.getItem('xedoc-hands-mask-enabled') === 'true'
let maskMode: MaskMode = readMaskMode()
let maskStability = readMaskStability()
let maskEdgeFeather = readMaskEdgeFeather()
let maskEdgeFeatherEnabled = readMaskSettingEnabled('edge-feather')
let maskHandOcclusionEnabled = readMaskSettingEnabled('hand-occlusion')
let maskHandOcclusionPadding = readMaskHandOcclusionPadding()
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
let activeMaskPresetId: string | null = null
let uploadedMaskOption: {
  layer: FaceMaskLayer | null
  imageUrl: string
  sourceBlob: Blob
} | null = null
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
let currentHandLandmarks: NormalizedLandmark[][] = []
let avatarRig: AvatarRig | null = null
let avatarModelUrl: string | null = null
let avatarLoading = false
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
setupCollapsiblePanels()
setMirrorMode(mirrorMode)
setSettingsVisible(settingsVisible)
setHandTrackingEnabled(handTrackingEnabled)
setHandMarkersEnabled(handMarkersEnabled)
setFaceTrackingEnabled(faceTrackingEnabled)
setMultiFaceTrackingEnabled(multiFaceTrackingEnabled, false)
setHudVisible(hudVisible)
setPerformanceMode(performanceMode)
setAvatarFaceEnabled(avatarFaceEnabled)
setAvatarHandsEnabled(avatarHandsEnabled)
setAvatarTorsoEnabled(avatarTorsoEnabled)
setAvatarSmoothing(avatarSmoothing)
setAvatarSmileSensitivity(avatarSmileSensitivity)
setAvatarScale(avatarScale)
setAvatarHeight(avatarHeight)
setAvatarHeadRollOffset(avatarHeadRollOffset)
setAvatarHeadPitchOffset(avatarHeadPitchOffset)
setAvatarHeadPitchScale(avatarHeadPitchScale)
setAvatarBackgroundImage(avatarBackgroundImage)
setAvatarEnabled(avatarEnabled)
setMaskMode(maskMode)
setAvatarFaceOverlayEnabled(avatarFaceOverlayEnabled)
setMaskStability(maskStability)
setMaskEdgeFeather(maskEdgeFeather)
setMaskEdgeFeatherEnabled(maskEdgeFeatherEnabled)
setMaskHandOcclusionEnabled(maskHandOcclusionEnabled)
setMaskHandOcclusionPadding(maskHandOcclusionPadding)
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

settingsButton.addEventListener('click', () => {
  setSettingsVisible(!settingsVisible)
  trackToggle('settings_toggled', settingsVisible)
})

maskToggle.addEventListener('change', () => {
  if (maskToggle.checked && !faceTrackingEnabled) {
    setFaceTrackingEnabled(true)
    trackToggle('face_tracking_toggled', true, {
      source: 'mask_enabled',
    })
  }

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

avatarToggle.addEventListener('change', () => {
  setAvatarEnabled(avatarToggle.checked)
  trackToggle('avatar_toggled', avatarEnabled, {
    face: avatarFaceEnabled,
    hands: avatarHandsEnabled,
    torso: avatarTorsoEnabled,
  })

  if (avatarEnabled) {
    void ensureAvatarModel()
  }
})

avatarFaceToggle.addEventListener('change', () => {
  setAvatarFaceEnabled(avatarFaceToggle.checked)
  trackToggle('avatar_face_toggled', avatarFaceEnabled)
})

avatarFaceOverlayToggle.addEventListener('change', () => {
  setAvatarFaceOverlayEnabled(avatarFaceOverlayToggle.checked)
  trackToggle('avatar_face_overlay_toggled', avatarFaceOverlayEnabled)
})

avatarHandsToggle.addEventListener('change', () => {
  setAvatarHandsEnabled(avatarHandsToggle.checked)
  trackToggle('avatar_hands_toggled', avatarHandsEnabled)
})

avatarTorsoToggle.addEventListener('change', () => {
  setAvatarTorsoEnabled(avatarTorsoToggle.checked)
  trackToggle('avatar_torso_toggled', avatarTorsoEnabled)
})

avatarSmoothingSlider.addEventListener('input', () => {
  setAvatarSmoothing(Number(avatarSmoothingSlider.value))
})

avatarSmoothingSlider.addEventListener('change', () => {
  trackMetrika('avatar_setting_changed', { setting: 'smoothing', value: avatarSmoothing })
})

avatarSmileSensitivitySlider.addEventListener('input', () => {
  setAvatarSmileSensitivity(Number(avatarSmileSensitivitySlider.value))
})

avatarSmileSensitivitySlider.addEventListener('change', () => {
  trackMetrika('avatar_setting_changed', { setting: 'smileSensitivity', value: avatarSmileSensitivity })
})

avatarScaleSlider.addEventListener('input', () => {
  setAvatarScale(Number(avatarScaleSlider.value))
})

avatarScaleSlider.addEventListener('change', () => {
  trackMetrika('avatar_setting_changed', { setting: 'scale', value: avatarScale })
})

avatarHeightSlider.addEventListener('input', () => {
  setAvatarHeight(Number(avatarHeightSlider.value))
})

avatarHeightSlider.addEventListener('change', () => {
  trackMetrika('avatar_setting_changed', { setting: 'height', value: avatarHeight })
})

avatarHeadRollOffsetSlider.addEventListener('input', () => {
  setAvatarHeadRollOffset(Number(avatarHeadRollOffsetSlider.value))
})

avatarHeadRollOffsetSlider.addEventListener('change', () => {
  trackMetrika('avatar_setting_changed', { setting: 'headRollOffset', value: avatarHeadRollOffset })
})

avatarHeadPitchOffsetSlider.addEventListener('input', () => {
  setAvatarHeadPitchOffset(Number(avatarHeadPitchOffsetSlider.value))
})

avatarHeadPitchOffsetSlider.addEventListener('change', () => {
  trackMetrika('avatar_setting_changed', { setting: 'headPitchOffset', value: avatarHeadPitchOffset })
})

avatarHeadPitchScaleSlider.addEventListener('input', () => {
  setAvatarHeadPitchScale(Number(avatarHeadPitchScaleSlider.value))
})

avatarHeadPitchScaleSlider.addEventListener('change', () => {
  trackMetrika('avatar_setting_changed', { setting: 'headPitchScale', value: avatarHeadPitchScale })
})

avatarSampleButton.addEventListener('click', () => {
  void loadAvatarModel(defaultAvatarModelUrl, 'Тестовая VRM')
})

for (const button of avatarModelButtons) {
  button.addEventListener('click', () => {
    const preset = avatarModelPresets.find((model) => model.id === button.dataset.avatarModel)

    if (!preset) {
      return
    }

    void loadAvatarModel(preset.url, preset.name)
  })
}

avatarFile.addEventListener('change', () => {
  const file = avatarFile.files?.[0]

  if (!file) {
    return
  }

  if (avatarModelUrl?.startsWith('blob:')) {
    URL.revokeObjectURL(avatarModelUrl)
  }

  const objectUrl = URL.createObjectURL(file)
  void loadAvatarModel(objectUrl, file.name)
  avatarFile.value = ''
})

avatarBackgroundFile.addEventListener('change', () => {
  const file = avatarBackgroundFile.files?.[0]

  if (file) {
    void loadAvatarBackgroundFile(file)
  }

  avatarBackgroundFile.value = ''
})

avatarBackgroundClearButton.addEventListener('click', () => {
  setAvatarBackgroundImage(null)
  trackMetrika('avatar_background_changed', {
    action: 'clear',
  })
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

maskPresetSelect.addEventListener('change', () => {
  const presetId = maskPresetSelect.value

  if (!presetId) {
    return
  }

  if (presetId === 'uploaded') {
    applyUploadedMaskOption()
    return
  }

  const preset = maskPresets.find((item) => item.id === presetId)

  if (preset) {
    loadMaskPreset(preset)
  }
})

gestureDialogEnabled.addEventListener('change', () => {
  const gestureValue = gestureDialog.dataset.gesture ?? ''

  if (isGestureKey(gestureValue)) {
    setGestureEnabled(gestureValue, gestureDialogEnabled.checked)
  }
})

gestureDialogAction.addEventListener('change', () => {
  const gestureValue = gestureDialog.dataset.gesture ?? ''

  if (!isGestureKey(gestureValue)) {
    return
  }

  trackMetrika('gesture_action_selected', {
    gesture: gestureValue,
    gestureTitle: gestureTitleFor(gestureValue),
    actionMode: gestureDialogAction.value,
  })
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

maskHandOcclusionEnabledToggle.addEventListener('change', () => {
  setMaskHandOcclusionEnabled(maskHandOcclusionEnabledToggle.checked)
  trackMaskSettingToggle('handOcclusion', maskHandOcclusionEnabled)
})

maskHandOcclusionPaddingSlider.addEventListener('input', () => {
  setMaskHandOcclusionPadding(Number(maskHandOcclusionPaddingSlider.value))
})

maskHandOcclusionPaddingSlider.addEventListener('change', () => {
  trackMaskSetting('handOcclusionPadding', maskHandOcclusionPadding)
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

  if (uploadedMaskOption?.imageUrl && uploadedMaskOption.imageUrl !== maskImageUrl) {
    URL.revokeObjectURL(uploadedMaskOption.imageUrl)
  }

  extraMaskSlots.forEach(revokeMaskSlot)
  resetFaceSwapFrame()

  if (avatarModelUrl?.startsWith('blob:')) {
    URL.revokeObjectURL(avatarModelUrl)
  }
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

    const [handTask, faceTask, maskFaceTask, poseTask] = await Promise.all([
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
      createPoseLandmarker(vision),
    ])

    recognizer = handTask
    faceLandmarker = faceTask
    maskFaceLandmarker = maskFaceTask
    poseLandmarker = poseTask

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

async function createPoseLandmarker(vision: VisionFileset) {
  return PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: poseModelPath,
      delegate: 'GPU',
    },
    runningMode: 'VIDEO',
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
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
  currentHandLandmarks = []
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
    const poseResult = avatarEnabled && (avatarTorsoEnabled || avatarHandsEnabled) && poseLandmarker
      ? poseLandmarker.detectForVideo(video, now)
      : null
    processResult(handResult, faceResult, poseResult, now)
    updateFps(now)
  }

  requestAnimationFrame(predictFrame)
}

function processResult(
  result: GestureRecognizerResult | null,
  faceResult: FaceLandmarkerResult | null,
  poseResult: PoseLandmarkerResult | null,
  now: number,
) {
  context.clearRect(0, 0, canvas.width, canvas.height)
  updateFaceSwapBridge(now)
  if (shouldDrawFaceOverlayOverAvatar()) {
    drawFaceSwapFrame()
  }

  const detected = new Set<GestureKey>()
  const handLandmarks = handTrackingEnabled ? (result?.landmarks ?? []) : []
  currentHandLandmarks = handLandmarks
  const topGesture = handTrackingEnabled ? result?.gestures[0]?.[0] : undefined
  const topScore = topGesture?.score ?? 0
  const modelGesture = topGesture?.categoryName ?? 'None'
  const landmarks = handLandmarks[0]

  updateAvatar(faceResult, handLandmarks, poseResult, now)

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

  if (handTrackingEnabled && isGestureKey(modelGesture) && topScore > 0.58 && isGestureEnabled(modelGesture)) {
    addDetectedGesture(detected, modelGesture)
    fireGesture(modelGesture, topScore, 'model')
  }

  let pinchStrength = 0
  let motionStrength = 0

  if (handTrackingEnabled && landmarks) {
    const pinch = updatePinch(landmarks, now)
    pinchStrength = pinch.strength

    if (pinch.active) {
      addDetectedGesture(detected, 'Pinch')
    }

    if (pinch.holdActive) {
      addDetectedGesture(detected, 'Pinch_Hold')
    }

    updateShapeGestures(landmarks, pinch.ratio, detected)
    motionStrength = updateMotion(landmarks, now, detected)
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
  if (!hasAnyGestureEnabled(shapeGestureKeys)) {
    return
  }

  const hand = analyzeHand(landmarks)
  const pinchConfidence = clamp(1 - pinchRatio, 0, 1)
  const directionY = hand.averageLongTipY - landmarks[0].y
  const directionConfidence = clamp(Math.abs(directionY) / (hand.palmWidth * 1.8), 0, 1)

  if (pinchRatio < 0.47 && hand.middle && hand.ring && hand.pinky) {
    if (isGestureEnabled('OK_Gesture')) {
      addDetectedGesture(detected, 'OK_Gesture')
      fireGesture('OK_Gesture', pinchConfidence, 'landmarks', {
        ratio: round(pinchRatio),
        longFingers: hand.longCount,
      })
      return
    }
  }

  if (hand.thumb && hand.index && !hand.middle && !hand.ring && !hand.pinky) {
    if (isGestureEnabled('Finger_Gun')) {
      addDetectedGesture(detected, 'Finger_Gun')
      fireGesture('Finger_Gun', 0.86, 'landmarks', { longFingers: hand.longCount })
      return
    }
  }

  if (hand.index && hand.middle && hand.ring && !hand.pinky) {
    addDetectedGesture(detected, 'Three_Fingers')
    fireGesture('Three_Fingers', 0.82, 'landmarks', { longFingers: hand.longCount })
  } else if (hand.longCount === 4 && !hand.thumb) {
    addDetectedGesture(detected, 'Four_Fingers')
    fireGesture('Four_Fingers', 0.82, 'landmarks', { longFingers: hand.longCount })
  }

  if (hand.longCount >= 4 && directionY < -hand.palmWidth * 0.7) {
    addDetectedGesture(detected, 'Palm_Up')
    fireGesture('Palm_Up', directionConfidence, 'landmarks', { directionY: round(directionY) })
  }

  if (hand.longCount >= 4 && directionY > hand.palmWidth * 0.7) {
    addDetectedGesture(detected, 'Palm_Down')
    fireGesture('Palm_Down', directionConfidence, 'landmarks', { directionY: round(directionY) })
  }
}

function updateMotion(landmarks: NormalizedLandmark[], now: number, detected: Set<GestureKey>) {
  if (!hasAnyGestureEnabled(swipeGestureKeys)) {
    motionSamples = []
    return 0
  }

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
    addDetectedGesture(detected, gesture)
    fireGesture(gesture, strength, 'motion', { dx: round(dx), dy: round(dy) })
    motionSamples = []
  }

  if (Math.abs(dy) > 0.2 && Math.abs(dy) > Math.abs(dx) * 1.45) {
    const gesture: GestureKey = dy < 0 ? 'Swipe_Up' : 'Swipe_Down'
    addDetectedGesture(detected, gesture)
    fireGesture(gesture, strength, 'motion', { dx: round(dx), dy: round(dy) })
    motionSamples = []
  }

  return strength
}

function updateZoom(hands: NormalizedLandmark[][], now: number, detected: Set<GestureKey>) {
  if (!hasAnyGestureEnabled(zoomGestureKeys) || hands.length < 2) {
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
    addDetectedGesture(detected, gesture)
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

  if (!shouldDrawFaceOverlayOverAvatar()) {
    rememberMaskFace(face, now)
  } else if (maskEnabled && maskMode === 'mesh' && hasMeshMaskLayer()) {
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

function updateAvatar(
  faceResult: FaceLandmarkerResult | null,
  handLandmarks: NormalizedLandmark[][],
  poseResult: PoseLandmarkerResult | null,
  now: number,
) {
  if (!avatarEnabled) {
    return
  }

  const rig = ensureAvatarRig()
  syncAvatarRendererSize()

  if (!rig.currentObject && !avatarLoading) {
    void ensureAvatarModel()
  }

  const face = avatarFaceEnabled ? faceResult?.faceLandmarks?.[0] : undefined
  const pose = poseResult?.landmarks?.[0]
  applyAvatarFace(rig, faceResult, face)
  applyAvatarTorso(rig, avatarTorsoEnabled ? pose : undefined)
  applyAvatarHands(rig, handLandmarks, pose)
  renderAvatarScene(now)
}

function ensureAvatarRig() {
  if (avatarRig) {
    return avatarRig
  }

  const scene = new THREE.Scene()
  const camera = new THREE.PerspectiveCamera(30, 16 / 9, 0.1, 100)
  camera.position.set(0, 1.12, 2.45)
  camera.lookAt(0, 1.04, 0)

  const renderer = new THREE.WebGLRenderer({
    canvas: avatarCanvas,
    alpha: true,
    antialias: performanceMode !== 'performance',
    powerPreference: 'high-performance',
  })
  renderer.setClearColor(0x000000, 0)
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, performanceMode === 'performance' ? 1.25 : 2))

  const lightRig = new THREE.Group()
  const keyLight = new THREE.DirectionalLight(0xffffff, 2.2)
  keyLight.position.set(1.2, 2.2, 2.4)
  const fillLight = new THREE.DirectionalLight(0x93c5fd, 0.9)
  fillLight.position.set(-2.2, 1.2, 1.8)
  const ambientLight = new THREE.AmbientLight(0xffffff, 1.15)
  lightRig.add(keyLight, fillLight, ambientLight)
  scene.add(lightRig)

  const fallbackData = createFallbackAvatar()
  scene.add(fallbackData.group)

  const loader = new GLTFLoader()
  loader.register((parser) => new VRMLoaderPlugin(parser))

  avatarRig = {
    scene,
    camera,
    renderer,
    loader,
    lightRig,
    fallback: fallbackData.group,
    fallbackBones: fallbackData.bones,
    currentVrm: null,
    currentObject: fallbackData.group,
    expressionState: {
      aa: 0,
      happy: 0,
      blink: 0,
      blinkLeft: 0,
      blinkRight: 0,
    },
    lastRenderedAt: performance.now(),
  }

  syncAvatarRendererSize()
  prepareAvatarObjectFrame(fallbackData.group, true)
  applyAvatarTransform()
  renderAvatarScene(performance.now())
  return avatarRig
}

function createFallbackAvatar() {
  const group = new THREE.Group()
  group.name = 'FallbackAvatar'

  const bones: Record<string, THREE.Object3D> = {}
  const material = new THREE.MeshStandardMaterial({ color: 0x8dd8c2, roughness: 0.68, metalness: 0.08 })
  const darkMaterial = new THREE.MeshStandardMaterial({ color: 0x17201d, roughness: 0.72 })
  const accentMaterial = new THREE.MeshStandardMaterial({ color: 0x2f6ed3, roughness: 0.55 })

  const hips = new THREE.Group()
  const chest = new THREE.Group()
  const neck = new THREE.Group()
  const head = new THREE.Group()
  hips.position.set(0, 0.35, 0)
  chest.position.set(0, 0.58, 0)
  neck.position.set(0, 0.46, 0)
  head.position.set(0, 0.24, 0)
  group.add(hips)
  hips.add(chest)
  chest.add(neck)
  neck.add(head)

  const torso = new THREE.Mesh(new THREE.CapsuleGeometry(0.42, 0.72, 8, 18), material)
  torso.rotation.z = Math.PI
  torso.position.y = 0.1
  chest.add(torso)

  const headMesh = new THREE.Mesh(new THREE.SphereGeometry(0.25, 32, 24), material)
  headMesh.scale.set(0.9, 1.08, 0.92)
  head.add(headMesh)

  const leftEye = new THREE.Mesh(new THREE.SphereGeometry(0.032, 16, 10), darkMaterial)
  const rightEye = leftEye.clone()
  leftEye.position.set(-0.08, 0.04, 0.23)
  rightEye.position.set(0.08, 0.04, 0.23)
  head.add(leftEye, rightEye)

  const mouth = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.018, 0.012), darkMaterial)
  mouth.position.set(0, -0.105, 0.235)
  head.add(mouth)

  const createArm = (side: -1 | 1) => {
    const shoulder = new THREE.Group()
    const upper = new THREE.Group()
    const lower = new THREE.Group()
    const hand = new THREE.Group()
    shoulder.position.set(side * 0.42, 0.28, 0)
    upper.position.set(0, 0, 0)
    lower.position.set(0, -0.34, 0)
    hand.position.set(0, -0.32, 0)
    shoulder.add(upper)
    upper.add(lower)
    lower.add(hand)

    const upperMesh = new THREE.Mesh(new THREE.CapsuleGeometry(0.065, 0.28, 6, 12), accentMaterial)
    upperMesh.position.y = -0.18
    const lowerMesh = new THREE.Mesh(new THREE.CapsuleGeometry(0.052, 0.26, 6, 12), accentMaterial)
    lowerMesh.position.y = -0.16
    const handMesh = new THREE.Mesh(new THREE.SphereGeometry(0.07, 16, 12), material)
    upper.add(upperMesh)
    lower.add(lowerMesh)
    hand.add(handMesh)
    chest.add(shoulder)

    return { shoulder, upper, lower, hand }
  }

  const leftArm = createArm(-1)
  const rightArm = createArm(1)

  bones.hips = hips
  bones.chest = chest
  bones.neck = neck
  bones.head = head
  bones.leftUpperArm = leftArm.upper
  bones.leftLowerArm = leftArm.lower
  bones.leftHand = leftArm.hand
  bones.rightUpperArm = rightArm.upper
  bones.rightLowerArm = rightArm.lower
  bones.rightHand = rightArm.hand
  bones.mouth = mouth

  group.position.set(0, -0.35, 0)
  return { group, bones }
}

async function ensureAvatarModel() {
  if (avatarRig?.currentVrm || avatarLoading) {
    return
  }

  await loadAvatarModel(defaultAvatarModelUrl, 'Тестовая VRM')
}

async function loadAvatarModel(url: string, label: string) {
  const rig = ensureAvatarRig()
  avatarLoading = true
  avatarState.textContent = `Загрузка: ${label}`

  try {
    const gltf = await rig.loader.loadAsync(url)
    const vrm = gltf.userData.vrm as VRM | undefined
    const object = vrm?.scene ?? gltf.scene

    if (rig.currentObject && rig.currentObject !== rig.fallback) {
      rig.scene.remove(rig.currentObject)
    }

    rig.fallback.visible = false
    rig.currentVrm = vrm ?? null
    rig.currentObject = object
    object.position.set(0, 0, 0)
    object.rotation.set(0, 0, 0)
    rig.scene.add(object)
    prepareAvatarObjectFrame(object, object === rig.fallback)
    avatarModelUrl = url
    updateAvatarModelButtons(url)
    applyAvatarTransform()
    avatarState.textContent = vrm ? `VRM: ${label}` : `Модель: ${label}`
    trackMetrika('avatar_model_changed', {
      type: vrm ? 'vrm' : 'gltf',
      source: avatarModelPresets.some((preset) => preset.url === url) ? 'preset' : 'upload',
    })
  } catch (error) {
    console.error(error)
    rig.fallback.visible = true
    rig.currentVrm = null
    rig.currentObject = rig.fallback
    updateAvatarModelButtons(null)
    avatarState.textContent = 'Fallback-аватар'
  } finally {
    avatarLoading = false
    renderAvatarScene(performance.now())
  }
}

function applyAvatarFace(rig: AvatarRig, faceResult: FaceLandmarkerResult | null, face?: NormalizedLandmark[]) {
  const idle = performance.now() * 0.001
  const pose = face ? getHeadPose(face) : null
  const yaw = pose ? pose.yaw : Math.sin(idle * 0.7) * 0.03
  const pitchOffset = (avatarHeadPitchOffset * Math.PI) / 180
  const pitch = (pose ? pose.pitch : Math.sin(idle * 0.9) * 0.025) * (avatarHeadPitchScale / 100) + pitchOffset
  const rollOffset = (-avatarHeadRollOffset * Math.PI) / 180
  const roll = (face ? -getFaceRoll(face) : Math.sin(idle * 0.6) * 0.02) + rollOffset
  const response = getAvatarResponse()

  rotateAvatarBone(rig, VRMHumanBoneName.Head, 'head', {
    x: clamp(pitch * 0.85, -0.48, 0.48),
    y: clamp(yaw * 1.05, -0.7, 0.7),
    z: clamp(roll * 0.55, -0.32, 0.32),
  }, response)
  rotateAvatarBone(rig, VRMHumanBoneName.Neck, 'neck', {
    x: clamp(pitch * 0.24, -0.18, 0.18),
    y: clamp(yaw * 0.28, -0.24, 0.24),
    z: clamp(roll * 0.18, -0.16, 0.16),
  }, response)

  const jawOpen = getBlendshapeScore(faceResult, 'jawOpen')
  const rawSmile = (getBlendshapeScore(faceResult, 'mouthSmileLeft') + getBlendshapeScore(faceResult, 'mouthSmileRight')) / 2
  const smile = clamp((rawSmile - 0.08) * (avatarSmileSensitivity / 100), 0, 1)
  const blinkLeft = normalizeAvatarBlink(getBlendshapeScore(faceResult, 'eyeBlinkLeft'))
  const blinkRight = normalizeAvatarBlink(getBlendshapeScore(faceResult, 'eyeBlinkRight'))
  const blink = clamp(Math.max(blinkLeft, blinkRight) * 1.08, 0, 1)
  const aa = smoothAvatarExpression(rig, 'aa', clamp(jawOpen * 1.25, 0, 1), response)
  const happy = smoothAvatarExpression(rig, 'happy', clamp(smile * 1.2, 0, 1), response * 0.75)
  const smoothBlink = smoothAvatarBlink(rig, 'blink', blink)
  const smoothBlinkLeft = smoothAvatarBlink(rig, 'blinkLeft', blinkLeft)
  const smoothBlinkRight = smoothAvatarBlink(rig, 'blinkRight', blinkRight)
  rig.currentVrm?.expressionManager?.setValue('aa', aa)
  rig.currentVrm?.expressionManager?.setValue('happy', happy)
  rig.currentVrm?.expressionManager?.setValue('blink', smoothBlink)
  rig.currentVrm?.expressionManager?.setValue('blinkLeft', smoothBlinkLeft)
  rig.currentVrm?.expressionManager?.setValue('blinkRight', smoothBlinkRight)

  const fallbackMouth = rig.fallbackBones.mouth
  if (fallbackMouth) {
    fallbackMouth.scale.y = lerp(fallbackMouth.scale.y, 1 + jawOpen * 5.5 + smile * 1.4, response)
    fallbackMouth.scale.x = lerp(fallbackMouth.scale.x, 1 + smile * 1.9, response)
  }
}

function applyAvatarTorso(rig: AvatarRig, pose?: NormalizedLandmark[]) {
  const response = getAvatarResponse()
  const leftShoulder = pose?.[11]
  const rightShoulder = pose?.[12]
  const leftHip = pose?.[23]
  const rightHip = pose?.[24]

  if (!leftShoulder || !rightShoulder) {
    rotateAvatarBone(rig, VRMHumanBoneName.Chest, 'chest', { x: 0, y: 0, z: 0 }, response * 0.35)
    return
  }

  const shoulderCenter = midpoint(leftShoulder, rightShoulder)
  const hipCenter = leftHip && rightHip ? midpoint(leftHip, rightHip) : { ...shoulderCenter, y: shoulderCenter.y + 0.34 }
  const shoulderSlope = (rightShoulder.y - leftShoulder.y) * (mirrorMode ? -1 : 1)
  const torsoLean = (shoulderCenter.x - hipCenter.x) * (mirrorMode ? -1 : 1)
  const shoulderDepth = (rightShoulder.z ?? 0) - (leftShoulder.z ?? 0)

  rotateAvatarBone(rig, VRMHumanBoneName.Chest, 'chest', {
    x: clamp((shoulderCenter.y - hipCenter.y + 0.32) * 1.2, -0.22, 0.26),
    y: clamp(shoulderDepth * 1.3, -0.34, 0.34),
    z: clamp(torsoLean * 1.3 + shoulderSlope * 1.8, -0.34, 0.34),
  }, response)
  rotateAvatarBone(rig, VRMHumanBoneName.Spine, 'hips', {
    x: 0,
    y: clamp(shoulderDepth * 0.45, -0.16, 0.16),
    z: clamp(torsoLean * 0.45, -0.16, 0.16),
  }, response)
}

function applyAvatarHands(rig: AvatarRig, hands: NormalizedLandmark[][], pose?: NormalizedLandmark[]) {
  if (!avatarHandsEnabled) {
    applyAvatarRestArms(rig, getAvatarResponse() * 0.45)
    applyAvatarRestFingers(rig, 'left', getAvatarResponse() * 0.35)
    applyAvatarRestFingers(rig, 'right', getAvatarResponse() * 0.35)
    return
  }

  const response = getAvatarResponse()
  let leftApplied = false
  let rightApplied = false
  let leftFingersApplied = false
  let rightFingersApplied = false

  if (hands.length > 0) {
    for (const hand of hands.slice(0, 2)) {
      const displayCenter = getAvatarHandDisplayCenter(hand)
      const side = getAvatarSideForDisplayX(displayCenter.x)
      const isLeft = side === 'left'

      applyAvatarArmFromHandLandmarks(rig, side, hand, pose, response)

      if (isLeft) {
        leftApplied = true
      } else {
        rightApplied = true
      }

      applyAvatarHandLandmarks(rig, side, hand, response)
      if (isLeft) {
        leftFingersApplied = true
      } else {
        rightFingersApplied = true
      }
    }
  }

  if (!leftApplied) {
    leftApplied = applyAvatarArmFromPose(rig, 'left', pose?.[11], pose?.[13], pose?.[15], response)
  }

  if (!rightApplied) {
    rightApplied = applyAvatarArmFromPose(rig, 'right', pose?.[12], pose?.[14], pose?.[16], response)
  }

  if (!leftApplied) {
    applyAvatarRestArm(rig, 'left', response * 0.22)
  }

  if (!rightApplied) {
    applyAvatarRestArm(rig, 'right', response * 0.22)
  }

  if (!leftFingersApplied) {
    applyAvatarRestFingers(rig, 'left', response * 0.28)
  }

  if (!rightFingersApplied) {
    applyAvatarRestFingers(rig, 'right', response * 0.28)
  }
}

function applyAvatarArmFromPose(
  rig: AvatarRig,
  side: AvatarSide,
  shoulder: NormalizedLandmark | undefined,
  elbow: NormalizedLandmark | undefined,
  wrist: NormalizedLandmark | undefined,
  response: number,
) {
  if (!isReliablePoseChain(shoulder, elbow, wrist)) {
    return false
  }

  const safeShoulder = shoulder!
  const safeElbow = elbow!
  const safeWrist = wrist!
  const sideSign = side === 'left' ? -1 : 1
  const mirrorSign = mirrorMode ? -1 : 1
  const heightOffset = getAvatarHandHeightOffset()
  const upperDx = (safeElbow.x - safeShoulder.x) * mirrorSign
  const upperDy = safeElbow.y - safeShoulder.y + heightOffset
  const lowerDx = (safeWrist.x - safeElbow.x) * mirrorSign
  const lowerDy = safeWrist.y - safeElbow.y
  const wristDepth = safeWrist.z ?? 0
  const upperName = side === 'left' ? VRMHumanBoneName.LeftUpperArm : VRMHumanBoneName.RightUpperArm
  const lowerName = side === 'left' ? VRMHumanBoneName.LeftLowerArm : VRMHumanBoneName.RightLowerArm
  const handName = side === 'left' ? VRMHumanBoneName.LeftHand : VRMHumanBoneName.RightHand

  rotateAvatarBone(rig, upperName, `${side}UpperArm`, {
    x: clamp(-upperDy * 1.45 + 0.24, -0.85, 0.95),
    y: clamp(sideSign * wristDepth * 0.85, -0.38, 0.38),
    z: clamp(sideSign * (0.62 + upperDx * 2.35), -1.05, 1.05),
  }, response)
  rotateAvatarBone(rig, lowerName, `${side}LowerArm`, {
    x: clamp(-lowerDy * 1.55 + 0.08, -0.9, 0.9),
    y: 0,
    z: clamp(sideSign * lowerDx * 2.2, -0.85, 0.85),
  }, response)
  rotateAvatarBone(rig, handName, `${side}Hand`, {
    x: clamp(-lowerDy * 0.9, -0.48, 0.48),
    y: clamp(sideSign * lowerDx * 0.8, -0.42, 0.42),
    z: clamp(sideSign * lowerDx * 0.9, -0.45, 0.45),
  }, response)

  return true
}

function applyAvatarArmFromHand(
  rig: AvatarRig,
  side: AvatarSide,
  center: NormalizedLandmark,
  response: number,
) {
  const sideSign = side === 'left' ? -1 : 1
  const displayX = toAvatarDisplayX(center.x)
  const x = displayX - 0.5
  const y = center.y - 0.5 + getAvatarHandHeightOffset()
  const raise = clamp((0.56 - center.y) * 2.4, 0, 1.15)
  const lateral = clamp(Math.abs(x) * 2.2, 0, 1.1)
  const closeToTorso = 1 - clamp(lateral / 1.05, 0, 1)
  const elbowBend = clamp(0.18 + raise * 0.38 + closeToTorso * 0.58, 0.12, 1.0)
  const shoulderOut = clamp(0.6 - raise * 1.08 + lateral * 0.12, -0.55, 0.72)
  const upperName = side === 'left' ? VRMHumanBoneName.LeftUpperArm : VRMHumanBoneName.RightUpperArm
  const lowerName = side === 'left' ? VRMHumanBoneName.LeftLowerArm : VRMHumanBoneName.RightLowerArm

  rotateAvatarBone(rig, upperName, `${side}UpperArm`, {
    x: clamp(0.3 - raise * 0.2 - y * 0.12, -0.28, 0.62),
    y: clamp(sideSign * x * 0.24, -0.42, 0.42),
    z: sideSign * shoulderOut,
  }, response)
  rotateAvatarBone(rig, lowerName, `${side}LowerArm`, {
    x: elbowBend,
    y: sideSign * closeToTorso * 0.16,
    z: clamp(sideSign * x * 0.38, -0.42, 0.42),
  }, response)
}

function getAvatarHandDisplayCenter(landmarks: NormalizedLandmark[]) {
  const center = handCenter(landmarks)

  return {
    ...center,
    x: toAvatarDisplayX(center.x),
  }
}

function getAvatarSideForDisplayX(x: number): AvatarSide {
  const visibleSide: AvatarSide = x < 0.5 ? 'left' : 'right'

  if (mirrorMode) {
    return visibleSide
  }

  return visibleSide === 'left' ? 'right' : 'left'
}

function getAvatarHandTrackingFrame(pose: NormalizedLandmark[] | undefined) {
  const leftShoulder = pose?.[11]
  const rightShoulder = pose?.[12]

  if (isReliablePoseChain(leftShoulder, rightShoulder)) {
    const leftX = toAvatarDisplayX(leftShoulder!.x)
    const rightX = toAvatarDisplayX(rightShoulder!.x)
    const width = clamp(Math.abs(rightX - leftX) * 1.55, 0.22, 0.58)

    return {
      x: (leftX + rightX) / 2,
      y: (leftShoulder!.y + rightShoulder!.y) / 2 + 0.04,
      width,
    }
  }

  return {
    x: 0.5,
    y: 0.56,
    width: 0.34,
  }
}

function toAvatarDisplayX(x: number) {
  return mirrorMode ? 1 - x : x
}

function applyAvatarArmFromHandLandmarks(
  rig: AvatarRig,
  side: AvatarSide,
  landmarks: NormalizedLandmark[],
  pose: NormalizedLandmark[] | undefined,
  response: number,
) {
  const wrist = landmarks[0]
  const indexBase = landmarks[5]
  const middleBase = landmarks[9]
  const ringBase = landmarks[13]
  const littleBase = landmarks[17]
  const middleTip = landmarks[12]

  if (!wrist || !indexBase || !middleBase || !ringBase || !littleBase || !middleTip) {
    applyAvatarArmFromHand(rig, side, handCenter(landmarks), response * 0.7)
    return
  }

  const sideSign = side === 'left' ? -1 : 1
  const frame = getAvatarHandTrackingFrame(pose)
  const palmCenter = {
    x: (wrist.x + indexBase.x + middleBase.x + ringBase.x + littleBase.x) / 5,
    y: (wrist.y + indexBase.y + middleBase.y + ringBase.y + littleBase.y) / 5,
    z: ((wrist.z ?? 0) + (indexBase.z ?? 0) + (middleBase.z ?? 0) + (ringBase.z ?? 0) + (littleBase.z ?? 0)) / 5,
    visibility: Math.min(
      wrist.visibility ?? 1,
      indexBase.visibility ?? 1,
      middleBase.visibility ?? 1,
      ringBase.visibility ?? 1,
      littleBase.visibility ?? 1,
    ),
  }
  const displayPalmX = toAvatarDisplayX(palmCenter.x)
  const displayWristX = toAvatarDisplayX(wrist.x)
  const displayMiddleBaseX = toAvatarDisplayX(middleBase.x)
  const displayMiddleTipX = toAvatarDisplayX(middleTip.x)
  const poseArm = getAvatarPoseArmForHand(pose, displayPalmX)
  const relativeX = clamp((displayPalmX - frame.x) / frame.width, -1.9, 1.9)
  const relativeY = clamp((frame.y - palmCenter.y - getAvatarHandHeightOffset() * 0.42) / frame.width, -1.1, 2.3)
  const screenSideSign = displayPalmX < frame.x ? -1 : 1
  const palmVectorX = displayMiddleBaseX - displayWristX
  const palmVectorY = middleBase.y - wrist.y
  const fingerVectorX = displayMiddleTipX - displayMiddleBaseX
  const fingerVectorY = middleTip.y - middleBase.y
  const palmAngle = Math.atan2(palmVectorY, palmVectorX)
  const fingerAngle = Math.atan2(fingerVectorY, fingerVectorX)
  const verticalReach = clamp(relativeY, -0.18, 1.35)
  const sideReach = clamp(Math.abs(relativeX), 0, 1.28)
  const raise = clamp((verticalReach + 0.08) / 1.18, 0, 1.18)
  const closeToTorso = 1 - clamp(sideReach / 1.05, 0, 1)
  const poseElbowBend = poseArm ? getAvatarPoseElbowBend(poseArm.shoulder, poseArm.elbow, poseArm.wrist) : 0
  const elbowBend = clamp(0.18 + raise * 0.28 + closeToTorso * 0.34 + poseElbowBend * 0.72, 0.12, 1.18)
  const shoulderOut = clamp(0.62 - raise * 1.16 + sideReach * 0.14, -0.62, 0.74)
  const forwardDepth = clamp(-palmCenter.z * 1.25, -0.38, 0.52)
  const upperName = side === 'left' ? VRMHumanBoneName.LeftUpperArm : VRMHumanBoneName.RightUpperArm
  const lowerName = side === 'left' ? VRMHumanBoneName.LeftLowerArm : VRMHumanBoneName.RightLowerArm
  const handName = side === 'left' ? VRMHumanBoneName.LeftHand : VRMHumanBoneName.RightHand

  rotateAvatarBone(rig, upperName, `${side}UpperArm`, {
    x: clamp(0.3 - raise * 0.22 + forwardDepth * 0.28, -0.34, 0.66),
    y: clamp(sideSign * (forwardDepth * 0.72 + relativeX * 0.18), -0.64, 0.64),
    z: sideSign * shoulderOut,
  }, response)
  rotateAvatarBone(rig, lowerName, `${side}LowerArm`, {
    x: elbowBend,
    y: clamp(sideSign * (forwardDepth * 0.48 + closeToTorso * 0.18), -0.48, 0.48),
    z: clamp(sideSign * (screenSideSign * sideSign * sideReach * 0.18 + relativeX * 0.28), -0.5, 0.5),
  }, response)
  rotateAvatarBone(rig, handName, `${side}Hand`, {
    x: clamp(0.02 - palmVectorY * 1.65 + raise * 0.18, -0.62, 0.78),
    y: clamp(sideSign * forwardDepth * 0.95, -0.62, 0.62),
    z: clamp(sideSign * normalizeAngle(fingerAngle - palmAngle) * 0.58, -0.68, 0.68),
  }, response)
  applyAvatarScreenHandCorrection(rig, side, palmCenter, response)
}

function applyAvatarScreenHandCorrection(
  rig: AvatarRig,
  side: AvatarSide,
  target: NormalizedLandmark,
  response: number,
) {
  const handName = side === 'left' ? VRMHumanBoneName.LeftHand : VRMHumanBoneName.RightHand
  const upperName = side === 'left' ? VRMHumanBoneName.LeftUpperArm : VRMHumanBoneName.RightUpperArm
  const lowerName = side === 'left' ? VRMHumanBoneName.LeftLowerArm : VRMHumanBoneName.RightLowerArm
  const hand = rig.currentVrm?.humanoid.getNormalizedBoneNode(handName) ?? rig.fallbackBones[`${side}Hand`]
  const upper = rig.currentVrm?.humanoid.getNormalizedBoneNode(upperName) ?? rig.fallbackBones[`${side}UpperArm`]
  const lower = rig.currentVrm?.humanoid.getNormalizedBoneNode(lowerName) ?? rig.fallbackBones[`${side}LowerArm`]

  if (!hand || !upper || !lower) {
    return
  }

  hand.updateWorldMatrix(true, false)
  const position = new THREE.Vector3()
  hand.getWorldPosition(position)
  position.project(rig.camera)

  if (!Number.isFinite(position.x) || !Number.isFinite(position.y)) {
    return
  }

  const renderedX = (position.x + 1) / 2
  const renderedY = (1 - position.y) / 2
  const targetX = target.x
  const targetY = target.y
  const dx = clamp(targetX - renderedX, -0.42, 0.42)
  const dy = clamp(targetY - renderedY, -0.42, 0.42)
  const sideSign = side === 'left' ? -1 : 1
  const amount = clamp(response * 0.58, 0.04, 0.42)
  const targetUpperZ = clamp(upper.rotation.z + sideSign * dx * 1.35 - dy * 0.72, -1.55, 1.55)
  const targetUpperX = clamp(upper.rotation.x - dy * 0.95, -1.35, 0.9)
  const targetLowerX = clamp(lower.rotation.x + Math.abs(dx) * 0.28 - dy * 0.72, -0.24, 1.38)

  upper.rotation.z = lerp(upper.rotation.z, targetUpperZ, amount)
  upper.rotation.x = lerp(upper.rotation.x, targetUpperX, amount)
  lower.rotation.x = lerp(lower.rotation.x, targetLowerX, amount)
}

function getAvatarPoseArmForHand(pose: NormalizedLandmark[] | undefined, displayPalmX: number) {
  const candidates = [
    {
      shoulder: pose?.[11],
      elbow: pose?.[13],
      wrist: pose?.[15],
    },
    {
      shoulder: pose?.[12],
      elbow: pose?.[14],
      wrist: pose?.[16],
    },
  ].filter((candidate) => isReliablePoseChain(candidate.shoulder, candidate.elbow, candidate.wrist))

  if (candidates.length === 0) {
    return null
  }

  return candidates.reduce((best, candidate) => {
    const bestDistance = Math.abs(toAvatarDisplayX(best.wrist!.x) - displayPalmX)
    const candidateDistance = Math.abs(toAvatarDisplayX(candidate.wrist!.x) - displayPalmX)
    return candidateDistance < bestDistance ? candidate : best
  })
}

function getAvatarPoseElbowBend(
  shoulder: NormalizedLandmark | undefined,
  elbow: NormalizedLandmark | undefined,
  wrist: NormalizedLandmark | undefined,
) {
  if (!shoulder || !elbow || !wrist) {
    return 0
  }

  return clamp((Math.PI - landmarkAngle(shoulder, elbow, wrist)) / 1.15, 0, 1)
}

function applyAvatarHandLandmarks(
  rig: AvatarRig,
  side: AvatarSide,
  landmarks: NormalizedLandmark[],
  response: number,
) {
  if (!rig.currentVrm || landmarks.length < 21) {
    return
  }

  applyAvatarPalmPose(rig, side, landmarks, response)
  applyAvatarFingerCurl(rig, side, landmarks, 'thumb', [1, 2, 3, 4], response)
  applyAvatarFingerCurl(rig, side, landmarks, 'index', [5, 6, 7, 8], response)
  applyAvatarFingerCurl(rig, side, landmarks, 'middle', [9, 10, 11, 12], response)
  applyAvatarFingerCurl(rig, side, landmarks, 'ring', [13, 14, 15, 16], response)
  applyAvatarFingerCurl(rig, side, landmarks, 'little', [17, 18, 19, 20], response)
}

function applyAvatarPalmPose(
  rig: AvatarRig,
  side: AvatarSide,
  landmarks: NormalizedLandmark[],
  response: number,
) {
  const wrist = landmarks[0]
  const indexBase = landmarks[5]
  const middleBase = landmarks[9]
  const littleBase = landmarks[17]

  if (!wrist || !indexBase || !middleBase || !littleBase) {
    return
  }

  const sideSign = side === 'left' ? -1 : 1
  const palmLift = middleBase.y - wrist.y
  const palmTwist = (middleBase.z ?? 0) - (wrist.z ?? 0)
  const palmSpread = toAvatarDisplayX(indexBase.x) - toAvatarDisplayX(littleBase.x)
  const handName = side === 'left' ? VRMHumanBoneName.LeftHand : VRMHumanBoneName.RightHand

  rotateAvatarBone(rig, handName, `${side}Hand`, {
    x: clamp(-palmLift * 1.35, -0.62, 0.62),
    y: clamp(sideSign * palmTwist * 2.4, -0.58, 0.58),
    z: clamp(sideSign * palmSpread * 1.55, -0.62, 0.62),
  }, response * 0.72)
}

function applyAvatarFingerCurl(
  rig: AvatarRig,
  side: AvatarSide,
  landmarks: NormalizedLandmark[],
  finger: 'thumb' | 'index' | 'middle' | 'ring' | 'little',
  indices: [number, number, number, number],
  response: number,
) {
  const [baseIndex, proximalIndex, intermediateIndex, tipIndex] = indices
  const base = landmarks[baseIndex]
  const proximal = landmarks[proximalIndex]
  const intermediate = landmarks[intermediateIndex]
  const tip = landmarks[tipIndex]

  if (!base || !proximal || !intermediate || !tip) {
    return
  }

  const proximalCurl = getFingerCurl(base, proximal, intermediate)
  const distalCurl = getFingerCurl(proximal, intermediate, tip)
  const tipDistance = distance(tip, landmarks[0])
  const baseDistance = distance(base, landmarks[0])
  const reachCurl = clamp(1 - tipDistance / Math.max(baseDistance * 2.6, 0.001), 0, 1)
  const curl = clamp(proximalCurl * 0.55 + distalCurl * 0.35 + reachCurl * 0.28, 0, 1)
  const spread = getFingerSpread(landmarks, finger, indices)
  const bones = getAvatarFingerBones(side, finger)
  const sideSign = side === 'left' ? -1 : 1
  const thumbSign = finger === 'thumb' ? -sideSign : sideSign
  const curlSign = finger === 'thumb' ? 0.78 : 1
  const baseRotation = {
    x: clamp(curl * 1.05 * curlSign, 0, finger === 'thumb' ? 0.82 : 1.12),
    y: clamp(spread * thumbSign, -0.38, 0.38),
    z: finger === 'thumb' ? clamp(sideSign * (0.34 - curl * 0.42), -0.42, 0.42) : 0,
  }
  const midRotation = {
    x: clamp(curl * 1.18, 0, finger === 'thumb' ? 0.86 : 1.22),
    y: 0,
    z: 0,
  }
  const tipRotation = {
    x: clamp(distalCurl * 0.92 + curl * 0.28, 0, finger === 'thumb' ? 0.78 : 1.02),
    y: 0,
    z: 0,
  }

  if (bones[0]) {
    rotateAvatarBone(rig, bones[0], '', baseRotation, response * 0.82)
  }

  if (bones[1]) {
    rotateAvatarBone(rig, bones[1], '', midRotation, response * 0.82)
  }

  if (bones[2]) {
    rotateAvatarBone(rig, bones[2], '', tipRotation, response * 0.82)
  }
}

function getAvatarFingerBones(
  side: AvatarSide,
  finger: 'thumb' | 'index' | 'middle' | 'ring' | 'little',
): [VRMHumanBoneName, VRMHumanBoneName, VRMHumanBoneName | null] {
  const left = side === 'left'

  if (finger === 'thumb') {
    return left
      ? [VRMHumanBoneName.LeftThumbMetacarpal, VRMHumanBoneName.LeftThumbProximal, VRMHumanBoneName.LeftThumbDistal]
      : [VRMHumanBoneName.RightThumbMetacarpal, VRMHumanBoneName.RightThumbProximal, VRMHumanBoneName.RightThumbDistal]
  }

  const map = {
    index: left
      ? [VRMHumanBoneName.LeftIndexProximal, VRMHumanBoneName.LeftIndexIntermediate, VRMHumanBoneName.LeftIndexDistal]
      : [VRMHumanBoneName.RightIndexProximal, VRMHumanBoneName.RightIndexIntermediate, VRMHumanBoneName.RightIndexDistal],
    middle: left
      ? [VRMHumanBoneName.LeftMiddleProximal, VRMHumanBoneName.LeftMiddleIntermediate, VRMHumanBoneName.LeftMiddleDistal]
      : [VRMHumanBoneName.RightMiddleProximal, VRMHumanBoneName.RightMiddleIntermediate, VRMHumanBoneName.RightMiddleDistal],
    ring: left
      ? [VRMHumanBoneName.LeftRingProximal, VRMHumanBoneName.LeftRingIntermediate, VRMHumanBoneName.LeftRingDistal]
      : [VRMHumanBoneName.RightRingProximal, VRMHumanBoneName.RightRingIntermediate, VRMHumanBoneName.RightRingDistal],
    little: left
      ? [VRMHumanBoneName.LeftLittleProximal, VRMHumanBoneName.LeftLittleIntermediate, VRMHumanBoneName.LeftLittleDistal]
      : [VRMHumanBoneName.RightLittleProximal, VRMHumanBoneName.RightLittleIntermediate, VRMHumanBoneName.RightLittleDistal],
  } satisfies Record<typeof finger, [VRMHumanBoneName, VRMHumanBoneName, VRMHumanBoneName]>

  return map[finger]
}

function getFingerCurl(a: NormalizedLandmark, b: NormalizedLandmark, c: NormalizedLandmark) {
  const angle = landmarkAngle(a, b, c)
  return clamp((Math.PI - angle) / 1.35, 0, 1)
}

function getFingerSpread(
  landmarks: NormalizedLandmark[],
  finger: 'thumb' | 'index' | 'middle' | 'ring' | 'little',
  indices: [number, number, number, number],
) {
  const wrist = landmarks[0]
  const middleBase = landmarks[9]
  const base = landmarks[indices[0]]
  const proximal = landmarks[indices[1]]

  if (!wrist || !middleBase || !base || !proximal) {
    return 0
  }

  const palmAngle = Math.atan2(middleBase.y - wrist.y, middleBase.x - wrist.x)
  const fingerAngle = Math.atan2(proximal.y - base.y, proximal.x - base.x)
  const raw = normalizeAngle(fingerAngle - palmAngle)
  const weight = finger === 'thumb' ? 0.42 : 0.22

  return clamp(raw * weight, -0.42, 0.42)
}

function landmarkAngle(a: NormalizedLandmark, b: NormalizedLandmark, c: NormalizedLandmark) {
  const ab = {
    x: a.x - b.x,
    y: a.y - b.y,
    z: ((a.z ?? 0) - (b.z ?? 0)) * 0.65,
  }
  const cb = {
    x: c.x - b.x,
    y: c.y - b.y,
    z: ((c.z ?? 0) - (b.z ?? 0)) * 0.65,
  }
  const abLength = Math.hypot(ab.x, ab.y, ab.z)
  const cbLength = Math.hypot(cb.x, cb.y, cb.z)

  if (abLength <= 0.00001 || cbLength <= 0.00001) {
    return Math.PI
  }

  const dot = (ab.x * cb.x + ab.y * cb.y + ab.z * cb.z) / (abLength * cbLength)
  return Math.acos(clamp(dot, -1, 1))
}

function normalizeAngle(value: number) {
  let normalized = value

  while (normalized > Math.PI) {
    normalized -= Math.PI * 2
  }

  while (normalized < -Math.PI) {
    normalized += Math.PI * 2
  }

  return normalized
}

function applyAvatarRestArms(rig: AvatarRig, response: number) {
  applyAvatarRestArm(rig, 'left', response)
  applyAvatarRestArm(rig, 'right', response)
  applyAvatarRestFingers(rig, 'left', response)
  applyAvatarRestFingers(rig, 'right', response)
}

function applyAvatarRestArm(rig: AvatarRig, side: AvatarSide, response: number) {
  const sideSign = side === 'left' ? -1 : 1
  const upperName = side === 'left' ? VRMHumanBoneName.LeftUpperArm : VRMHumanBoneName.RightUpperArm
  const lowerName = side === 'left' ? VRMHumanBoneName.LeftLowerArm : VRMHumanBoneName.RightLowerArm
  const handName = side === 'left' ? VRMHumanBoneName.LeftHand : VRMHumanBoneName.RightHand

  rotateAvatarBone(rig, upperName, `${side}UpperArm`, {
    x: 0.34,
    y: 0,
    z: sideSign * 0.58,
  }, response)
  rotateAvatarBone(rig, lowerName, `${side}LowerArm`, {
    x: 0.22,
    y: 0,
    z: sideSign * 0.22,
  }, response)
  rotateAvatarBone(rig, handName, `${side}Hand`, {
    x: 0,
    y: 0,
    z: sideSign * 0.08,
  }, response)
}

function applyAvatarRestFingers(rig: AvatarRig, side: AvatarSide, response: number) {
  if (!rig.currentVrm) {
    return
  }

  for (const finger of ['thumb', 'index', 'middle', 'ring', 'little'] as const) {
    for (const bone of getAvatarFingerBones(side, finger)) {
      if (bone) {
        rotateAvatarBone(rig, bone, '', { x: 0, y: 0, z: 0 }, response)
      }
    }
  }
}

function rotateAvatarBone(
  rig: AvatarRig,
  vrmBone: VRMHumanBoneName,
  fallbackBone: string,
  rotation: { x: number; y: number; z: number },
  response: number,
) {
  const target = rig.currentVrm?.humanoid.getNormalizedBoneNode(vrmBone) ?? rig.fallbackBones[fallbackBone]

  if (!target) {
    return
  }

  target.rotation.x = lerp(target.rotation.x, rotation.x, response)
  target.rotation.y = lerp(target.rotation.y, rotation.y, response)
  target.rotation.z = lerp(target.rotation.z, rotation.z, response)
}

function renderAvatarScene(now: number) {
  const rig = avatarRig

  if (!rig || !avatarEnabled) {
    return
  }

  const delta = clamp((now - rig.lastRenderedAt) / 1000, 0.001, 0.08)
  rig.lastRenderedAt = now
  rig.currentVrm?.update(delta)
  rig.renderer.render(rig.scene, rig.camera)
}

function syncAvatarRendererSize() {
  const rig = avatarRig

  if (!rig) {
    return
  }

  const width = canvas.width || video.videoWidth || 1280
  const height = canvas.height || video.videoHeight || 720

  if (avatarCanvas.width !== width || avatarCanvas.height !== height) {
    rig.renderer.setSize(width, height, false)
    rig.camera.aspect = width / Math.max(height, 1)
    rig.camera.updateProjectionMatrix()
  }
}

function applyAvatarTransform() {
  const rig = avatarRig
  const object = rig?.currentObject

  if (!rig || !object) {
    return
  }

  const baseScale = typeof object.userData.avatarBaseScale === 'number' ? object.userData.avatarBaseScale : 1
  const baseY = typeof object.userData.avatarBaseY === 'number' ? object.userData.avatarBaseY : 0
  const scale = (avatarScale / 100) * baseScale
  object.scale.setScalar(scale)
  object.position.y = baseY + avatarHeight / 100
  renderAvatarScene(performance.now())
}

async function loadAvatarBackgroundFile(file: File) {
  if (!file.type.startsWith('image/')) {
    avatarState.textContent = 'Фон не изображение'
    return
  }

  try {
    avatarState.textContent = 'Загружаем фон'
    const dataUrl = await resizeImageFileToDataUrl(file, 1920, 0.86)
    setAvatarBackgroundImage(dataUrl)
    avatarState.textContent = 'Фон обновлен'
    trackMetrika('avatar_background_changed', {
      action: 'upload',
      size: file.size,
    })
  } catch (error) {
    console.error(error)
    avatarState.textContent = 'Фон не загрузился'
  }
}

async function resizeImageFileToDataUrl(file: File, maxSize: number, quality: number) {
  const image = await loadImageFromFile(file)
  const scale = Math.min(1, maxSize / Math.max(image.naturalWidth, image.naturalHeight, 1))
  const width = Math.max(1, Math.round(image.naturalWidth * scale))
  const height = Math.max(1, Math.round(image.naturalHeight * scale))
  const targetCanvas = document.createElement('canvas')
  const targetContext = getCanvasContext(targetCanvas)
  targetCanvas.width = width
  targetCanvas.height = height
  targetContext.drawImage(image, 0, 0, width, height)
  URL.revokeObjectURL(image.src)
  return targetCanvas.toDataURL('image/jpeg', quality)
}

function loadImageFromFile(file: File) {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image()
    const objectUrl = URL.createObjectURL(file)
    image.onload = () => resolve(image)
    image.onerror = () => {
      URL.revokeObjectURL(objectUrl)
      reject(new Error('Image failed to load'))
    }
    image.src = objectUrl
  })
}

function prepareAvatarObjectFrame(object: THREE.Object3D, fallback = false) {
  object.updateMatrixWorld(true)
  const box = new THREE.Box3().setFromObject(object)
  const size = new THREE.Vector3()
  const center = new THREE.Vector3()
  box.getSize(size)
  box.getCenter(center)
  const height = Math.max(size.y, 0.001)
  const desiredHeight = fallback ? 2.15 : 2.65
  const baseScale = desiredHeight / height
  object.userData.avatarBaseScale = baseScale
  object.userData.avatarBaseY = 0.56 - center.y * baseScale
}

function getAvatarResponse() {
  if (avatarSmoothing <= 90) {
    return 1 - avatarSmoothing / 120
  }

  return 0.25 - ((avatarSmoothing - 90) / 90) * 0.17
}

function getAvatarHandHeightOffset() {
  return clamp(avatarHeight / 1800, -0.34, 0.34)
}

function normalizeAvatarBlink(score: number) {
  const normalized = clamp((score - 0.1) / 0.5, 0, 1)
  const eased = normalized * normalized * (3 - normalized * 2)

  return eased < 0.04 ? 0 : eased
}

function smoothAvatarExpression(
  rig: AvatarRig,
  key: keyof AvatarRig['expressionState'],
  next: number,
  response: number,
) {
  const current = rig.expressionState[key]
  const filtered = lerp(current, next, clamp(response, 0.05, 0.8))
  rig.expressionState[key] = filtered
  return filtered
}

function smoothAvatarBlink(rig: AvatarRig, key: 'blink' | 'blinkLeft' | 'blinkRight', next: number) {
  const current = rig.expressionState[key]
  const movement = Math.abs(next - current)
  const baseResponse = getAvatarResponse()
  const response = clamp(baseResponse * 0.38 + movement * 0.52 + (next > current ? 0.1 : 0), 0.06, 0.82)
  const filtered = movement < 0.025 ? current : lerp(current, next, response)
  const settled = filtered < 0.035 ? 0 : filtered

  rig.expressionState[key] = settled
  return settled
}

function getBlendshapeScore(result: FaceLandmarkerResult | null, name: string) {
  return result?.faceBlendshapes?.[0]?.categories.find((category) => category.categoryName === name)?.score ?? 0
}

function isReliablePoseChain(...landmarks: Array<NormalizedLandmark | undefined>) {
  return landmarks.every((landmark) => landmark && (landmark.visibility ?? 1) >= 0.48)
}

function getFaceRoll(face: NormalizedLandmark[]) {
  const left = face[33] ?? face[263]
  const right = face[263] ?? face[33]

  if (!left || !right) {
    return 0
  }

  return Math.atan2(right.y - left.y, right.x - left.x)
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
  applyHandOcclusionToMask(landmarks)

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

function applyHandOcclusionToMask(faceLandmarks: NormalizedLandmark[]) {
  if (!maskHandOcclusionEnabled || !handTrackingEnabled || currentHandLandmarks.length === 0) {
    return
  }

  const faceBox = expandRect(getLandmarkRect(faceLandmarks), maskHandOcclusionPadding)
  maskRenderContext.save()
  maskRenderContext.globalCompositeOperation = 'destination-out'
  maskRenderContext.fillStyle = '#000'
  maskRenderContext.strokeStyle = '#000'
  maskRenderContext.lineCap = 'round'
  maskRenderContext.lineJoin = 'round'

  for (const hand of currentHandLandmarks) {
    const handPoints = hand.map(landmarkToCanvasPoint)
    const handBox = expandRect(getPointRect(handPoints), maskHandOcclusionPadding)

    if (!rectsIntersect(faceBox, handBox)) {
      continue
    }

    eraseHandFromMask(handPoints, maskHandOcclusionPadding)
  }

  maskRenderContext.restore()
}

function eraseHandFromMask(points: Point2D[], padding: number) {
  const box = getPointRect(points)
  const handScale = Math.max(box.maxX - box.minX, box.maxY - box.minY)
  const jointRadius = clamp(handScale * 0.055 + padding * 0.58, 10, 58)
  const palmRadius = clamp(jointRadius * 1.28, 14, 72)
  const strokeWidth = clamp(jointRadius * 1.55, 18, 92)
  const palmIndices = [0, 1, 5, 9, 13, 17]
  const palmHull = convexHull(palmIndices.map((index) => points[index]).filter(Boolean))

  if (palmHull.length >= 3) {
    drawPolygon(maskRenderContext, expandPolygon(palmHull, palmRadius * 0.6))
    maskRenderContext.fill()
  }

  maskRenderContext.lineWidth = strokeWidth

  for (const connection of GestureRecognizer.HAND_CONNECTIONS) {
    const start = points[connection.start]
    const end = points[connection.end]

    if (!start || !end) {
      continue
    }

    maskRenderContext.beginPath()
    maskRenderContext.moveTo(start.x, start.y)
    maskRenderContext.lineTo(end.x, end.y)
    maskRenderContext.stroke()
  }

  for (const [index, point] of points.entries()) {
    const isPalm = palmIndices.includes(index)
    maskRenderContext.beginPath()
    maskRenderContext.arc(point.x, point.y, isPalm ? palmRadius : jointRadius, 0, Math.PI * 2)
    maskRenderContext.fill()
  }
}

function drawPolygon(targetContext: CanvasRenderingContext2D, points: Point2D[]) {
  if (points.length === 0) {
    return
  }

  targetContext.beginPath()
  targetContext.moveTo(points[0].x, points[0].y)

  for (const point of points.slice(1)) {
    targetContext.lineTo(point.x, point.y)
  }

  targetContext.closePath()
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

function addDetectedGesture(detected: Set<GestureKey>, gesture: GestureKey) {
  if (isGestureEnabled(gesture)) {
    detected.add(gesture)
  }
}

function fireFaceSignal(
  gesture: GestureKey,
  confidence: number,
  detected: Set<GestureKey>,
  details?: GestureEvent['details'],
) {
  addDetectedGesture(detected, gesture)
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
  if (!isGestureEnabled(gesture)) {
    return
  }

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
  const enabledCount = gestureDefinitions.filter((gesture) => isGestureEnabled(gesture.key)).length

  gestureGrid.innerHTML = gestureDefinitions
    .map(
      (gesture) => `
        <article class="gesture-tile${isGestureEnabled(gesture.key) ? '' : ' is-disabled'}" data-gesture="${gesture.key}">
          <label class="gesture-check" title="Трекать жест">
            <input
              class="gesture-enabled"
              data-gesture-toggle="${gesture.key}"
              type="checkbox"
              ${isGestureEnabled(gesture.key) ? 'checked' : ''}
            />
            <span></span>
          </label>
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

  actionHint.textContent = `${presets[currentPreset].title}: ${enabledCount} из ${gestureDefinitions.length} команд`

  gestureGrid.querySelectorAll<HTMLElement>('.gesture-check').forEach((label) => {
    label.addEventListener('click', (event) => event.stopPropagation())
  })

  gestureGrid.querySelectorAll<HTMLInputElement>('.gesture-enabled').forEach((input) => {
    input.addEventListener('click', (event) => event.stopPropagation())
    input.addEventListener('change', () => {
      const gestureValue = input.dataset.gestureToggle ?? ''

      if (isGestureKey(gestureValue)) {
        setGestureEnabled(gestureValue, input.checked)
      }
    })
  })

  gestureGrid.querySelectorAll<HTMLElement>('.gesture-tile').forEach((tile) => {
    tile.addEventListener('click', () => {
      const gestureValue = tile.dataset.gesture ?? ''

      if (isGestureKey(gestureValue)) {
        openGestureDialog(gestureValue)
      }
    })
  })
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
    tile.classList.toggle('is-live', active.has(key) && isGestureEnabled(key))
    tile.classList.toggle('is-disabled', !isGestureEnabled(key))
  })
}

function setGestureEnabled(gesture: GestureKey, enabled: boolean) {
  if (enabled) {
    enabledGestures.add(gesture)
  } else {
    enabledGestures.delete(gesture)
    cooldowns.delete(gesture)
  }

  saveEnabledGestures()
  renderGestureGrid()
  setActiveGestures(new Set())
  refreshIcons()
  trackMetrika('gesture_tracking_toggled', {
    gesture,
    gestureTitle: gestureTitleFor(gesture),
    enabled,
    enabledCount: enabledGestures.size,
  })

  if (gestureDialog.open && gestureDialog.dataset.gesture === gesture) {
    openGestureDialog(gesture)
  }
}

function openGestureDialog(gesture: GestureKey) {
  const definition = gestureDefinitions.find((item) => item.key === gesture)

  if (!definition) {
    return
  }

  gestureDialog.dataset.gesture = gesture
  gestureDialogTitle.textContent = definition.title
  gestureDialogSignal.textContent = `${definition.signal} • ${gesture}`
  gestureDialogEnabled.checked = isGestureEnabled(gesture)
  gestureDialogAction.value = 'preset'
  gestureDialogActionHint.textContent = `Сейчас в режиме «${presets[currentPreset].title}»: ${presets[currentPreset].actions[gesture]}.`

  if (!gestureDialog.open) {
    if (typeof gestureDialog.showModal === 'function') {
      gestureDialog.showModal()
    } else {
      gestureDialog.setAttribute('open', 'true')
    }
  }

  trackMetrika('gesture_settings_opened', {
    gesture,
    gestureTitle: definition.title,
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
    void prepareMaskLayer(nextImage, nextUrl, file, faceRank, {
      source: 'upload',
      presetId: null,
    })
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

async function loadMaskPreset(preset: (typeof maskPresets)[number]) {
  setMaskLoadState(0, `Загрузка: ${preset.label}`)
  trackMetrika('mask_preset_selected', {
    preset: preset.id,
  })

  try {
    const response = await fetch(preset.url)

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`)
    }

    const blob = await response.blob()
    const imageUrl = URL.createObjectURL(blob)
    const image = new Image()

    image.addEventListener('load', () => {
      void prepareMaskLayer(image, imageUrl, blob, 0, {
        source: 'preset',
        presetId: preset.id,
      })
    })

    image.addEventListener('error', () => {
      URL.revokeObjectURL(imageUrl)
      setMaskLoadState(0, 'Не загрузилась')
      trackMetrika('mask_upload_error', {
        slot: 'main',
        reason: 'preset_image_load',
        preset: preset.id,
      })
    })

    image.src = imageUrl
  } catch (error) {
    console.error(error)
    setMaskLoadState(0, 'Не загрузилась')
    trackMetrika('mask_upload_error', {
      slot: 'main',
      reason: 'preset_fetch',
      preset: preset.id,
    })
  }
}

function applyUploadedMaskOption() {
  if (!uploadedMaskOption) {
    setMaskLoadState(0, 'Нет загруженной')
    return
  }

  if (maskImageUrl && maskImageUrl !== uploadedMaskOption.imageUrl && maskImageUrl.startsWith('blob:')) {
    URL.revokeObjectURL(maskImageUrl)
  }

  resetFaceSwapFrame()
  maskLayer = uploadedMaskOption.layer
  maskImageUrl = uploadedMaskOption.imageUrl
  maskSourceBlob = uploadedMaskOption.sourceBlob
  activeMaskPresetId = 'uploaded'
  maskPreview.src = uploadedMaskOption.imageUrl
  setMaskEnabled(true)
  updateMaskPresetButtons()
  trackMetrika('mask_preset_selected', {
    preset: 'uploaded',
  })
}

async function prepareMaskLayer(
  image: HTMLImageElement,
  imageUrl: string,
  sourceBlob: Blob,
  faceRank = 0,
  options: { source: 'upload' | 'preset'; presetId: string | null } = { source: 'upload', presetId: null },
) {
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

  if (maskImageUrl && maskImageUrl !== uploadedMaskOption?.imageUrl && maskImageUrl.startsWith('blob:')) {
    URL.revokeObjectURL(maskImageUrl)
  }

  resetFaceSwapFrame()

  maskLayer = layer
  maskImageUrl = imageUrl
  maskSourceBlob = sourceBlob
  activeMaskPresetId = options.source === 'preset' ? options.presetId : 'uploaded'

  if (options.source === 'upload') {
    if (uploadedMaskOption?.imageUrl && uploadedMaskOption.imageUrl !== imageUrl) {
      URL.revokeObjectURL(uploadedMaskOption.imageUrl)
    }

    uploadedMaskOption = {
      layer,
      imageUrl,
      sourceBlob,
    }
  }

  maskPreview.src = imageUrl
  setMaskEnabled(true)
  updateMaskPresetButtons()
  trackMetrika('mask_changed', {
    slot: analyticsSlot,
    mode: maskMode,
    width: image.naturalWidth,
    height: image.naturalHeight,
    faceDetected: Boolean(layer),
    maskCount: getLoadedMaskCount(),
    source: options.source,
    preset: options.presetId ?? '',
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

function setAvatarEnabled(next: boolean) {
  avatarEnabled = next
  localStorage.setItem('xedoc-hands-avatar-enabled', String(next))
  avatarToggle.checked = next
  stage.classList.toggle('is-avatar-mode', next)
  avatarState.textContent = next ? avatarState.textContent.replace('Выключен', 'Аватар включен') : 'Выключен'

  if (next) {
    ensureAvatarRig()
    syncAvatarRendererSize()
    renderAvatarScene(performance.now())
  }
}

function setAvatarFaceEnabled(next: boolean) {
  avatarFaceEnabled = next
  localStorage.setItem('xedoc-hands-avatar-face', next ? 'on' : 'off')
  avatarFaceToggle.checked = next
}

function setAvatarFaceOverlayEnabled(next: boolean) {
  avatarFaceOverlayEnabled = next
  localStorage.setItem('xedoc-hands-avatar-face-overlay', next ? 'on' : 'off')
  avatarFaceOverlayToggle.checked = next
}

function shouldDrawFaceOverlayOverAvatar() {
  return !avatarEnabled || avatarFaceOverlayEnabled
}

function setAvatarHandsEnabled(next: boolean) {
  avatarHandsEnabled = next
  localStorage.setItem('xedoc-hands-avatar-hands', next ? 'on' : 'off')
  avatarHandsToggle.checked = next
}

function setAvatarTorsoEnabled(next: boolean) {
  avatarTorsoEnabled = next
  localStorage.setItem('xedoc-hands-avatar-torso', next ? 'on' : 'off')
  avatarTorsoToggle.checked = next
}

function setAvatarSmoothing(next: number) {
  avatarSmoothing = clamp(Math.round(next), 0, 180)
  localStorage.setItem('xedoc-hands-avatar-smoothing', String(avatarSmoothing))
  avatarSmoothingSlider.value = String(avatarSmoothing)
  avatarSmoothingValue.textContent = `${avatarSmoothing}%`
}

function setAvatarSmileSensitivity(next: number) {
  avatarSmileSensitivity = clamp(Math.round(next), 0, 160)
  localStorage.setItem('xedoc-hands-avatar-smile-sensitivity', String(avatarSmileSensitivity))
  avatarSmileSensitivitySlider.value = String(avatarSmileSensitivity)
  avatarSmileSensitivityValue.textContent = `${avatarSmileSensitivity}%`
}

function setAvatarScale(next: number) {
  avatarScale = clamp(Math.round(next), 70, 220)
  localStorage.setItem('xedoc-hands-avatar-scale', String(avatarScale))
  avatarScaleSlider.value = String(avatarScale)
  avatarScaleValue.textContent = `${avatarScale}%`
  applyAvatarTransform()
}

function setAvatarHeight(next: number) {
  avatarHeight = clamp(Math.round(next), -600, 600)
  localStorage.setItem('xedoc-hands-avatar-height', String(avatarHeight))
  avatarHeightSlider.value = String(avatarHeight)
  avatarHeightValue.textContent = signedValue(avatarHeight)
  applyAvatarTransform()
}

function setAvatarHeadRollOffset(next: number) {
  avatarHeadRollOffset = clamp(Math.round(next), -30, 30)
  localStorage.setItem('xedoc-hands-avatar-head-roll-offset', String(avatarHeadRollOffset))
  avatarHeadRollOffsetSlider.value = String(avatarHeadRollOffset)
  avatarHeadRollOffsetValue.textContent = `${signedValue(avatarHeadRollOffset)}°`
}

function setAvatarHeadPitchOffset(next: number) {
  avatarHeadPitchOffset = clamp(Math.round(next), -30, 30)
  localStorage.setItem('xedoc-hands-avatar-head-pitch-offset', String(avatarHeadPitchOffset))
  avatarHeadPitchOffsetSlider.value = String(avatarHeadPitchOffset)
  avatarHeadPitchOffsetValue.textContent = `${signedValue(avatarHeadPitchOffset)}°`
}

function setAvatarHeadPitchScale(next: number) {
  avatarHeadPitchScale = clamp(Math.round(next), 50, 250)
  localStorage.setItem('xedoc-hands-avatar-head-pitch-scale', String(avatarHeadPitchScale))
  avatarHeadPitchScaleSlider.value = String(avatarHeadPitchScale)
  avatarHeadPitchScaleValue.textContent = `${avatarHeadPitchScale}%`
}

function updateAvatarModelButtons(url: string | null) {
  for (const button of avatarModelButtons) {
    const preset = avatarModelPresets.find((model) => model.id === button.dataset.avatarModel)
    button.classList.toggle('is-active', Boolean(preset && preset.url === url))
  }

  avatarSampleButton.classList.toggle('is-active', url === defaultAvatarModelUrl)
}

function setAvatarBackgroundImage(next: string | null) {
  avatarBackgroundImage = next
  stage.classList.toggle('has-avatar-background', Boolean(next))
  stage.style.setProperty('--avatar-background-image', next ? `url("${next}")` : 'none')
  avatarBackgroundClearButton.disabled = !next

  try {
    if (next) {
      localStorage.setItem('xedoc-hands-avatar-background', next)
    } else {
      localStorage.removeItem('xedoc-hands-avatar-background')
    }
  } catch (error) {
    console.warn('Avatar background was too large to save', error)
    localStorage.removeItem('xedoc-hands-avatar-background')
    avatarState.textContent = 'Фон показан, но не сохранен'
  }
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

function setMaskHandOcclusionEnabled(next: boolean) {
  maskHandOcclusionEnabled = next
  setMaskSettingEnabled('hand-occlusion', next)
  maskHandOcclusionEnabledToggle.checked = next
  maskHandOcclusionPaddingSlider.disabled = !next
}

function setMaskHandOcclusionPadding(next: number) {
  maskHandOcclusionPadding = clamp(Math.round(next), 0, 80)
  localStorage.setItem('xedoc-hands-mask-hand-occlusion-padding', String(maskHandOcclusionPadding))
  maskHandOcclusionPaddingSlider.value = String(maskHandOcclusionPadding)
  maskHandOcclusionPaddingValue.textContent = `${maskHandOcclusionPadding}px`
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

  updateMaskPresetButtons()
  maskToggle.disabled = !canEnable
  maskToggle.checked = canEnable && maskEnabled
  maskPreview.classList.toggle('is-visible', hasMainImage)
  maskStabilityBlock.classList.toggle('is-hidden', maskMode !== 'mesh')
  maskEdgeFeatherBlock.classList.toggle('is-hidden', maskMode !== 'mesh')
  maskHandOcclusionBlock.classList.toggle('is-hidden', maskMode !== 'mesh')
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

function updateMaskPresetButtons() {
  const uploadedOption = Array.from(maskPresetSelect.options).find((option) => option.value === 'uploaded')

  if (uploadedOption) {
    uploadedOption.disabled = !uploadedMaskOption
  }

  maskPresetSelect.value = activeMaskPresetId ?? ''
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
  currentHandLandmarks = []
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

function setSettingsVisible(next: boolean) {
  settingsVisible = next
  localStorage.setItem('xedoc-hands-settings-visible', next ? 'on' : 'off')
  appShell.classList.toggle('is-settings-hidden', !next)
  settingsButton.classList.toggle('is-active', next)
  settingsButton.setAttribute('aria-pressed', String(next))
}

function syncCanvasSize() {
  if (!video.videoWidth || !video.videoHeight) {
    return
  }

  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    stage.style.setProperty('--camera-aspect', `${video.videoWidth} / ${video.videoHeight}`)
    syncAvatarRendererSize()
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

function readAvatarSmoothing() {
  const raw = localStorage.getItem('xedoc-hands-avatar-smoothing')
  const saved = raw === null ? Number.NaN : Number(raw)
  return Number.isFinite(saved) ? clamp(Math.round(saved), 0, 180) : 35
}

function readAvatarSmileSensitivity() {
  const raw = localStorage.getItem('xedoc-hands-avatar-smile-sensitivity')
  const saved = raw === null ? Number.NaN : Number(raw)
  return Number.isFinite(saved) ? clamp(Math.round(saved), 0, 160) : 100
}

function readAvatarScale() {
  const raw = localStorage.getItem('xedoc-hands-avatar-scale')
  const saved = raw === null ? Number.NaN : Number(raw)
  return Number.isFinite(saved) ? clamp(Math.round(saved), 70, 220) : 100
}

function readAvatarHeight() {
  const raw = localStorage.getItem('xedoc-hands-avatar-height')
  const saved = raw === null ? Number.NaN : Number(raw)
  return Number.isFinite(saved) ? clamp(Math.round(saved), -600, 600) : 0
}

function readAvatarHeadRollOffset() {
  const raw = localStorage.getItem('xedoc-hands-avatar-head-roll-offset')
  const saved = raw === null ? Number.NaN : Number(raw)
  return Number.isFinite(saved) ? clamp(Math.round(saved), -30, 30) : 0
}

function readAvatarHeadPitchOffset() {
  const raw = localStorage.getItem('xedoc-hands-avatar-head-pitch-offset')
  const saved = raw === null ? Number.NaN : Number(raw)
  return Number.isFinite(saved) ? clamp(Math.round(saved), -30, 30) : 0
}

function readAvatarHeadPitchScale() {
  const raw = localStorage.getItem('xedoc-hands-avatar-head-pitch-scale')
  const saved = raw === null ? Number.NaN : Number(raw)
  return Number.isFinite(saved) ? clamp(Math.round(saved), 50, 250) : 100
}

function readAvatarBackgroundImage() {
  const saved = localStorage.getItem('xedoc-hands-avatar-background')
  return saved?.startsWith('data:image/') ? saved : null
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

function readMaskHandOcclusionPadding() {
  const raw = localStorage.getItem('xedoc-hands-mask-hand-occlusion-padding')
  const saved = raw === null ? Number.NaN : Number(raw)
  return Number.isFinite(saved) ? clamp(Math.round(saved), 0, 80) : 24
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

function setupCollapsiblePanels() {
  const collapsedPanels = readCollapsedPanels()

  document
    .querySelectorAll<HTMLHeadingElement>(
      '.control-area .panel > .panel-head > h2, .gesture-strip > .strip-head > h2, .event-log-band > .strip-head > h2',
    )
    .forEach((title) => {
      const panel = title.closest<HTMLElement>('.panel, .gesture-strip, .event-log-band')
      const key = title.textContent?.trim()

      if (!panel || !key) {
        return
      }

      title.classList.add('panel-toggle-title')
      title.tabIndex = 0
      title.setAttribute('role', 'button')
      title.setAttribute('aria-expanded', collapsedPanels.has(key) ? 'false' : 'true')
      setPanelCollapsed(panel, title, collapsedPanels.has(key))

      const toggle = () => {
        const next = !panel.classList.contains('is-collapsed')
        setPanelCollapsed(panel, title, next)

        if (next) {
          collapsedPanels.add(key)
        } else {
          collapsedPanels.delete(key)
        }

        saveCollapsedPanels(collapsedPanels)
        trackMetrika('panel_collapsed_toggled', {
          panel: key,
          collapsed: next,
        })
      }

      title.addEventListener('click', toggle)
      title.addEventListener('keydown', (event) => {
        if (event.key !== 'Enter' && event.key !== ' ') {
          return
        }

        event.preventDefault()
        toggle()
      })
    })
}

function setPanelCollapsed(panel: HTMLElement, title: HTMLElement, collapsed: boolean) {
  panel.classList.toggle('is-collapsed', collapsed)
  title.setAttribute('aria-expanded', collapsed ? 'false' : 'true')
}

function readCollapsedPanels() {
  try {
    const raw = localStorage.getItem('xedoc-hands-collapsed-panels')
    const parsed = raw ? JSON.parse(raw) : []
    return new Set<string>(Array.isArray(parsed) ? parsed.filter((value): value is string => typeof value === 'string') : [])
  } catch {
    return new Set<string>()
  }
}

function saveCollapsedPanels(panels: Set<string>) {
  localStorage.setItem('xedoc-hands-collapsed-panels', JSON.stringify([...panels]))
}

function readEnabledGestures() {
  try {
    const raw = localStorage.getItem('xedoc-hands-enabled-gestures')
    const parsed = raw ? JSON.parse(raw) : null

    if (!Array.isArray(parsed)) {
      return new Set<GestureKey>(defaultGestureKeys)
    }

    const enabled = parsed.filter((value): value is GestureKey => isGestureKey(value))
    return new Set<GestureKey>(enabled)
  } catch {
    return new Set<GestureKey>(defaultGestureKeys)
  }
}

function saveEnabledGestures() {
  localStorage.setItem('xedoc-hands-enabled-gestures', JSON.stringify([...enabledGestures]))
}

function isGestureEnabled(gesture: GestureKey) {
  return enabledGestures.has(gesture)
}

function hasAnyGestureEnabled(gestures: GestureKey[]) {
  return gestures.some((gesture) => isGestureEnabled(gesture))
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

function expandPolygon(points: Point2D[], pixels: number) {
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

function getLandmarkRect(
  landmarks: NormalizedLandmark[],
  pointMapper: (landmark: NormalizedLandmark) => Point2D = landmarkToCanvasPoint,
) {
  return getPointRect(landmarks.map(pointMapper))
}

function getPointRect(points: Point2D[]): Rect2D {
  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity

  for (const point of points) {
    minX = Math.min(minX, point.x)
    minY = Math.min(minY, point.y)
    maxX = Math.max(maxX, point.x)
    maxY = Math.max(maxY, point.y)
  }

  return { minX, minY, maxX, maxY }
}

function expandRect(rect: Rect2D, pixels: number): Rect2D {
  return {
    minX: rect.minX - pixels,
    minY: rect.minY - pixels,
    maxX: rect.maxX + pixels,
    maxY: rect.maxY + pixels,
  }
}

function rectsIntersect(first: Rect2D, second: Rect2D) {
  return first.minX <= second.maxX && first.maxX >= second.minX && first.minY <= second.maxY && first.maxY >= second.minY
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

function lerp(from: number, to: number, amount: number) {
  return from + (to - from) * clamp(amount, 0, 1)
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
