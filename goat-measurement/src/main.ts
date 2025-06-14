import './style.css'
import { DistanceProviderInput, DistanceProviderSecond, DistanceProviderStatic } from './distance-provider.ts'
import { initPWA } from './pwa.ts'
import { YOLO } from './yolotfjs.ts'
import './utils.ts'
import { AngleProviderStatic, AngleProviderSensor } from './angle-provider.ts'
import { testAll } from './testing.ts'
import { WeightPredictor } from './weight-prediction.ts'

const app = document.querySelector<HTMLDivElement>('#app')!
app.innerHTML = `
<div>
  <div id="pwa-toast">
    <div class="message">
      <div id="toast-message"></div>
    </div>
    <button id="pwa-close">Close</button>
    <button id="pwa-refresh">Reload</button>
  </div>
  <div id="app-container">
    <div id="video-container">
      <div id="toast-container"></div>
      <video id="video"></video>
      <div id="overlay-container">
        <div class="overlay-background"></div>
        <img id="overlay" src="${import.meta.env.BASE_URL}overlay2.png"/>
        <div class="overlay-background"></div>
      </div>
      <div id="result-overlay" class="hidden">
        <canvas id="result-canvas"></canvas>
        <div id="value-container">
        </div>
        <button id="result-overlay-close">X</button>
      </div>
    </div>
    <div id="right-controls">
      <button id="toggleDirection">Toggle Direction</button>
      <button id="mainButton"></button>
      <button id="imageBtn">Test</button>
      <button id="clearTest">Clear Test</button>
      <input id="calibrationValue" />
      <input id="apiPath" />
      <div id="angle"></div>
    </div>
  </div>
  <div id="test">
  </div>
</div>
`

const appContainer = document.querySelector<HTMLDivElement>("div#app-container")!
const testButton = document.querySelector<HTMLButtonElement>('#imageBtn')!
const mainButton = document.querySelector<HTMLButtonElement>("button#mainButton")!
const video = document.querySelector<HTMLVideoElement>('video#video')!
const testContainer = document.querySelector<HTMLElement>("#test")!
const resultCanvas = document.querySelector<HTMLCanvasElement>("canvas#result-canvas")!
const resultClose = document.querySelector<HTMLButtonElement>("#result-overlay-close")!
const resultOverlay = document.querySelector("#result-overlay")!
const valueContainer = resultOverlay.querySelector("#value-container")!
const clearTest = document.querySelector<HTMLButtonElement>("#clearTest")!
const overlayImage = document.querySelector<HTMLImageElement>("#overlay")!
const toastContainer = document.querySelector<HTMLDivElement>("#toast-container")!
const directionButton = document.querySelector<HTMLButtonElement>("#toggleDirection")!
const calibrationInput = document.querySelector<HTMLInputElement>("#calibrationValue")!
const angleContainer = document.querySelector<HTMLDivElement>("#angle")!
const apiPathInput = document.querySelector<HTMLInputElement>("#apiPath")!
const rightControls = document.querySelector<HTMLDivElement>("#right-controls")!

type Direction = "left" | "right"
type AppState = {
  direction: Direction
  calibration: number
  apiPath: string
  tag: string
  preferredDeviceId: string | null
  streamSettings: MediaTrackSettings | null
  capture: ImageCapture | null
  track: MediaStreamTrack | null
}

const state: AppState = {
  direction: "left",
  calibration: 3.6856,
  apiPath: "http://localhost:8080",
  tag: "3605",
  preferredDeviceId: null,
  streamSettings: null,
  capture: null,
  track: null,
}

calibrationInput.addEventListener("change", () => {
  state.calibration = parseFloat(calibrationInput.value)
})
calibrationInput.value = state.calibration.toString()
apiPathInput.addEventListener("change", () => {
  state.apiPath = apiPathInput.value
})
apiPathInput.value = state.apiPath

// Append video devices
navigator.mediaDevices.enumerateDevices().then((d) => {
  d = d.filter((d) => d.kind === "videoinput")
  const select = document.createElement("select")
  for (let device of d) {
    const option = document.createElement("option")
    option.value = device.deviceId
    option.innerText = device.label
    select.appendChild(option)
  }
  rightControls.appendChild(select)

  select.addEventListener("change", (ev) => {
    state.preferredDeviceId = ev.target.value
    setupVideo()
  })

  state.preferredDeviceId = select.value
  setupVideo()
})

appContainer.style.width = `${window.innerWidth}px`
appContainer.style.height = `${window.innerHeight}px`
window.addEventListener("resize", () => {
  appContainer.style.width = `${window.innerWidth}px`
  appContainer.style.height = `${window.innerHeight}px`
})

resultClose.addEventListener("click", () => {
  if (resultOverlay.classList.contains("hidden")) {
    showResultOverlay()
  } else {
    hideResultOverlay()
  }
})

clearTest.addEventListener("click", () => {
  testContainer.innerHTML = ""
})

directionButton.addEventListener("click", () => {
  if (state.direction == "right") {
    state.direction = "left"
    overlayImage.style.transform = ""
  } else {
    state.direction = "right"
    overlayImage.style.transform = "scaleX(-1)"
  }
})

const angleProvider = new AngleProviderSensor()
const weightPredictor = new WeightPredictor(
  new YOLO(),
  angleProvider,
  new DistanceProviderInput()
)


setInterval(async () => {
  const angle = await angleProvider.angle(video)
  angleContainer.innerText = `${angle.toFixed(2)}`
}, 100)

window.addEventListener("resize", () => {
  resizeOverlayImage(video.offsetWidth, video.offsetHeight)
})
// @ts-ignore this is not supported in all browsers
navigator.permissions.query({ name: "camera" }).then(async (perm) => {
  if (perm.state != 'denied') {
    await setupVideo()
  } else {
    toast("camera permission denied")
  }
})

async function setupVideo() {
  if (!state.preferredDeviceId) return
  if (state.track) {
    state.track.stop()
  }

  const stream = await navigator.mediaDevices.getUserMedia({ video: { deviceId: state.preferredDeviceId, facingMode: { ideal: "environment" }, width: { ideal: 640 }, height: { ideal: 640 } } })!
  console.log(stream.getVideoTracks())
  state.track = stream.getVideoTracks()[0]
  state.streamSettings = state.track.getSettings()
  state.capture = new ImageCapture(state.track)
  resizeOverlayImage(video.offsetWidth, video.offsetHeight)

  video.srcObject = stream
  video.onloadedmetadata = () => {
    video.play()
  }
}

mainButton.addEventListener("click", async () => {
  if (!state.capture) return
  const depthCanvas = document.createElement("canvas")
  const imageCanvas = document.createElement("canvas")

  const tmpCanvas = document.createElement("canvas")
  const originalImage = await takePhoto(state.capture)
  const bitmap = await createImageBitmap(originalImage)
  tmpCanvas.height = 640
  tmpCanvas.width = 640
  if (bitmap.height > bitmap.width) {
    const diff = bitmap.height - bitmap.width
    tmpCanvas.getContext("2d")!.drawImage(bitmap, 0, diff / 2, bitmap.width, bitmap.width, 0, 0, 640, 640)
  } else {
    const diff = bitmap.width - bitmap.height
    tmpCanvas.getContext("2d")!.drawImage(bitmap, diff / 2, 0, bitmap.height, bitmap.height, 0, 0, 640, 640)
  }

  const res = await weightPredictor.predictWeight(
    tmpCanvas,
    imageCanvas,
    resultCanvas,
    depthCanvas,
    state.direction,
    state.calibration,
  )

  let [realBodyLength, realShoulderHeight, realRumpHeight, realBodyHeight, weight, distance, angle] = [0, 0, 0, 0, 0, 0, 0]
  if (res == null) {
    toast("<span>Keine Ziege erkannt</span>")
  } else {
    ;[realBodyLength, realShoulderHeight, realRumpHeight, realBodyHeight, weight, distance, angle] = res
  }
  valueContainer.innerHTML =
    `
        <div class="container">Body length: ${realBodyLength.toFixed(2)}</div>
        <div class="container">Shoulder height: ${realShoulderHeight.toFixed(2)}</div>
        <div class="container">rump height: ${realRumpHeight.toFixed(2)}</div>
        <div class="container">weight: ${weight.toFixed(2)}</div>
        <div class="container">distance: ${distance.toFixed(2)}</div>
        <div class="container">angle: ${angle.toFixed(2)}</div>
        <input id="tagInput" value="${state.tag}"/>
        <button id="btnSendData">Send Data<span id="spinner" class="hidden">Spinner</span></button>
        `
  showResultOverlay()

  valueContainer.querySelector<HTMLInputElement>("#tagInput")?.addEventListener("change", (ev) => {
    state.tag = (ev.currentTarget as HTMLInputElement).value
  })

  valueContainer.querySelector("button")?.addEventListener("click", async (ev) => {
    showSpinner()
    let maskImage: Blob | null;
    imageCanvas.toBlob(async (blob) => {
      maskImage = blob
      console.log(blob)
      if (blob != null) {
        await sendData({
          bodyLength: realBodyLength,
          rumpHeight: realRumpHeight,
          shoulderHeight: realShoulderHeight,
          weight: weight,
          distance: distance,
          angle: angle,
          maskedImage: blob,
          bodyHeight: realBodyHeight,
          originalImage: originalImage,
          tag: state.tag
        })
      }
      hideSpinner()
    })
  })
})

testButton.addEventListener('click', async () => {
  //testSingle(testContainer, yolo, new AngleProviderStatic(21.6), new DistanceProviderStatic(1.354))
  testAll(testContainer)
})

/**
 * Takes a photo and appends it onto the test area @paramcapture with image stream
 * @param imageCapture capture with image stream
 * @return blob
 */
async function takePhoto(imageCapture: ImageCapture) {
  const blob = await imageCapture.takePhoto()
  return blob
}

function showResultOverlay() {
  resultOverlay.classList.remove("hidden")
}

function hideResultOverlay() {
  resultOverlay.classList.add("hidden")
}

function resizeOverlayImage(videoWidth: number, videoHeight: number) {
  if (state.streamSettings) {
    const streamWidth = state.streamSettings.width ?? Number.MAX_VALUE
    const streamHeight = state.streamSettings.height ?? Number.MAX_VALUE
    const aspectRatio = state.streamSettings.aspectRatio ?? 1
    const minSize = Math.min(videoWidth, videoHeight)
    console.log("overlay resize", videoWidth, videoHeight, streamWidth, streamHeight)
    overlayImage.style.width = `${minSize / aspectRatio}px`
    overlayImage.style.height = `${minSize}px`
  }
}

function toast(html: string) {
  const toastElement = document.createElement("div")
  toastElement.classList.add("toast")
  toastElement.innerHTML = html
  toastContainer.appendChild(toastElement)
  setTimeout(() => {
    toastContainer.removeChild(toastElement)
  }, 3000)
}


type Payload = {
  bodyLength: number
  shoulderHeight: number
  rumpHeight: number
  maskedImage: Blob
  weight: number
  distance: number
  angle: number
  bodyHeight: number
  originalImage: Blob
  tag: string
}

async function sendData(payload: Payload) {
  const formData = new FormData()
  for (const [key, value] of Object.entries(payload)) {
    if (typeof value === "number") {
      formData.append(key, value.toString())
    } else {
      formData.append(key, value)
    }
  }

  const request = new Request(state.apiPath, {
    method: "POST",
    body: formData
  })

  try {
    const result = await fetch(request)
    console.log(result)
    if (result.ok) {
      toast("successfully send")
    } else {
      toast("error while sending")
    }
  } catch (e) {
    toast("error while sending")
    console.log(e)
  }
}

function showSpinner() {
  document.querySelector("#spinner")?.classList.remove("hidden")
}

function hideSpinner() {
  document.querySelector("#spinner")?.classList.add("hidden")
}

initPWA(app)
