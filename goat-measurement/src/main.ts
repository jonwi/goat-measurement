import './style.css'
import { DistanceProviderInput, DistanceProviderStatic } from './distance-provider.ts'
import { initPWA } from './pwa.ts'
import { YOLO } from './yolotfjs.ts'
import './utils.ts'
import { AngleProviderStatic, AngleProviderSensor } from './angle-provider.ts'
import { testAll, testSingle } from './testing.ts'
import { predictWeight } from './weight-prediction.ts'
import { bodyMeasurement, convertToCm } from './utils.ts'

type Direction = "left" | "right"
type AppState = {
  direction: Direction
  calibration: number
}

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
        <img id="overlay" src="/overlay2.png"/>
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

const state: AppState = { direction: "left", calibration: 149.85 }
calibrationInput.addEventListener("change", () => {
  state.calibration = parseFloat(calibrationInput.value)
})
calibrationInput.value = "149.85"

appContainer.style.width = `${window.innerWidth - 10}px`
appContainer.style.height = `${window.innerHeight - 10}px`
window.addEventListener("resize", () => {
  appContainer.style.width = `${window.innerWidth - 10}px`
  appContainer.style.height = `${window.innerHeight - 10}px`
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

let yolo = new YOLO()
const yoloProm = yolo.loadModel()
const distanceProvider = new DistanceProviderInput()
const angleProvider = new AngleProviderSensor()

setInterval(async () => {
  const angle = await angleProvider.angle()
  angleContainer.innerText = `${angle.toFixed(2)}`
}, 100)

navigator.permissions.query({ name: "camera" }).then(async (perm) => {
  if (perm.state != 'denied') {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: "environment" }, width: { ideal: 640 }, height: { ideal: 640 } } })!
    const streamSettings = stream.getVideoTracks()[0].getSettings()
    resizeOverlayImage(video.offsetWidth, video.offsetHeight, streamSettings.width ?? Number.MAX_VALUE, streamSettings.height ?? Number.MAX_VALUE, streamSettings.aspectRatio ?? 1)
    window.addEventListener("resize", () => {
      resizeOverlayImage(video.offsetWidth, video.offsetHeight, streamSettings.width ?? Number.MAX_VALUE, streamSettings.height ?? Number.MAX_VALUE, streamSettings.aspectRatio ?? 1)
    })

    video.srcObject = stream
    video.onloadedmetadata = () => {
      video.play()
    }

    mainButton.addEventListener("click", async () => {
      const depthCanvas = document.createElement("canvas")
      const imageCanvas = document.createElement("canvas")

      await yoloProm
      const maskProm = yolo.predict(video, imageCanvas, resultCanvas)
      const distanceProm = distanceProvider.distance(video, depthCanvas)
      const angleProm = angleProvider.angle()

      let [mask, distance, angle] = await Promise.all([maskProm, distanceProm, angleProm])

      if (mask != null) {
        if (state.direction == "right") {
          mask = mask.reverse(1)
        }
        let [bodyLength, shoulderHeight, rumpHeight] = await bodyMeasurement(mask, resultCanvas)
        const [realBodyLength, realShoulderHeight, realRumpHeight] = convertToCm(
          bodyLength,
          shoulderHeight,
          rumpHeight,
          { distance: distance, angle: angle, calibration: state.calibration }
        )
        const weight = predictWeight(realBodyLength, realShoulderHeight, realRumpHeight, 0)
        valueContainer.innerHTML =
          `
          <div class="container">Body length: ${realBodyLength.toFixed(2)}</div>
          <div class="container">Shoulder height: ${realShoulderHeight.toFixed(2)}</div>
          <div class="container">rump height: ${realRumpHeight.toFixed(2)}</div>
          <div class="container">weight: ${weight.toFixed(2)}</div>
          <div class="container">distance: ${distance.toFixed(2)}</div>
          <div class="container">angle: ${angle.toFixed(2)}</div>
          <button>Send Data</button>
          `
        showResultOverlay()

        valueContainer.querySelector("button")?.addEventListener("click", () => {
          sendData({
            bodyLength: realBodyLength,
            rumpHeight: realRumpHeight,
            shoulderHeight: realShoulderHeight,
            weight: weight,
            distance: distance,
            angle: angle,
            image: imageCanvas.toDataURL()
          })
        })
      } else {
        toast("<span>Keine Ziege erkannt</span>")
      }
    })
  } else {
    toast("camera permission denied")
  }
})


testButton.addEventListener('click', async () => {
  await yoloProm
  testSingle(testContainer, yolo, new AngleProviderStatic(21.6), new DistanceProviderStatic(1.354))
  // testAll(testContainer, yolo, new AngleProviderStatic(1.5), new DistanceProviderStatic(1.5))
})

function showResultOverlay() {
  resultOverlay.classList.remove("hidden")
}

function hideResultOverlay() {
  resultOverlay.classList.add("hidden")
}

function resizeOverlayImage(videoWidth: number, videoHeight: number, streamWidth: number, streamHeight: number, aspectRatio: number) {
  const minSize = Math.min(videoWidth, videoHeight)
  if (minSize < streamWidth || minSize < streamHeight) {
    // TODO: check if height or width restricted. almost always it will be height because of screen.
    overlayImage.style.width = `${minSize / aspectRatio}px`
    overlayImage.style.height = `${minSize}px`
  } else {
    overlayImage.style.width = `${streamWidth}px`
    overlayImage.style.height = `${streamHeight}px`
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
  image: string
  weight: number
  distance: number
  angle: number
}

async function sendData(payload: Payload) {
  const request = new Request("http://localhost:8080", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
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

initPWA(app)
