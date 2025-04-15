import './style.css'
import { DistanceProviderStatic } from './distance-provider.ts'
import { initPWA } from './pwa.ts'
import { YOLO } from './yolotfjs.ts'
import './utils.ts'
import { AngleProvider } from './angle-provider.ts'
import { testAll, testSingle } from './testing.ts'
import { predictWeight } from './weight-prediction.ts'
import { bodyMeasurement, convertToCm } from './utils.ts'

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
      <video id="video"></video>
      <div id="overlay-container">
        <img id="overlay" src="/overlay2.png"/>
      </div>
      <div id="result-overlay" class="hidden">
        <canvas id="result-canvas"></canvas>
        <button id="result-overlay-close">X</button>
      </div>
    </div>
    <div id="right-controls">
      <button id="mainButton"></button>
      <button id="imageBtn">Take Photo</button>
    </div>
  </div>
  <div id="test">
  </div>
</div>
`

const appContainer = document.querySelector<HTMLDivElement>("div#app-container")!
const imageButton = document.querySelector<HTMLButtonElement>('#imageBtn')!
const mainButton = document.querySelector<HTMLButtonElement>("button#mainButton")!
const video = document.querySelector<HTMLVideoElement>('video#video')!
const testContainer = document.querySelector<HTMLElement>("#test")!
const resultCanvas = document.querySelector<HTMLCanvasElement>("canvas#result-canvas")!
const resultClose = document.querySelector<HTMLButtonElement>("#result-overlay-close")!
const resultOverlay = document.querySelector("#result-overlay")!

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

navigator.permissions.query({ name: "camera" }).then(async (perm) => {
  console.log(perm)
  if (perm.state != 'denied') {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 640 } } })!
    console.log(stream.getVideoTracks())
    video.srcObject = stream
    video.onloadedmetadata = () => {
      video.play()
    }

    mainButton.addEventListener("click", async () => {
      const container = document.createElement("div")
      const debugCanvas = document.createElement("canvas")
      const depthCanvas = document.createElement("canvas")

      await yoloProm
      const maskProm = yolo.predict(video, resultCanvas)
      const distanceProm = distanceProvider.distance(video, depthCanvas)
      const angleProm = angleProvider.angle()

      const [mask, distance, angle] = await Promise.all([maskProm, distanceProm, angleProm])

      if (mask != null) {
        let [bodyLength, shoulderHeight, rumpHeight] = await bodyMeasurement(mask, resultCanvas)
        const [realBodyLength, realShoulderHeight, realRumpHeight] = convertToCm(bodyLength, shoulderHeight, rumpHeight, { distance: distance, angle: angle })
        const weight = predictWeight(realBodyLength, realShoulderHeight, realRumpHeight, 0)
        const outputContainer = document.createElement("div")
        outputContainer.innerHTML =
          `
          <div>Body length: ${bodyLength.toFixed(2)}</div>
          <div>Shoulder height: ${shoulderHeight.toFixed(2)}</div>
          <div>rump height: ${rumpHeight.toFixed(2)}</div>
          <div>weight: ${weight.toFixed(2)}</div>
          <div>distance: ${distance.toFixed(2)}</div>
          <div>angle: ${angle.toFixed(2)}</div>
          `
        container.appendChild(debugCanvas)
        container.appendChild(outputContainer)
        testContainer.appendChild(container)
        showResultOverlay()
      }
    })
  }
})

let yolo = new YOLO()
const yoloProm = yolo.loadModel()
const distanceProvider = new DistanceProviderStatic()
const angleProvider = new AngleProvider()

imageButton.addEventListener('click', async () => {
  await yoloProm
  testAll(testContainer, yolo, angleProvider, distanceProvider)
})

function showResultOverlay() {
  resultOverlay.classList.remove("hidden")
}

function hideResultOverlay() {
  resultOverlay.classList.add("hidden")
}

initPWA(app)
