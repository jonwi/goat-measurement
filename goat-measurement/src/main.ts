import './style.css'
import { DistanceProiderStatic as DistanceProviderStatic, DistanceProvider } from './distance-provider.ts'
import { initPWA } from './pwa.ts'
import { YOLO } from './yolotfjs.ts'
import './utils.ts'
import './sam2.ts'
import { AngleProvider } from './angle-provider.ts'
import { testAll, testSingle } from './testing.ts'

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

const imageButton = document.querySelector<HTMLButtonElement>('#imageBtn')!
const video = document.querySelector<HTMLVideoElement>('video#video')!
navigator.permissions.query({ name: "camera" }).then(async (perm) => {
  console.log(perm)
  if (perm.state != 'denied') {
    const stream = await navigator.mediaDevices.getUserMedia({ video })!
    video.srcObject = stream
    video.onloadedmetadata = () => {
      video.play();
    }
  }
})

const testContainer = document.querySelector<HTMLElement>("#test")!
let yolo = new YOLO()
const yoloProm = yolo.loadModel()
const distanceProvider = new DistanceProviderStatic()
const angleProvider = new AngleProvider()

imageButton.addEventListener('click', async () => {
  await yoloTFJS()
})

async function yoloTFJS() {
  await yoloProm
  testAll(testContainer, yolo, angleProvider, distanceProvider)
}

initPWA(app)
