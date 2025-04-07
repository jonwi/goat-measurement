import './style.css'
import { initPWA } from './pwa.ts'
import { YOLO } from './yolotfjs.ts'
import './utils.ts'
import { body_measurement, convert_to_cm } from './utils.ts'

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
    Test stuff that we dont need for the app
    <div id="image-container">
      <img id="image" src="/example.jpg" />
    </div>
    <canvas id="debug-output" width="640" height="640"></canvas>
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

const imageEl = document.querySelector<HTMLImageElement>("#image")!
const debugCanvas = document.querySelector<HTMLCanvasElement>("#debug-output")!

let yolo = new YOLO()
const yoloProm = yolo.loadModel()

imageButton.addEventListener('click', async () => {
  await yoloTFJS()
})

async function yoloTFJS() {
  await yoloProm
  let mask = await yolo.predict(imageEl, debugCanvas)
  if (mask != null) {
    let [body_length, shoulder_height, rump_height] = await body_measurement(mask, debugCanvas)
    console.log("pixel values: ")
    console.log(body_length, shoulder_height, rump_height)
    console.log("real values: ")
    console.log(convert_to_cm(body_length, shoulder_height, rump_height, 1.64))
  }
}

initPWA(app)
