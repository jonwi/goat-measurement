import './style.css'
import { initPWA } from './pwa.ts'
import { YOLO } from './yolotfjs.ts'

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
<h1>Goat Weight Measurement</h1>
  <video id="video"></video>
  <button id="imageBtn">Take Photo</button>
  <div id="image-container">
    <img id="image" src="/example.jpg" />
  </div>
</div>
`

const imageButton = document.querySelector<HTMLButtonElement>('#imageBtn')!
const video = document.querySelector<HTMLVideoElement>('video#video')!
const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1024, height: 1024 } })!
video.srcObject = stream
video.onloadedmetadata = () => {
  video.play();
}
const imageEl = document.querySelector<HTMLImageElement>("#image")!

imageButton.addEventListener('click', async () => {
  await yoloTFJS()
})

async function yoloTFJS() {
  let yolo = new YOLO()
  await yolo.loadModel()
  yolo.predict(imageEl)
}

initPWA(app)
