import { AngleProvider } from "./angle-provider"
import { DistanceProvider } from "./distance-provider"
import { body_measurement, convert_to_cm } from "./utils"
import { YOLO } from "./yolotfjs"

export async function testSingle(container: HTMLElement, yolo: YOLO, angleProvider: AngleProvider, distanceProvider: DistanceProvider) {
  container.innerHTML =
    `
<div>
  <div class="result-container">
    <div class="image-container">
      <img src="test/IMG_1612_Carina.JPG" />
    </div>
    <canvas class="debug-canvas" ></canvas>
    <canvas class="depth-canvas" ></canvas>
  </div>
</div>
`
  let rContainers = container.querySelectorAll(".result-container")
  for (let rContainer of rContainers) {
    const imageEl = rContainer.querySelector("img")!
    const debugCanvas = rContainer.querySelector<HTMLCanvasElement>(".debug-canvas")!
    const depthCanvas = rContainer.querySelector<HTMLCanvasElement>(".depth-canvas")!
    await imageEl.decode()
    await test(rContainer, imageEl, debugCanvas, depthCanvas, yolo, distanceProvider, angleProvider)
  }

  console.log("finished all tests")
}

export async function testAll(container: HTMLElement, yolo: YOLO, angleProvider: AngleProvider, distanceProvider: DistanceProvider) {
  container.innerHTML =
    `
<div>
  <div class="result-container">
    <div class="image-container">
      <img src="test/IMG_1600_Twentyone.JPG" />
    </div>
    <canvas class="debug-canvas" ></canvas>
    <canvas class="depth-canvas" ></canvas>
  </div>
  <div class="result-container">
    <div class="image-container">
      <img src="test/IMG_1595_Diego.JPG" />
    </div>
    <canvas class="debug-canvas" ></canvas>
    <canvas class="depth-canvas" ></canvas>
  </div>
  <div class="result-container">
    <div class="image-container">
      <img src="test/IMG_1590_Diego.JPG" />
    </div>
    <canvas class="debug-canvas" ></canvas>
    <canvas class="depth-canvas" ></canvas>
  </div>
  <div class="result-container">
    <div class="image-container">
      <img src="test/IMG_1591_Diego.JPG" />
    </div>
    <canvas class="debug-canvas" ></canvas>
    <canvas class="depth-canvas" ></canvas>
  </div>
  <div class="result-container">
    <div class="image-container">
      <img src="test/IMG_1592_Diego.JPG" />
    </div>
    <canvas class="debug-canvas" ></canvas>
    <canvas class="depth-canvas" ></canvas>
  </div>
  <div class="result-container">
    <div class="image-container">
      <img src="test/IMG_1593_Diego.JPG" />
    </div>
    <canvas class="debug-canvas" ></canvas>
    <canvas class="depth-canvas" ></canvas>
  </div>
  <div class="result-container">
    <div class="image-container">
      <img src="test/IMG_1594_Diego.JPG" />
    </div>
    <canvas class="debug-canvas" ></canvas>
    <canvas class="depth-canvas" ></canvas>
  </div>
  <div class="result-container">
    <div class="image-container">
      <img src="test/IMG_1598_Twentyone.JPG" />
    </div>
    <canvas class="debug-canvas" ></canvas>
    <canvas class="depth-canvas" ></canvas>
  </div>
  <div class="result-container">
    <div class="image-container">
      <img src="test/IMG_1599_Twentyone.JPG" />
    </div>
    <canvas class="debug-canvas" ></canvas>
    <canvas class="depth-canvas" ></canvas>
  </div>
  <div class="result-container">
    <div class="image-container">
      <img src="test/IMG_1606_Zara.JPG" />
    </div>
    <canvas class="debug-canvas" ></canvas>
    <canvas class="depth-canvas" ></canvas>
  </div>
  <div class="result-container">
    <div class="image-container">
      <img src="test/IMG_1605_Zara.JPG" />
    </div>
    <canvas class="debug-canvas" ></canvas>
    <canvas class="depth-canvas" ></canvas>
  </div>
  <div class="result-container">
    <div class="image-container">
      <img src="test/IMG_1612_Carina.JPG" />
    </div>
    <canvas class="debug-canvas" ></canvas>
    <canvas class="depth-canvas" ></canvas>
  </div>
  <div class="result-container">
    <div class="image-container">
      <img src="test/IMG_1610_Carina.JPG" />
    </div>
    <canvas class="debug-canvas" ></canvas>
    <canvas class="depth-canvas" ></canvas>
  </div>
  <div class="result-container">
    <div class="image-container">
      <img src="test/IMG_1611_Carina.JPG" />
    </div>
    <canvas class="debug-canvas" ></canvas>
    <canvas class="depth-canvas" ></canvas>
  </div>
  <div class="result-container">
    <div class="image-container">
      <img src="test/IMG_1619_36175.JPG" />
    </div>
    <canvas class="debug-canvas" ></canvas>
    <canvas class="depth-canvas" ></canvas>
  </div>
  <div class="result-container">
    <div class="image-container">
      <img src="test/IMG_1618_36175.JPG" />
    </div>
    <canvas class="debug-canvas" ></canvas>
    <canvas class="depth-canvas" ></canvas>
  </div>
</div>
`
  let rContainers = container.querySelectorAll(".result-container")
  for (let rContainer of rContainers) {
    const imageEl = rContainer.querySelector("img")!
    const debugCanvas = rContainer.querySelector<HTMLCanvasElement>(".debug-canvas")!
    const depthCanvas = rContainer.querySelector<HTMLCanvasElement>(".depth-canvas")!
    await imageEl.decode()
    await test(rContainer, imageEl, debugCanvas, depthCanvas, yolo, distanceProvider, angleProvider)
  }

  console.log("finished all tests")
}

async function test(container: Element, imageEl: HTMLImageElement, debugCanvas: HTMLCanvasElement, depthCanvas: HTMLCanvasElement, yolo: YOLO, distanceProvider: DistanceProvider, angleProvider: AngleProvider) {
  const maskProm = yolo.predict(imageEl, debugCanvas)
  // const sam2mask = await createMask(imageEl, debugCanvas)
  const distanceProm = distanceProvider.distance(imageEl, depthCanvas)
  const angleProm = angleProvider.angle()

  const [mask, distance, angle] = await Promise.all([maskProm, distanceProm, angleProm])

  if (mask != null) {
    let [bodyLength, shoulderHeight, rumpHeight] = await body_measurement(mask, debugCanvas)
    const [realBodyLength, realShoulderHeight, realRumpHeight] = convert_to_cm(bodyLength, shoulderHeight, rumpHeight, { distance: distance, angle: angle })
    testOutput(container, realBodyLength, realShoulderHeight, realRumpHeight, 0, distance, angle)
  }
}

async function testOutput(container: Element, bodyLength: number, shoudlerHeight: number, rumpHeight: number, weight: number, distance: number, angle: number) {
  const outputContainer = document.createElement("div")
  outputContainer.innerHTML =
    `
    <div>Body length: ${bodyLength.toFixed(2)}</div>
    <div>Shoulder height: ${shoudlerHeight.toFixed(2)}</div>
    <div>rump height: ${rumpHeight.toFixed(2)}</div>
    <div>weight: ${weight.toFixed(2)}</div>
    <div>distance: ${distance.toFixed(2)}</div>
    <div>angle: ${angle.toFixed(2)}</div>
    `
  container.appendChild(outputContainer)
}
