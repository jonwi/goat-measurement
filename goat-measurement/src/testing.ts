import { AngleProvider } from "./angle-provider"
import { DistanceProvider } from "./distance-provider"
import { bodyMeasurement, convertToCm } from "./utils"
import { predictWeight } from "./weight-prediction"
import { YOLO } from "./yolotfjs"

export async function testSingle(container: HTMLElement, yolo: YOLO, angleProvider: AngleProvider, distanceProvider: DistanceProvider) {
  container.innerHTML = `<div>${createResultContainer("test/24_image.png")}</div>`

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
  /*
  const images = [
    "test/IMG_1600_Twentyone.JPG",
    "test/IMG_1595_Diego.JPG",
    "test/IMG_1590_Diego.JPG",
    "test/IMG_1591_Diego.JPG",
    "test/IMG_1592_Diego.JPG",
    "test/IMG_1593_Diego.JPG",
    "test/IMG_1594_Diego.JPG",
    "test/IMG_1598_Twentyone.JPG",
    "test/IMG_1599_Twentyone.JPG",
    "test/IMG_1606_Zara.JPG",
    "test/IMG_1605_Zara.JPG",
    "test/IMG_1612_Carina.JPG",
    "test/IMG_1610_Carina.JPG",
    "test/IMG_1611_Carina.JPG",
    "test/IMG_1619_36175.JPG",
    "test/IMG_1618_36175.JPG",
  ]
  */
  const images = [
    "test/23_image.png",
    "test/24_image.png",
    "test/25_image.png",
    "test/26_image.png",
    "test/27_image.png",
    "test/28_image.png",
    "test/29_image.png",
  ]
  container.innerHTML =
    `
    <div>
      ${images.map((i) => { createResultContainer(i) })}
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

function createResultContainer(imgSrc: string) {
  return `
  <div class="result-container">
    <div class="image-container">
      <img src="${imgSrc}" />
    </div>
    <canvas class="debug-canvas" ></canvas>
    <canvas class="depth-canvas" ></canvas>
  </div>
`
}

async function test(container: Element, imageEl: HTMLImageElement, debugCanvas: HTMLCanvasElement, depthCanvas: HTMLCanvasElement, yolo: YOLO, distanceProvider: DistanceProvider, angleProvider: AngleProvider) {
  console.log("canvas", debugCanvas)
  const maskProm = yolo.predict(imageEl, debugCanvas, debugCanvas)
  // const sam2mask = await createMask(imageEl, debugCanvas)
  const distanceProm = distanceProvider.distance(imageEl, depthCanvas)
  const angleProm = angleProvider.angle()

  const [mask, distance, angle] = await Promise.all([maskProm, distanceProm, angleProm])

  if (mask != null) {
    let [bodyLength, shoulderHeight, rumpHeight] = await bodyMeasurement(mask, debugCanvas)
    const [realBodyLength, realShoulderHeight, realRumpHeight] = convertToCm(
      bodyLength,
      shoulderHeight,
      rumpHeight,
      { distance: distance, angle: angle, calibration: 200 }
    )
    const weight = predictWeight(realBodyLength, realShoulderHeight, realRumpHeight, 0)
    testOutput(container, realBodyLength, realShoulderHeight, realRumpHeight, weight, distance, angle)
  }
}

async function testOutput(container: Element, bodyLength: number, shoudlerHeight: number, rumpHeight: number, weight: number, distance: number, angle: number) {
  const outputContainer = document.createElement("div")
  outputContainer.innerHTML =
    `
    <div> Body length: ${bodyLength.toFixed(2)} </div>
    <div> Shoulder height: ${shoudlerHeight.toFixed(2)} </div>
    <div> rump height: ${rumpHeight.toFixed(2)} </div>
    <div> weight: ${weight.toFixed(2)} </div>
    <div> distance: ${distance.toFixed(2)} </div>
    <div> angle: ${angle.toFixed(2)} </div>
    `
  container.appendChild(outputContainer)
}
