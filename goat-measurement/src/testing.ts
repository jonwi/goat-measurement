import { AngleProvider, AngleProviderStatic } from "./angle-provider"
import { DistanceProvider, DistanceProviderSecond } from "./distance-provider"
import { bodyMeasurement, convertToCm } from "./utils"
import { predictWeight, WeightPredictor } from "./weight-prediction"
import { YOLO } from "./yolotfjs"

const FIRST_MEASUREMENT = [
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

const SECOND_MEASUREMENT = [
  "test/4_image.png",
  "test/5_image.png",
  "test/6_image.png",
  "test/7_image.png",
  "test/8_image.png",
  "test/9_image.png",
  "test/10_image.png",
  "test/11_image.png",
  "test/12_image.png",
  "test/13_image.png",
  "test/14_image.png",
  "test/15_image.png",
  "test/16_image.png",
  "test/17_image.png",
  "test/18_image.png",
  "test/19_image.png",
  "test/20_image.png",
  "test/21_image.png",
  "test/22_image.png",
  "test/23_image.png",
  "test/24_image.png",
  "test/25_image.png",
  "test/26_image.png",
  "test/27_image.png",
  "test/28_image.png",
  "test/29_image.png",
]

const SECOND_REFERENZ = [
  "test/23_image.png",
  "test/24_image.png",
  "test/25_image.png",
  "test/26_image.png",
  "test/27_image.png",
  "test/28_image.png",
  "test/29_image.png",
]

const SECOND_REFERENZ_NO_DIEGO = [
  "test/23_image.png",
  "test/24_image.png",
  "test/25_image.png",
  "test/26_image.png",
  "test/27_image.png",
  "test/28_image.png",
]

const SECOND_CLEAN = [
  "test/4_image.png",
  "test/5_image.png",
  "test/7_image.png",
  "test/8_image.png",
  "test/9_image.png",
  "test/10_image.png",
  "test/14_image.png",
  "test/15_image.png",
  "test/16_image.png",
  "test/17_image.png",
  "test/18_image.png",
  "test/19_image.png",
  "test/20_image.png",
  "test/21_image.png",
  "test/22_image.png",
  "test/23_image.png",
  "test/24_image.png",
  "test/25_image.png",
  "test/26_image.png",
  "test/27_image.png",
  "test/28_image.png",
  "test/29_image.png",
]

const SECOND_DETECTED = [
  "test/4_image.png",
  "test/5_image.png",
  "test/7_image.png",
  "test/8_image.png",
  "test/9_image.png",
  "test/14_image.png",
  "test/15_image.png",
  "test/16_image.png",
  "test/17_image.png",
  "test/21_image.png",
  "test/22_image.png",
  "test/23_image.png",
  "test/24_image.png",
  "test/25_image.png",
  "test/26_image.png",
  "test/27_image.png",
  "test/28_image.png",
  "test/29_image.png",
]

const SECOND_OUTSIDE = [
  "test/4_image.png",
  "test/5_image.png",
  "test/7_image.png",
  "test/8_image.png",
  "test/9_image.png",
  "test/14_image.png",
  "test/15_image.png",
  "test/16_image.png",
  "test/17_image.png",
  "test/21_image.png",
  "test/22_image.png",
]

export async function testAll(container: HTMLElement) {
  const images = SECOND_REFERENZ_NO_DIEGO

  container.innerHTML =
    `
    <div>
      ${images.map((img, i) => { return createResultContainer(img) }).join("")}
    </div>
    `
  let rContainers = container.querySelectorAll(".result-container")

  const weightPredictor = new WeightPredictor(
    new YOLO(),
    new AngleProviderStatic(10),
    new DistanceProviderSecond(),
  )
  const bodyPcts = []
  const shoulderPcts = []
  const rumpPcts = []
  const weightPcts = []

  for (let rContainer of rContainers) {
    const imageEl = rContainer.querySelector("img")!
    const debugCanvas = rContainer.querySelector<HTMLCanvasElement>(".debug-canvas")!
    const depthCanvas = rContainer.querySelector<HTMLCanvasElement>(".depth-canvas")!
    await imageEl.decode()
    const percentages = await test(rContainer, imageEl, debugCanvas, depthCanvas, weightPredictor)
    if (percentages != null) {
      const [bodyPercentage, shoulderPercentage, rumpPercentage, weightPercentage, anglePercentage, distancePercentage] = percentages
      bodyPcts.push(bodyPercentage)
      shoulderPcts.push(shoulderPercentage)
      rumpPcts.push(rumpPercentage)
      weightPcts.push(weightPercentage)
    }
  }
  console.log("finished all tests")
  console.log("BodyLength accuracy:", mean(bodyPcts))
  console.log("ShoulderHeight accuracy:", mean(shoulderPcts))
  console.log("RumpHeight accuracy:", mean(rumpPcts))
  console.log("Weight accuracy:", mean(weightPcts))
}

function meanAbsolutePercentageError(arr: number[]) {
  return arr.map((c) => Math.abs(c - 1)).reduce((p, c) => p + c, 0) / arr.length
}

function mean(arr: number[]) {
  return arr.reduce((p, c) => p + c, 0) / arr.length
}

function absolutePercentageError(pred: number, truth: number) {
  return Math.abs(pred - truth) / truth
}

function createResultContainer(imgSrc: string) {
  return `
  <div class="result-container">
    <div class="image-container">
      <img src="${imgSrc}" />
    </div>
    <canvas class="debug-canvas" ></canvas>
  </div>
`
}

async function test(container: Element, imageEl: HTMLImageElement, debugCanvas: HTMLCanvasElement, depthCanvas: HTMLCanvasElement, weightPredictor: WeightPredictor) {
  console.log("canvas", debugCanvas)
  const imagePrefix = imageEl.src.split("_")[0]
  console.log("num", imagePrefix)

  const groundTruth = await getData(imagePrefix)

  const res = await weightPredictor.predictWeight(
    imageEl,
    debugCanvas,
    debugCanvas,
    depthCanvas,
    groundTruth["Direction"],
    3.575
  )
  if (res != null) {
    const [realBodyLength, realShoulderHeight, realRumpHeight, realBodyHeight, weight, distance, angle] = res
    const percentages = testOutput(container, realBodyLength, realShoulderHeight, realRumpHeight, realBodyHeight, weight, distance, angle, groundTruth)
    return percentages
  }
  return null
}

type ImageData = {
  Direction: "left" | "right",
  Angle: number,
  Distance: number,
  BodyLength: number,
  ShoulderHeight: number,
  RumpHeight: number,
  Weight: number,
}

async function getData(imagePrefix: string): Promise<ImageData> {
  const res = await fetch(`${imagePrefix}_data.json`)
  return await res.json()
}

async function testOutput(container: Element, bodyLength: number, shoulderHeight: number, rumpHeight: number, bodyHeight: number, weight: number, distance: number, angle: number, groundTruth: ImageData) {
  const outputContainer = document.createElement("div")
  const bodyPercentage = absolutePercentageError(bodyLength, groundTruth.BodyLength)
  const shoulderPercentage = absolutePercentageError(shoulderHeight, groundTruth.ShoulderHeight)
  const rumpPercentage = absolutePercentageError(rumpHeight, groundTruth.RumpHeight)
  const weightPercentage = absolutePercentageError(weight, groundTruth.Weight)
  const anglePercentage = absolutePercentageError(angle, groundTruth.Angle)
  const distancePercentage = absolutePercentageError(distance, groundTruth.Distance)

  outputContainer.innerHTML =
    `
    <div> Body length: ${bodyLength.toFixed(2)} ${groundTruth.BodyLength} <span>%: ${bodyPercentage.toFixed(2)}</span></div>
    <div> Shoulder height: ${shoulderHeight.toFixed(2)} ${groundTruth.ShoulderHeight} <span>%: ${shoulderPercentage.toFixed(2)}</span></div>
    <div> rump height: ${rumpHeight.toFixed(2)} ${groundTruth.RumpHeight} <span>%: ${rumpPercentage.toFixed(2)}</span></div>
    <div> body height: ${bodyHeight.toFixed(2)}  </div>
    <div> weight: ${weight.toFixed(2)} ${groundTruth.Weight} <span>%: ${weightPercentage.toFixed(2)}</span></div>
    <div> distance: ${distance.toFixed(2)} ${groundTruth.Distance} <span>%: ${distancePercentage.toFixed(2)}</span></div>
    <div> angle: ${angle.toFixed(2)} ${groundTruth.Angle} <span>%: ${anglePercentage.toFixed(2)}</span></div>
    `
  container.appendChild(outputContainer)
  return [bodyPercentage, shoulderPercentage, rumpPercentage, weightPercentage, anglePercentage, distancePercentage]
}
