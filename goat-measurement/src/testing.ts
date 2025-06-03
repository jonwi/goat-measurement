import { AngleProviderStatic } from "./angle-provider"
import { DistanceProviderSecond } from "./distance-provider"
import { ONNX } from "./onnx"
import { WeightPredictor } from "./weight-prediction"
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

const ONE = [
  "test/25_image.png",
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
  const images = ONE

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

  let lowestMeanWeight = 100
  let lowestCalibration = 0

  for (let calibration of Array.from(Array(1).keys()).map((i) => 2.5 + i / 100)) {
    console.log("using calibration:", calibration)
    const bodyPcts = []
    const shoulderPcts = []
    const rumpPcts = []
    const weightPcts = []
    for (let rContainer of rContainers) {
      const imageEl = rContainer.querySelector("img")!
      const debugCanvas = rContainer.querySelector<HTMLCanvasElement>(".debug-canvas")!
      const depthCanvas = rContainer.querySelector<HTMLCanvasElement>(".depth-canvas")!
      await imageEl.decode()

      const testResult = await test(rContainer, imageEl, debugCanvas, depthCanvas, weightPredictor, calibration)
      if (testResult != null) {
        bodyPcts.push(testResult.bodyPercentage)
        shoulderPcts.push(testResult.shoulderPercentage)
        rumpPcts.push(testResult.rumpPercentage)
        weightPcts.push(testResult.weightPercentage)
      }
    }
    const meanWeight = mean(weightPcts)
    if (meanWeight < lowestMeanWeight) {
      lowestMeanWeight = meanWeight
      lowestCalibration = calibration
      console.log("new lowest weight", meanWeight, calibration)
    }
    console.log("finished all tests")
    console.log("BodyLength mape:", mean(bodyPcts))
    console.log("ShoulderHeight mape:", mean(shoulderPcts))
    console.log("RumpHeight mape:", mean(rumpPcts))
    console.log("Weight mape:", mean(weightPcts))
  }

  console.log("lowest weight mape at", lowestCalibration, lowestMeanWeight)
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

async function test(container: Element, imageEl: HTMLImageElement, debugCanvas: HTMLCanvasElement, depthCanvas: HTMLCanvasElement, weightPredictor: WeightPredictor, calibration: number) {
  const imagePrefix = imageEl.src.split("_")[0]

  const groundTruth = await getData(imagePrefix)

  const res = await weightPredictor.predictWeight(
    imageEl,
    debugCanvas,
    debugCanvas,
    depthCanvas,
    groundTruth["Direction"],
    calibration
  )
  if (res != null) {
    const [realBodyLength, realShoulderHeight, realRumpHeight, realBodyHeight, weight, distance, angle] = res
    const result = await testOutput(container, realBodyLength, realShoulderHeight, realRumpHeight, realBodyHeight, weight, distance, angle, groundTruth)
    return result
  }
  return null
}

type TestResult = {
  bodyPercentage: number
  shoulderPercentage: number
  rumpPercentage: number
  weightPercentage: number
  bodyLength: number
  shoulderHeight: number
  rumpHeight: number
  bodyHeight: number
  weight: number
  angle: number
  distance: number
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
  //container.appendChild(outputContainer)

  const result: TestResult = {
    bodyPercentage: bodyPercentage,
    shoulderPercentage: shoulderPercentage,
    rumpPercentage: rumpPercentage,
    weightPercentage: weightPercentage,
    bodyLength: bodyLength,
    shoulderHeight: shoulderHeight,
    rumpHeight: rumpHeight,
    bodyHeight: bodyHeight,
    weight: weight,
    angle: angle,
    distance: distance,
  }
  return result
}
