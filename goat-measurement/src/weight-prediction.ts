import { AngleProvider } from "./angle-provider"
import { DistanceProvider } from "./distance-provider"
import { bodyMeasurement, convertToCm } from "./utils"
import { YOLO } from "./yolotfjs"



export class WeightPredictor {
  angleProvider: AngleProvider
  distanceProvider: DistanceProvider
  yolo: YOLO
  yoloProm: Promise<void>

  constructor(yolo: YOLO, angleProvider: AngleProvider, distanceProvider: DistanceProvider) {
    this.angleProvider = angleProvider
    this.distanceProvider = distanceProvider
    this.yolo = yolo
    this.yoloProm = yolo.loadModel()
  }

  async predictWeight(
    input: HTMLImageElement | HTMLVideoElement,
    imageCanvas: HTMLCanvasElement,
    resultCanvas: HTMLCanvasElement,
    depthCanvas: HTMLCanvasElement,
    direction: "right" | "left",
    calibration: number
  ) {
    await this.yoloProm
    const maskProm = this.yolo.predict(input, imageCanvas, resultCanvas)
    const distanceProm = this.distanceProvider.distance(input, depthCanvas)
    const angleProm = this.angleProvider.angle(input)

    let [[mask, box], distance, angle] = await Promise.all([maskProm, distanceProm, angleProm])

    let [realBodyLength, realShoulderHeight, realRumpHeight, weight] = [0, 0, 0, 0]
    if (mask != null && box != null) {
      let [bodyLength, shoulderHeight, rumpHeight] = await bodyMeasurement(mask, box, resultCanvas, direction)

        ;[realBodyLength, realShoulderHeight, realRumpHeight] = convertToCm(
          bodyLength,
          shoulderHeight,
          rumpHeight,
          { distance: distance, angle: angle, calibration: calibration }
        );
      weight = this.linearRegression(realBodyLength, realShoulderHeight, realRumpHeight, 0)
      return [realBodyLength, realShoulderHeight, realRumpHeight, weight, distance, angle]
    }
    return null
  }

  linearRegression(bodyLength: number, shoulderHeight: number, rumpHeight: number, heartGirth: number) {
    console.log("bodyLength", bodyLength, "shoulderHeight", shoulderHeight, "rumpHeight", rumpHeight)
    return bodyLength * 0.45287999 + rumpHeight * 1.30813392 + shoulderHeight * 0.55532975 - 111.45145379928671
  }
}
