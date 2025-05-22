import { AngleProvider } from "./angle-provider"
import { DistanceProvider } from "./distance-provider"
import { bodyMeasurement, convertToCm } from "./utils"
import { YOLO } from "./yolotfjs"



const data: number[][] = []

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

  /**
   * Predicts the weight from and image. Writes to imageCanvas, resultCanvas and depthCanvas.
   * @param input input for the prediction either video or image
   * @param imageCanvas this canvas receives the image that was used as base for prediction
   * @param resultCanvas this canvas will receive the mask and body measurements
   * @param depthCanvas this canvas can show depth information
   * @param direction this is the direction the goat is facing
   * @param calibration this value is used to get cm from pixels
   * @returns realBodyLength, realShoulderHeight, realRumpHeight, realBodyHeight, weight, distance, angle
   */
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

    let [realBodyLength, realShoulderHeight, realRumpHeight, realBodyHeight, weight] = [0, 0, 0, 0, 0]
    if (mask != null && box != null) {
      let [bodyLength, shoulderHeight, rumpHeight, bodyHeight] = await bodyMeasurement(mask, box, resultCanvas, direction)

        ;[realBodyLength, realShoulderHeight, realRumpHeight, realBodyHeight] = convertToCm(
          bodyLength,
          shoulderHeight,
          rumpHeight,
          bodyHeight,
          { distance: distance, angle: angle, calibration: calibration }
        );
      weight = this.linearRegression(realBodyLength, realShoulderHeight, realRumpHeight, 0, bodyHeight)

      data.push([realBodyLength, realShoulderHeight, realRumpHeight, realBodyHeight, weight, distance, angle])
      console.log(data)
      return [realBodyLength, realShoulderHeight, realRumpHeight, realBodyHeight, weight, distance, angle]
    }
    return null
  }

  linearRegression(bodyLength: number, shoulderHeight: number, rumpHeight: number, heartGirth: number, bodyHeight: number) {
    // no intercept poly features
    //let features = [bodyLength, shoulderHeight, rumpHeight, bodyLength * bodyLength, bodyLength * shoulderHeight, bodyLength * rumpHeight, shoulderHeight * shoulderHeight, shoulderHeight * rumpHeight, rumpHeight * rumpHeight]
    //let coef = [-2.08840742, -6.20480081, 7.19363164, 0.05400854, -0.12578866, 0.04646824, 0.42782325, -0.59769451, 0.22111895]
    //const intercept = 0

    // intercept poly features
    //let features = [bodyLength, shoulderHeight, rumpHeight, bodyLength * bodyLength, bodyLength * shoulderHeight, bodyLength * rumpHeight, shoulderHeight * shoulderHeight, shoulderHeight * rumpHeight, rumpHeight * rumpHeight]
    //let coef = [-1.88491089, -12.39351589, 8.00282563, 0.05059217, -0.11032338, 0.03724437, 0.47468567, -0.61793588, 0.22989428]
    //const intercept = 173.6511563294875

    // intercept positive only poly features
    //let features = [bodyLength, shoulderHeight, rumpHeight, bodyLength * bodyLength, bodyLength * shoulderHeight, bodyLength * rumpHeight, shoulderHeight * shoulderHeight, shoulderHeight * rumpHeight, rumpHeight * rumpHeight]
    //let coef = [0., 0., 0., 0.00543865, 0., 0., 0.00705051, 0., 0.00632291]
    //const intercept = -43.40788277320177

    // no intercept no shoulders poly features
    //let features = [bodyLength, rumpHeight, bodyLength * bodyLength, bodyLength * rumpHeight, rumpHeight * rumpHeight]
    //let coef = [-2.12596237, 1.19998256, 0.04949122, -0.06613749, 0.0395875]
    //const intercept = 0

    // no intercept 
    //let features = [bodyLength, shoulderHeight, rumpHeight]
    //let coef = [1.05934156, -0.11443692, -0.2678467]
    //const intercept = 0

    // no intercept no shoulders
    //let features = [bodyLength, rumpHeight]
    //let coef = [1.04816933, -0.36693258]
    //const intercept = 0

    // standard
    //let features = [bodyLength, shoulderHeight, rumpHeight]
    //let coef = [0.78557921, 0.88108601, 0.98848215]
    //const intercept = -136.81080809539807

    // no shoulders
    let features = [bodyLength, rumpHeight]
    let coef = [0.88075418, 1.64152699]
    const intercept = -129.71623666348054

    const value = features.map((v, i) => v * coef[i]).reduce((p, c) => p + c, 0)
    const weight = value + intercept

    return weight
  }
}
