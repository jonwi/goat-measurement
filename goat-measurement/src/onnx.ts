import { InferenceSession } from 'onnxruntime-web'
import * as ort from 'onnxruntime-web/all'
import { GoatPredictor } from './goat-predictor'
import * as tf from "@tensorflow/tfjs"
import { Box } from './yolotfjs'

export class ONNX implements GoatPredictor {
  debug = true
  inputHeight = 640
  inputWidth = 640
  xyxy = 4
  classes = 1
  numMasks = 32
  maskWidth = 160
  maskHeight = 160
  originalWidth = 640
  originalHeight = 640
  scaledOriginalWidth = 640
  scaledOriginalHeight = 640

  session: InferenceSession | null = null

  constructor() {
    ort.env.debug = true
  }

  async loadModel() {
    this.session = await ort.InferenceSession.create("best.onnx", { executionProviders: ["webgpu"] })
    console.log(this.session)
  }

  async predict(img: HTMLImageElement | HTMLVideoElement, imageCanvas: HTMLCanvasElement, debugCanvas: HTMLCanvasElement): Promise<[tf.Tensor2D | null, Box | null]> {
    const original: tf.Tensor3D = tf.browser.fromPixels(img).toFloat().div(tf.scalar(255.0, "float32"))
    console.log(original)
    const pixelArray = new Float32Array(original.dataSync())
    const data = new ort.Tensor("float32", pixelArray, [1, 3, 640, 640])
    const result = await this.session!.run({ [this.session!.inputNames[0]]: data })
    console.log(result)

    const [detectionTensor, segmentationTensor] = this.convert(result)

    const res = this.postprocess(detectionTensor, segmentationTensor)
    if (res == null) {
      return [null, null]
    }
    const [mask, box] = res
    console.log(mask)

    if (mask && box) {
      await this.draw(imageCanvas, mask, box, original)
    }
    original.dispose()
    return [null, null]
  }

  convert(result: InferenceSession.OnnxValueMapType) {
    const detectionArray = result[this.session!.outputNames[0]]
    const segmentationArray = result[this.session!.outputNames[1]]
    console.log()
    //@ts-ignore
    const detectionTensor = tf.tensor(detectionArray.cpuData, detectionArray.dims)
    //@ts-ignore
    const segmentationTensor = tf.tensor(segmentationArray.cpuData, segmentationArray.dims)
    console.log(detectionTensor)
    console.log(segmentationTensor)
    return [detectionTensor, segmentationTensor]
  }

  /**
   * Processes the output of the model and creates the binary mask
   *
   * @returns a binary tf.Tensor2D or null if no detection of quality was made
  */
  postprocess(detectionTensor: tf.Tensor, segmentationTensor: tf.Tensor): [tf.Tensor2D | null, Box | null] {
    const startTime = new Date().getTime()

    const detections: tf.Tensor2D = detectionTensor.squeeze() // Shape: [37, 8400]
    console.log(detections)
    const segmentationMap: tf.Tensor3D = segmentationTensor.squeeze().transpose([1, 2, 0]) // Shape: [160, 160, 32]
    console.log(segmentationMap)

    const confidences = detections.slice([this.xyxy!, 0], [this.classes!, -1]) // Confidence scores for each detection
    // this sync is a major bottleneck but might also not make an impact at all when resolved
    let maxIndex = confidences.argMax(1).dataSync()[0]
    console.log(maxIndex)
    maxIndex = 8189
    const maxConfidence = confidences.gather(maxIndex, 1).dataSync()[0]

    if (maxConfidence > 0.85) {
      if (this.debug) console.log(`max confidence is only ${maxConfidence}, therefore there will be no detection.`)
      return [null, null]
    }
    if (this.debug) console.log("maxConfidence", maxConfidence)

    let box: Box | null = null

    const mask = tf.tidy(() => {
      const maskCoeffs: tf.Tensor2D = detections.slice([this.xyxy! + this.classes!, maxIndex], [this.numMasks!, 1]).squeeze()
      box = new Box(detections.slice([0, maxIndex], [this.xyxy!, 1]).dataSync<any>())
      const mx = Math.max(this.scaledOriginalHeight!, this.scaledOriginalWidth!)
      const heightStart = this.scaledOriginalHeight == mx ? 0 : ((mx - this.scaledOriginalHeight!) / 2)
      const widthStart = this.scaledOriginalWidth == mx ? 0 : ((mx - this.scaledOriginalWidth!) / 2)
      console.log("heightStart", heightStart, "widthStart", widthStart)
      console.log(box.topX(), box.topY(), box.width(), box.height())

      // @ts-ignore
      let mask: tf.Tensor2D =
        segmentationMap
          .matMul(maskCoeffs.expandDims(1))  // Shape [160, 160, 1]
          .squeeze()
          .expandDims(-1)
          .resizeBilinear([this.inputHeight!, this.inputWidth!], false, true)
          .squeeze()
          .slice(
            [box.topY(), box.topX()],
            [Math.min(box.height(), this.inputHeight! - box.topY()),
            Math.min(box.width(), this.inputWidth! - box.topX()),
            ])
          .pad([
            [box.topY(), this.inputHeight! - box.topY() - box.height() + 1],
            [box.topX(), this.inputWidth! - box.topX() - box.width() + 1],
          ]) // Shape [inputHeight, intputWidth]
          .slice([heightStart, widthStart], [this.scaledOriginalHeight!, this.scaledOriginalWidth!])

      let binary = tf.where<tf.Tensor2D>(mask.greater(0), tf.ones(mask.shape, 'int32'), tf.zeros(mask.shape, 'int32'))
      if (this.scaledOriginalWidth != mx) {
        box.x -= (mx - this.scaledOriginalWidth!) / 2
      }
      if (this.scaledOriginalHeight != mx) {
        box.y -= (mx - this.scaledOriginalHeight!) / 2
      }

      return binary
    })

    detections.dispose()
    segmentationMap.dispose()
    confidences.dispose()

    if (this.debug) console.log("postprocess time: ", new Date().getTime() - startTime)
    return [mask, box]
  }

  /**
   * Draws the result of a inference on the canvas element
   *
   * @param canvas target for drawing the result of the inference
  */
  async draw(canvas: HTMLCanvasElement | null, mask: tf.Tensor2D, box: Box, original: tf.Tensor3D) {

    let startTime = new Date().getTime()

    const newOverlay = tf.tidy(() => {
      let expandedMask = mask!.expandDims(-1)
      let overlay = tf.zeros<tf.Rank.R3>([this.scaledOriginalHeight!, this.scaledOriginalWidth!, 4], 'int32') // RGBA
      return overlay.where<tf.Tensor3D>(expandedMask.less(1), tf.tensor1d([128, 0, 0, 150], 'int32'))
    })

    if (this.debug) console.log("scaled", this.scaledOriginalWidth, this.scaledOriginalHeight)

    if (canvas) {
      let arr = await tf.browser.toPixels(newOverlay)
      newOverlay.dispose()
      let tempCanvas = document.createElement("canvas")
      tempCanvas.width = this.scaledOriginalWidth
      tempCanvas.height = this.scaledOriginalHeight
      let tmpCtx = tempCanvas.getContext('2d')!
      tmpCtx.putImageData(new ImageData(arr, this.scaledOriginalWidth, this.scaledOriginalHeight), 0, 0)
      canvas.height = this.scaledOriginalHeight
      canvas.width = this.scaledOriginalWidth
      canvas.style.height = `${this.scaledOriginalHeight}px`
      canvas.style.width = `${this.scaledOriginalWidth}px`
      let ctx = canvas.getContext('2d')!
      ctx.putImageData(new ImageData(await tf.browser.toPixels(original), this.scaledOriginalWidth, this.scaledOriginalHeight), 0, 0)
      ctx.drawImage(tempCanvas, 0, 0, this.scaledOriginalWidth, this.scaledOriginalHeight)
      ctx.rect(box.topX(), box.topY(), box.w, box.h)
      ctx.stroke()
    }
    if (this.debug) console.log("draw time: ", new Date().getTime() - startTime)
  }

}
