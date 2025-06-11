import * as tf from '@tensorflow/tfjs'
import { GoatPredictor } from './goat-predictor'
import '@tensorflow/tfjs-backend-webgpu'

export class YOLO implements GoatPredictor {
  debug = false
  originalWidth: number | null = null
  originalHeight: number | null = null
  scaledOriginalWidth: number | null = null
  scaledOriginalHeight: number | null = null
  inputWidth: number | null = null
  inputHeight: number | null = null
  xyxy: number | null = null
  classes: number | null = null
  numMasks: number | null = null
  maskWidth: number | null = null
  maskHeight: number | null = null

  box: Box | null = null

  model: any | null = null
  output: [tf.Tensor, tf.Tensor] | null = null
  input: tf.Tensor | null = null
  inputImage: tf.Tensor3D | null = null
  // 2d tensor with [maskHeight x maskWidth]
  mask: tf.Tensor2D | null = null

  async loadModel() {
    const startTime = new Date().getTime()

    await tf.setBackend("webgpu")
    this.model = await tf.loadGraphModel(`${import.meta.env.BASE_URL}model/model.json`)
      ;[this.inputHeight, this.inputWidth] = [640, 640]
      ;[this.xyxy, this.classes, this.numMasks] = [4, 1, 32]
      ;[this.maskWidth, this.maskHeight] = [160, 160]

    // cold start to compile the whole network may take a second
    this.model.execute(tf.zeros([1, this.inputHeight, this.inputWidth, 3]))
    if (this.debug) {
      console.log("model loaded in: ", new Date().getTime() - startTime)
      console.log(tf.getBackend())
    }
  }

  /**
   * Runs a goat detection model to detect a goat in imageEl. Outputs images to imageCanvas and canvas
   *
   * @param imageEl The source where an image is captured from
   * @param imageCanvas will draw the source image to this canvas
   * @param canvas will draw the result of the detection to this canvas
   * @returns a binary mask as a tf.Tensor2D or null if no detection was made, and a bounding box of the detection
  */
  async predict(imageEl: HTMLImageElement | HTMLVideoElement, imageCanvas: HTMLCanvasElement, canvas: HTMLCanvasElement | null = null): Promise<[tf.Tensor2D | null, Box | null]> {
    const startTime = new Date().getTime()
    this.preprocess(imageEl, imageCanvas, canvas)
    this.runInference()
    this.postprocess()
    await this.draw(canvas)
    if (this.debug) console.log("predict time: ", new Date().getTime() - startTime)
    return [this.mask, this.box]
  }

  /**
   * Preprocess to get the source image from imageEl and draw it to imageCanvas and canvas
   * This will set multiple class fields like inputImage and input
   *
   * @param imageEl source of the image to detect goats
   * @param imageCanvas destination for the source image to draw to
   * @param canvas result canvas
  */
  preprocess(imageEl: HTMLImageElement | ImageData | HTMLVideoElement, imageCanvas: HTMLCanvasElement, canvas: HTMLCanvasElement | null) {
    const startTime = new Date().getTime()
    if (this.input) {
      this.input.dispose()
      this.input = null
    }

    if (this.inputImage) {
      this.inputImage.dispose()
      this.inputImage = null
    }
    this.inputImage = tf.browser.fromPixels(imageEl)
    imageCanvas.height = this.inputImage.shape[0]
    imageCanvas.width = this.inputImage.shape[1]
    tf.browser.toPixels(this.inputImage, imageCanvas)
    if (canvas) {
      canvas.height = this.inputImage.shape[0]
      canvas.width = this.inputImage.shape[1]
      tf.browser.toPixels(this.inputImage, canvas)
    }

    this.input =
      tf.tidy(() => {
        const image = this.inputImage!
        this.originalHeight = image.shape[0]
        this.originalWidth = image.shape[1]
        let scalingFactor = 1
        if (this.originalHeight > this.originalWidth) {
          scalingFactor = this.inputHeight! / this.originalHeight
        } else {
          scalingFactor = this.inputWidth! / this.originalWidth
        }
        this.scaledOriginalHeight = this.originalHeight * scalingFactor
        this.scaledOriginalWidth = this.originalWidth * scalingFactor
        const mx = Math.max(this.scaledOriginalWidth, this.scaledOriginalHeight)

        if (this.debug) console.log("image sizes:", this.originalHeight, this.originalWidth, this.scaledOriginalHeight, this.scaledOriginalWidth, scalingFactor)
        return image
          .resizeBilinear([this.scaledOriginalHeight, this.scaledOriginalWidth])
          .pad([
            [
              this.scaledOriginalHeight == mx ? 0 : (mx - this.scaledOriginalHeight!) / 2,
              this.scaledOriginalHeight == mx ? 0 : (mx - this.scaledOriginalHeight!) / 2
            ],
            [
              this.scaledOriginalWidth == mx ? 0 : (mx - this.scaledOriginalWidth!) / 2,
              this.scaledOriginalWidth == mx ? 0 : (mx - this.scaledOriginalWidth!) / 2
            ],
            [0, 0]
          ])
          .resizeNearestNeighbor([this.inputWidth!, this.inputHeight!])
          .expandDims(0)
          .toFloat().div(tf.scalar(255))
      })
    if (this.debug) console.log("preprocess time: ", new Date().getTime() - startTime)
  }

  /**
   * Runs the model on this.input and writes to this.output
  */
  runInference() {
    const startTime = new Date().getTime()
    if (this.output) {
      this.output[0].dispose()
      this.output[1].dispose()
      this.output = null
    }
    this.output = this.model.execute(this.input!)
    if (this.debug) console.log("inference time: ", new Date().getTime() - startTime)
  }

  /**
   * Processes the output of the model and creates the binary mask
   *
   * @returns a binary tf.Tensor2D or null if no detection of quality was made
  */
  postprocess() {
    const startTime = new Date().getTime()
    const [detectionTensor, segmentationTensor] = this.output!

    if (this.mask) {
      this.mask.dispose()
      this.mask = null
    }

    const detections: tf.Tensor2D = detectionTensor.squeeze() // Shape: [37, 8400]
    const segmentationMap: tf.Tensor3D = segmentationTensor.squeeze() // Shape: [160, 160, 32]

    const confidences = detections.slice([this.xyxy!, 0], [this.classes!, -1]) // Confidence scores for each detection
    // this sync is a major bottleneck but might also not make an impact at all when resolved
    const maxIndex = confidences.argMax(1).dataSync()[0]
    const maxConfidence = confidences.gather(maxIndex, 1).dataSync()[0]

    if (maxConfidence < 0.85) {
      if (this.debug) console.log(`max confidence is only ${maxConfidence}, therefore the will be no detection.`)
      return
    }
    if (this.debug) console.log("maxConfidence", maxConfidence)

    this.mask = tf.tidy(() => {
      const maskCoeffs: tf.Tensor2D = detections.slice([this.xyxy! + this.classes!, maxIndex], [this.numMasks!, 1]).squeeze()
      this.box = new Box(detections.slice([0, maxIndex], [this.xyxy!, 1]).dataSync<any>())
      const mx = Math.max(this.scaledOriginalHeight!, this.scaledOriginalWidth!)
      const heightStart = this.scaledOriginalHeight == mx ? 0 : ((mx - this.scaledOriginalHeight!) / 2)
      const widthStart = this.scaledOriginalWidth == mx ? 0 : ((mx - this.scaledOriginalWidth!) / 2)

      // @ts-ignore
      let mask: tf.Tensor2D =
        segmentationMap
          .matMul(maskCoeffs.expandDims(1))  // Shape [160, 160, 1]
          .squeeze()
          .expandDims(-1)
          .resizeBilinear([this.inputHeight!, this.inputWidth!], false, true)
          .squeeze()
          .slice(
            [this.box!.topY(), this.box!.topX()],
            [Math.min(this.box!.height(), this.inputHeight!),
            Math.min(this.box!.width(), this.inputWidth!),
            ])
          .pad([
            [this.box!.topY(), this.inputHeight! - this.box!.topY() - this.box!.height()],
            [this.box!.topX(), this.inputWidth! - this.box!.topX() - this.box!.width()],
          ]) // Shape [inputHeight, intputWidth]
          .slice([heightStart, widthStart], [this.scaledOriginalHeight!, this.scaledOriginalWidth!])

      let binary = tf.where<tf.Tensor2D>(mask.greater(0), tf.ones(mask.shape, 'int32'), tf.zeros(mask.shape, 'int32'))
      if (this.scaledOriginalWidth != mx) {
        this.box.x -= (mx - this.scaledOriginalWidth!) / 2
      }
      if (this.scaledOriginalHeight != mx) {
        this.box.y -= (mx - this.scaledOriginalHeight!) / 2
      }

      return binary
    })

    detections.dispose()
    segmentationMap.dispose()
    confidences.dispose()

    if (this.debug) console.log("postprocess time: ", new Date().getTime() - startTime)
    return this.mask
  }


  /**
   * Draws the result of a inference on the canvas element
   *
   * @param canvas target for drawing the result of the inference
  */
  async draw(canvas: HTMLCanvasElement | null) {
    if (this.mask == null || this.box == null || this.inputHeight == null || this.inputWidth == null || this.originalHeight == null || this.originalWidth == null || this.scaledOriginalHeight == null || this.scaledOriginalWidth == null || this.inputImage == null)
      return

    let startTime = new Date().getTime()

    const newOverlay = tf.tidy(() => {
      let expandedMask = this.mask!.expandDims(-1)
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
      // ctx.drawImage(image, 0, 0, this.scaledOriginalWidth, this.scaledOriginalHeight)
      tf.browser.draw(this.inputImage, canvas)
      ctx.drawImage(tempCanvas, 0, 0, this.scaledOriginalWidth, this.scaledOriginalHeight)
      ctx.rect(this.box.topX(), this.box.topY(), this.box.w, this.box.h)
      ctx.stroke()
    }
    if (this.debug) console.log("draw time: ", new Date().getTime() - startTime)
  }

}

/**
 * Bounding Box representation of xywh where xy is the center point of the bounding box
  */
export class Box {
  x: number
  y: number
  w: number
  h: number

  /**
   * @param arr array with xywh format where xy is the center point of the bounding box
  */
  constructor(arr: Float32Array<ArrayBufferLike>) {
    this.x = arr[0]
    this.y = arr[1]
    this.w = arr[2]
    this.h = arr[3]
  }

  topX() {
    return Math.ceil(this.x - this.w / 2)
  }
  topY() {
    return Math.ceil(this.y - this.h / 2)
  }
  width() {
    return Math.ceil(this.w)
  }
  height() {
    return Math.ceil(this.h)
  }
}

