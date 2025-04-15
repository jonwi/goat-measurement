import * as tf from '@tensorflow/tfjs'

export class YOLO {

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
  // 2d tensor with [maskHeight x maskWidth]
  mask: tf.Tensor2D | null = null

  async loadModel() {
    const startTime = new Date().getTime()

    this.model = await tf.loadGraphModel('/model/model.json')
      ;[this.inputHeight, this.inputWidth] = [640, 640]
      ;[this.xyxy, this.classes, this.numMasks] = [4, 1, 32]
      ;[this.maskWidth, this.maskHeight] = [160, 160]

    // cold start to compile the whole network may take a second
    this.model.execute(tf.zeros([1, this.inputHeight, this.inputWidth, 3]))
    console.log("model loaded in: ", new Date().getTime() - startTime)
    console.log(tf.getBackend())
  }

  async predict(imageEl: HTMLImageElement | HTMLVideoElement, canvas: HTMLCanvasElement | null = null) {
    const startTime = new Date().getTime()
    this.preprocess(imageEl)
    this.runInference()
    this.postprocess()
    await this.draw(imageEl, canvas)
    console.log("predict time: ", new Date().getTime() - startTime)
    return this.mask
  }

  preprocess(imageEl: HTMLImageElement | ImageData | HTMLVideoElement) {
    const startTime = new Date().getTime()
    if (this.input) {
      this.input.dispose()
      this.input = null
    }

    this.input =
      tf.tidy(() => {
        const image = tf.browser.fromPixels(imageEl)
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

        console.log("image sizes:", this.originalHeight, this.originalWidth, this.scaledOriginalHeight, this.scaledOriginalWidth, scalingFactor)
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
    console.log("preprocess time: ", new Date().getTime() - startTime)
  }

  runInference() {
    const startTime = new Date().getTime()
    if (this.output) {
      this.output[0].dispose()
      this.output[1].dispose()
      this.output = null
    }
    this.output = this.model.execute(this.input!)
    console.log("inference time: ", new Date().getTime() - startTime)
  }

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
      console.log(`max confidence is only ${maxConfidence}, therefore the will be no detection.`)
      return
    }

    this.mask = tf.tidy(() => {
      const maskCoeffs = detections.slice([this.xyxy! + this.classes!, maxIndex], [this.numMasks!, 1]).squeeze()
      this.box = new Box(detections.slice([0, maxIndex], [this.xyxy!, 1]).dataSync<any>())
      const mx = Math.max(this.scaledOriginalHeight!, this.scaledOriginalWidth!)
      const heightStart = this.scaledOriginalHeight == mx ? 0 : ((mx - this.scaledOriginalHeight!) / 2)
      const widthStart = this.scaledOriginalWidth == mx ? 0 : ((mx - this.scaledOriginalWidth!) / 2)

      // Reconstruct mask
      let mask: tf.Tensor2D =
        segmentationMap
          .matMul(maskCoeffs.expandDims(1))  // Shape [160, 160, 1]
          .squeeze()
          .expandDims(-1)
          .resizeBilinear([this.inputHeight!, this.inputWidth!], false, true)
          .squeeze()
          .slice([this.box!.topY(), this.box!.topX()], [this.box!.height(), this.box!.width()])
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

    console.log("postprocess time: ", new Date().getTime() - startTime)
    return this.mask
  }


  async draw(image: HTMLImageElement | HTMLVideoElement, canvas: HTMLCanvasElement | null) {
    if (this.mask == null || this.box == null || this.inputHeight == null || this.inputWidth == null || this.originalHeight == null || this.originalWidth == null || this.scaledOriginalHeight == null || this.scaledOriginalWidth == null)
      return

    let startTime = new Date().getTime()

    const newOverlay = tf.tidy(() => {
      let expandedMask = this.mask!.expandDims(-1)
      let overlay = tf.zeros<tf.Rank.R3>([this.scaledOriginalHeight!, this.scaledOriginalWidth!, 4], 'int32') // RGBA
      return overlay.where<tf.Tensor3D>(expandedMask.less(1), tf.tensor1d([128, 0, 0, 150], 'int32'))
    })

    console.log("scaled", this.scaledOriginalWidth, this.scaledOriginalHeight)

    let arr = await tf.browser.toPixels(newOverlay)
    newOverlay.dispose()
    let tempCanvas = document.createElement("canvas")
    tempCanvas.width = this.scaledOriginalWidth
    tempCanvas.height = this.scaledOriginalHeight
    let tmpCtx = tempCanvas.getContext('2d')!
    tmpCtx.putImageData(new ImageData(arr, this.scaledOriginalWidth, this.scaledOriginalHeight), 0, 0)
    if (canvas) {
      canvas.height = this.scaledOriginalHeight
      canvas.width = this.scaledOriginalWidth
      canvas.style.height = `${this.scaledOriginalHeight}px`
      canvas.style.width = `${this.scaledOriginalWidth}px`
      let ctx = canvas.getContext('2d')!
      ctx.drawImage(image, 0, 0, this.scaledOriginalWidth, this.scaledOriginalHeight)
      ctx.drawImage(tempCanvas, 0, 0, this.scaledOriginalWidth, this.scaledOriginalHeight)
      ctx.rect(this.box.topX(), this.box.topY(), this.box.w, this.box.h)
      ctx.stroke()
    }
    console.log("draw time: ", new Date().getTime() - startTime)
  }

}

class Box {
  x: number
  y: number
  w: number
  h: number

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

