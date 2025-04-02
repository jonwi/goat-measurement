import * as tf from '@tensorflow/tfjs';

export class YOLO {

  canvas = document.createElement('canvas');
  originalWidth: number | null = null;
  originalHeight: number | null = null;
  inputWidth: number | null = null;
  inputHeight: number | null = null;
  xyxy: number | null = null;
  classes: number | null = null;
  numMasks: number | null = null;
  maskWidth: number | null = null;
  maskHeight: number | null = null;

  model: any | null = null;
  output: [tf.Tensor, tf.Tensor] | null = null;
  input: tf.Tensor | null = null;
  mask: tf.Tensor2D | null = null;

  async loadModel() {
    const startTime = new Date().getTime();

    this.model = await tf.loadGraphModel('public/model/model.json');
    [this.inputHeight, this.inputWidth] = [640, 640];
    [this.xyxy, this.classes, this.numMasks] = [4, 1, 32];
    [this.maskWidth, this.maskHeight] = [160, 160];
    this.canvas.width = this.inputWidth
    this.canvas.height = this.inputHeight

    // remove this in production
    document.querySelector("#test")?.appendChild(this.canvas)

    // cold start to compile the whole network may take a second
    this.model.execute(tf.zeros([1, this.inputHeight, this.inputWidth, 3]));
    console.log("model loaded in: ", new Date().getTime() - startTime)
    console.log(tf.getBackend())
  }

  async predict(imageEl: HTMLImageElement) {
    const startTime = new Date().getTime()
    this.preprocess(imageEl)
    this.runInference()
    this.postprocess()
    await this.draw(imageEl)
    console.log("predict time: ", new Date().getTime() - startTime)
    return this.mask
  }

  preprocess(imageEl: HTMLImageElement | ImageData) {
    const startTime = new Date().getTime()

    this.originalWidth = imageEl.width
    this.originalHeight = imageEl.height
    let mx = Math.max(this.originalWidth, this.originalHeight)

    if (this.input) {
      this.input.dispose()
    }

    this.input =
      tf.tidy(() => {
        return tf.browser.fromPixels(imageEl)
          .pad([
            [
              this.originalHeight == mx ? 0 : (mx - this.originalHeight!) / 2,
              this.originalHeight == mx ? 0 : (mx - this.originalHeight!) / 2
            ],
            [
              this.originalWidth == mx ? 0 : (mx - this.originalWidth!) / 2,
              this.originalWidth == mx ? 0 : (mx - this.originalWidth!) / 2
            ],
            [0, 0]
          ])
          .resizeNearestNeighbor([this.inputWidth!, this.inputHeight!])
          .expandDims(0)
          .toFloat().div(tf.scalar(255));
      })
    console.log("preprocess time: ", new Date().getTime() - startTime)
  }

  runInference() {
    const startTime = new Date().getTime()
    if (this.output) {
      this.output[0].dispose()
      this.output[1].dispose()
    }
    this.output = this.model.execute(this.input!);
    console.log("inference time: ", new Date().getTime() - startTime)
  }

  postprocess() {
    const startTime = new Date().getTime()
    const [detectionTensor, segmentationTensor] = this.output!

    if (this.mask) {
      this.mask.dispose()
    }

    this.mask = tf.tidy(() => {
      const detections: tf.Tensor2D = detectionTensor.squeeze(); // Shape: [37, 8400]
      const segmentationMap: tf.Tensor3D = segmentationTensor.squeeze(); // Shape: [160, 160, 32]

      const sliceTime = new Date().getTime()
      const confidences = detections.slice([this.xyxy!, 0], [this.classes!, -1]); // Confidence scores for each detection
      // this sync is a major bottleneck but might also not make an impact at all when resolved
      const maxIndex = confidences.argMax(1).dataSync()[0]
      const maskCoeffs = detections.slice([this.xyxy! + this.classes!, maxIndex], [this.numMasks!, 1]).squeeze()
      console.log("slice time: ", new Date().getTime() - sliceTime)

      const multTime = new Date().getTime()
      // Reconstruct mask
      let mask: tf.Tensor2D = tf.tidy(() => {
        return segmentationMap
          .matMul(maskCoeffs.expandDims(1))  // Shape [160, 160, 1]
          //.sigmoid() // maybe not needed
          //.greater(tf.scalar(0.9)) // maybe not needed
          .toFloat()
          .squeeze();
      });
      console.log("mult time: ", new Date().getTime() - multTime)
      return mask
    })

    console.log("postprocess time: ", new Date().getTime() - startTime)
    return this.mask;
  }

  async draw(image: HTMLImageElement) {
    let startTime = new Date().getTime()

    const newOverlay = tf.tidy(() => {
      let expandedMask = this.mask!.expandDims(-1)
      let resizedMask = expandedMask.resizeBilinear([this.inputHeight!, this.inputWidth!], false, true)
      let overlay = tf.zeros<tf.Rank.R3>([this.inputWidth!, this.inputHeight!, 4], 'int32') // RGBA
      return overlay.where<tf.Tensor3D>(resizedMask.less(1), tf.tensor1d([128, 0, 0, 150], 'int32'))
    })

    let arr = await tf.browser.toPixels(newOverlay);
    newOverlay.dispose()
    let tempCanvas = document.createElement("canvas")
    tempCanvas.width = this.inputWidth!
    tempCanvas.height = this.inputHeight!
    let tmpCtx = tempCanvas.getContext('2d')!
    tmpCtx.putImageData(new ImageData(arr, this.inputWidth!, this.inputHeight!), 0, 0)
    let ctx = this.canvas.getContext('2d')!
    ctx.drawImage(image, 80, 0, this.originalWidth!, this.originalHeight!)
    ctx.drawImage(tempCanvas, 0, 0, this.inputWidth!, this.inputHeight!)
    console.log("draw time: ", new Date().getTime() - startTime)
  }

}

