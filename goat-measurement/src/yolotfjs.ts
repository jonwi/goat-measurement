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
    this.model = await tf.loadGraphModel('public/model/model.json');
    [this.inputHeight, this.inputWidth] = [640, 640];
    [this.xyxy, this.classes, this.numMasks] = [4, 1, 32];
    [this.maskWidth, this.maskHeight] = [160, 160];
    this.canvas.width = this.inputWidth
    this.canvas.height = this.inputHeight
    document.body.appendChild(this.canvas)
  }

  async predict(imageEl: HTMLImageElement) {
    this.preprocess(imageEl)
    this.runInference()
    this.postprocess()
    await this.draw(imageEl)
    return this.mask
  }

  preprocess(imageEl: HTMLImageElement | ImageData) {
    this.originalWidth = imageEl.width
    this.originalHeight = imageEl.height
    let mx = Math.max(this.originalWidth, this.originalHeight)

    this.input = tf.browser.fromPixels(imageEl)
      .pad([
        [
          this.originalHeight == mx ? 0 : (mx - this.originalHeight) / 2,
          this.originalHeight == mx ? 0 : (mx - this.originalHeight) / 2
        ],
        [
          this.originalWidth == mx ? 0 : (mx - this.originalWidth) / 2,
          this.originalWidth == mx ? 0 : (mx - this.originalWidth) / 2
        ],
        [0, 0]
      ])
      .resizeNearestNeighbor([this.inputWidth!, this.inputHeight!])
      .expandDims(0)
      .toFloat().div(tf.scalar(255));
  }

  runInference() {
    this.output = this.model.execute(this.input!);
  }

  postprocess() {
    const [detectionTensor, segmentationTensor] = this.output!

    const detections: tf.Tensor2D = detectionTensor.squeeze(); // Shape: [37, 8400]
    const segmentationMap: tf.Tensor3D = segmentationTensor.squeeze(); // Shape: [160, 160, 32]

    const confidences = detections.slice([this.xyxy!, 0], [this.classes!, -1]); // Confidence scores for each detection
    const maxIndex = confidences.argMax(1).dataSync()[0]
    const maskCoeffs = detections.slice([this.xyxy! + this.classes!, maxIndex], [this.numMasks!, 1]).squeeze()

    // Reconstruct mask
    let mask: tf.Tensor2D = tf.tidy(() => {
      return segmentationMap
        .matMul(maskCoeffs.expandDims(1))  // Shape [160, 160, 1]
        //.sigmoid() // maybe not needed
        //.greater(tf.scalar(0.9)) // maybe not needed
        .toFloat()
        .squeeze();
    });

    this.mask = mask
    return mask;
  }

  async draw(image: HTMLImageElement) {
    let expandedMask = this.mask!.expandDims(-1)
    let resizedMask = expandedMask.resizeBilinear([this.inputHeight!, this.inputWidth!], false, true)
    let overlay = tf.zeros<tf.Rank.R3>([this.inputWidth!, this.inputHeight!, 4], 'int32') // RGBA
    const newOverlay = overlay.where<tf.Tensor3D>(resizedMask.less(1), tf.tensor1d([128, 0, 0, 150], 'int32'))

    let arr = await tf.browser.toPixels(newOverlay);
    let tempCanvas = document.createElement("canvas")
    tempCanvas.width = this.inputWidth!
    tempCanvas.height = this.inputHeight!
    let tmpCtx = tempCanvas.getContext('2d')!
    tmpCtx.putImageData(new ImageData(arr, this.inputWidth!, this.inputHeight!), 0, 0)
    let ctx = this.canvas.getContext('2d')!
    ctx.drawImage(image, 80, 0, this.originalWidth!, this.originalHeight!)
    ctx.drawImage(tempCanvas, 0, 0, this.inputWidth!, this.inputHeight!)
  }

}

