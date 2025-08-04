import * as tf from '@tensorflow/tfjs'
import { Box } from './yolotfjs'

/**
 * Calculates the longest line of consecutive ones
 *
 * @param t tensor
 * @returns begin, end, length
 */
async function binaryRle(t: tf.Tensor1D) {
  let l = t.shape[0]
  let indices = tf.range(0, t.shape[0], 1, "int32")

  let unequal = t.slice(1, l - 1).notEqual(t.slice(0, l - 1))
    .concat(tf.tensor1d([true]))
  let nonzero = await tf.booleanMaskAsync(indices, unequal)

  let diff = tf.ones([1]).concat(nonzero.slice(1, nonzero.shape[0] - 1).sub(nonzero.slice(0, nonzero.shape[0] - 1)))
  let ones = t.greater(0).gather(nonzero)
  let possible = diff.where(ones, tf.tensor1d([0]))
  let maxLengthIndex = possible.argMax()
  if (maxLengthIndex.dataSync()[0] > 0) {
    let length = possible.max().data()
    let end = nonzero.gather(maxLengthIndex).data()
    // underflow possible
    let begin = nonzero.gather(maxLengthIndex.sub(tf.tensor1d([1], "int32"))).data()

    let res = await Promise.all([begin, end, length])
    let unwrap = res.map(r => r[0])
    unwrap[0] += 1
    return unwrap
  }
  return [-1, -1, -1]
}

/**
 * Calculates the longest line of consecutive ones
 * This does not use any gpu acceleration
 *
 * @param t tensor
 * @returns begin, end, length
 */
function syncBinaryRle(t: tf.Tensor1D) {
  const data = t.dataSync()
  let length = 0
  let currentStartIndex = 0
  let maxLength = 0
  let startIndex = 0
  let endIndex = 0

  for (let i = 0; i < data.length; i++) {
    if (data[i] === 1) {
      if (length === 0) {
        currentStartIndex = i
      }
      length++
      if (length > maxLength) {
        maxLength = length
        startIndex = currentStartIndex
        endIndex = i
      }
    } else {
      length = 0
    }
  }
  return [startIndex, endIndex, maxLength]
}


/**
 * Extracts measurements in pixels from the mask
 *
 * @param mask a masked image of the goat
 * @param box a bounding box that is the bounding of the detection
 * @param canvas optional canvas where findings are drawn
 * @param direction the direction the goat is facing
 * @returns bodyLength, shoulderHeight, rumpHeight, bodyHeight in pixels
 */
export async function bodyMeasurement(mask: tf.Tensor2D, box: Box, canvas: HTMLCanvasElement | null = null, direction: "left" | "right") {
  // mask is hxw 640x640
  let detection = mask.slice([box.topY(), box.topX()], [box.height(), box.width()])
  let height = box.height()
  const range = tf.range(1, box.width() + 1, 1)
  const filledColumns = detection.sum(0).toBool().toInt()
  const xStart = filledColumns.mul(range.reverse()).argMax()
  const xEnd = filledColumns.mul(range).argMax()
  detection = detection.slice([0, xStart.dataSync()[0]], [height, xEnd.sub(xStart).dataSync()[0]])
  let width = detection.shape[1]
  const x = xStart.dataSync()[0] + box.topX()

  const colRange = tf.range(1, height + 1, 1)
  let lastIndices = detection.mul(colRange.expandDims(-1)).argMax(0)
  const headWidth = Math.floor(width * 0.2) // this might need more finetuning
  const shoulderWidth = Math.floor(width * 0.4)
  const rumpWidth = Math.floor(width * 0.4)
  const tailWidth = Math.floor(width * 0.1)
  let [headStart, shoulderSideStart, rumpSideStart, tailStart] = [0, 0, 0, 0]
  if (direction == "left") {
    shoulderSideStart = headWidth
    rumpSideStart = headWidth + shoulderWidth
    headStart = 0
    tailStart = width - tailWidth
  } else {
    shoulderSideStart = rumpWidth
    rumpSideStart = tailStart
    headStart = rumpWidth + shoulderWidth
    tailStart = 0
  }
  drawRect(canvas, x + headStart, box.topY(), headWidth, box.height(), "#3022dd24")
  drawRect(canvas, x + shoulderSideStart, box.topY(), shoulderWidth, box.height(), "#d52a3d24")
  drawRect(canvas, x + rumpSideStart, box.topY(), rumpWidth, box.height(), "#55d32c24")
  drawRect(canvas, x + tailStart, box.topY(), tailWidth, box.height(), "#ffcd4324")

  const shoulderSide = lastIndices.slice(shoulderSideStart, shoulderWidth)
  const lowestShoulderIndex = shoulderSide.argMax()
  drawVerticalLine(canvas, lowestShoulderIndex.dataSync()[0] + x + shoulderSideStart, "green")

  const rumpSide = lastIndices.slice(rumpSideStart, rumpWidth)
  const lowestRumpIndex = rumpSide.argMax()
  const firstIndex = detection.mul(colRange.reverse().expandDims(1)).argMax<tf.Tensor1D>(0).slice(rumpSideStart, rumpWidth)
  let hill
  if (direction == "left") {
    hill = firstHill(detection.mul(colRange.reverse().expandDims(1)).argMax<tf.Tensor1D>(0).slice(rumpSideStart, rumpWidth))
  } else {
    hill = tf.scalar(rumpWidth).sub(firstHill(detection.mul(colRange.reverse().expandDims(1)).argMax<tf.Tensor1D>(0).slice(rumpSideStart, rumpWidth).reverse()))
  }
  drawVerticalLine(canvas, hill.dataSync()[0] + x + rumpSideStart, "grey")
  drawVerticalLine(canvas, lowestRumpIndex.dataSync()[0] + x + rumpSideStart, "red")

  const newMiddle = lowestShoulderIndex
    .add(tf.scalar(shoulderSideStart, "int32"))
    .add(lowestRumpIndex.add(tf.scalar(rumpSideStart, "int32")))
    .div(tf.scalar(2, "int32"))
  drawVerticalLine(canvas, newMiddle.dataSync()[0] + x, "purple")

  const middleStart = detection.gather(newMiddle, 1).squeeze().mul(colRange.reverse()).argMax()
  const middleEnd = detection.gather(newMiddle, 1).squeeze().mul(colRange).argMax()
  const middleLength = middleEnd.sub(middleStart)
  draw(canvas, newMiddle.dataSync()[0], middleStart.dataSync()[0], newMiddle.dataSync()[0], middleEnd.dataSync()[0], "yellow", x, box.topY())
  const bodyHeight = middleEnd.sub(middleStart)

  const bodyLengthIndex = middleLength.mul(tf.scalar(0.5)).add(middleStart).cast("int32")
  const bodyLengthLine = detection.gather(bodyLengthIndex, 0).squeeze()
  if (bodyLengthLine.shape.length > 1) {
    console.error("bodyLengthLine is not a Tensor1D")
    console.log(bodyLengthLine)
  }
  // @ts-ignore this is not a Tensor2D no clue why it thinks that we reduce the dimensions by one
  let [bodyLengthStart, bodyLengthEnd, bodyLength] = syncBinaryRle(bodyLengthLine)
  draw(canvas, bodyLengthStart, bodyLengthIndex.dataSync()[0], bodyLengthEnd, bodyLengthIndex.dataSync()[0], "black", x, box.topY())

  const centerWidth = (bodyLengthEnd + bodyLengthStart) / 2 + bodyLengthStart
  const centerHeight = bodyLengthIndex.dataSync()[0]
  const degrees = (direction == "left") ? -15 : 15
  const rotated = rotateDegrees(degrees, [centerHeight, centerWidth], detection.shape)
  const rotatedProjection = rotated.toBool().logicalAnd(detection.toBool()).toInt()
  await drawImage(canvas, rotatedProjection, box)

  const yFront = rotatedProjection
    .cumsum(1)
    .gather(tf.tensor([rotated.shape[1] - 1], [1], "int32"), 1)
    .squeeze()
    .toBool().toInt()
    .mul(tf.range(1, rotated.shape[0] + 1)).toInt()
    .max().dataSync()[0]
  if (direction == "right") {
    const xFront = rotatedProjection.cumsum(0).gather(tf.tensor([rotated.shape[0] - 1], [1], "int32"), 0)
      .squeeze().toBool().toInt().mul(tf.range(1, rotated.shape[1] + 1)).toInt().argMax().dataSync()[0]
    drawCirc(canvas, xFront, yFront, box, "green")
    draw(canvas, xFront, yFront, bodyLengthStart, centerHeight, "orange", box.topX(), box.topY())
    const length = distance(xFront, yFront, bodyLengthStart, centerHeight)
    bodyLength = length
  } else {
    const xFront = rotatedProjection.cumsum(0).gather(tf.tensor([rotated.shape[0] - 1], [1], "int32"), 0)
      .squeeze().toBool().toInt().mul(tf.range(rotated.shape[1] + 1, 1, -1)).toInt().argMax().dataSync()[0]
    drawCirc(canvas, xFront, yFront, box, "green")
    draw(canvas, bodyLengthEnd, centerHeight, xFront, yFront, "orange", box.topX(), box.topY())
    const length = distance(xFront, yFront, bodyLengthEnd, centerHeight)
    bodyLength = length
  }

  const shoulderIndex = lowestShoulderIndex.add(tf.scalar(shoulderSideStart, "int32"))
  const shoulderStart = detection.gather(shoulderIndex, 1).squeeze().mul(colRange.reverse()).argMax(0)
  const shoulderEnd = lastIndices.gather(shoulderIndex)
  draw(canvas, shoulderIndex.dataSync()[0], shoulderStart.dataSync()[0], shoulderIndex.dataSync()[0], shoulderEnd.dataSync()[0], "blue", x, box.topY())
  const shoulderHeight = shoulderEnd.sub(shoulderStart)

  const rumpIndex = lowestRumpIndex.add(tf.scalar(rumpSideStart, "int32"))
  const rumpTop = detection.gather(hill.add(tf.scalar(rumpSideStart, "int32")).toInt(), 1).squeeze().mul(colRange.reverse()).argMax()
  const rumpBottom = lastIndices.gather(rumpIndex)
  draw(canvas, rumpIndex.dataSync()[0], rumpTop.dataSync()[0], rumpIndex.dataSync()[0], rumpBottom.dataSync()[0], "orange", x, box.topY())
  const rumpHeight = rumpBottom.sub(rumpTop)

  return [bodyLength, shoulderHeight.dataSync()[0], rumpHeight.dataSync()[0], bodyHeight.dataSync()[0]]
}

/**
 * calculates euclidian distance between two points
 * @param x1
 * @param x2
 * @param y1
 * @param y2
 * @returns distance
 */
function distance(x1: number, y1: number, x2: number, y2: number) {
  return Math.sqrt(((x1 - x2) * (x1 - x2)) + ((y1 - y2) * (y1 - y2)))
}

/**
 * @param degrees rotation degrees
 * @param center center of rotation
 * @param shape size of the mask
 */
function rotateDegrees(degrees: number, center: [number, number], shape: [number, number]): tf.Tensor2D {
  const [height, width] = center
  let line = tf.zeros(shape, "int32")
  const buffer = line.bufferSync()

  for (let i = 0; i < shape[1]; i++) {
    buffer.set(255, height, i)
  }
  line = buffer.toTensor().toFloat().expandDims(0).expandDims(-1)

  const rotated = tf.image.rotateWithOffset(line, Math.PI * -degrees / 180, 0)

  return rotated.squeeze().toInt()
}

/**
 * @param canvas canvas to draw to
 * @param mask tensor mask to draw
 * @param imageShape size of the original image
 * @param box  detection box
 */
async function drawImage(canvas: HTMLCanvasElement | null, mask: tf.Tensor2D, box: Box, imageShape = [640, 640]) {
  if (canvas) {
    const [height, width] = mask.shape
    const padded = mask.pad([
      [box.topY(), imageShape[0] - box.topY() - height],
      [box.topX(), imageShape[1] - box.topX() - width]
    ])

    const newOverlay = tf.tidy(() => {
      let expandedMask = padded.expandDims(-1)
      let overlay = tf.zeros<tf.Rank.R3>([imageShape[0], imageShape[1], 4], 'int32') // RGBA
      return overlay.where<tf.Tensor3D>(expandedMask.less(1), tf.tensor1d([0, 0, 0, 255], 'int32'))
    })

    let tempCanvas = document.createElement("canvas")
    tempCanvas.width = imageShape[0]
    tempCanvas.height = imageShape[1]
    await tf.browser.toPixels(newOverlay, tempCanvas)
    newOverlay.dispose()
    let ctx = canvas.getContext('2d')!
    ctx.drawImage(tempCanvas, 0, 0, imageShape[0], imageShape[1])
  }
}

/**
 * Returns the first index where the following number is bigger.
 *
 * @param tensor number tensor
 * @returns index of tensor
 */
function firstHill(tensor: tf.Tensor1D) {
  const left = tensor.slice(0, tensor.shape[0] - 1)
  const right = tensor.slice(1, tensor.shape[0] - 1)
  const leftDiff = left.sub(right).less(0).toInt().mul(tf.range(tensor.shape[0], 1)).argMax()
  return leftDiff
}

/**
 * Draws a rectangle in the canvas
 * 
 * @param canvas target to draw to if null nothing happens
 * @param x top left corner x coordinate
 * @param y top left corner y coordinate
 * @param with with of the rect
 * @param height height of the rect
 * @param style style for the rect
 */
function drawRect(canvas: HTMLCanvasElement | null, x: number, y: number, width: number, height: number, style: string) {
  if (canvas == null) return
  let ctx = canvas.getContext("2d")!
  ctx.fillStyle = style
  ctx.fillRect(x, y, width, height)
}

/**
 * Draws a small rect at postion
 * @param canvas target to draw if null nothing happens
 * @param x top left corner z coordinate
 * @param y top left corner y coordinate
 * @param style style for the rect
 */
function drawCirc(canvas: HTMLCanvasElement | null, x: number, y: number, box: Box, style: string) {
  if (canvas == null) return
  let ctx = canvas.getContext("2d")!
  ctx.fillStyle = style
  ctx.fillRect(x + box.topX() - 1, y + box.topY() - 1, 3, 3)
}

/**
 * Draws a vertical line on the canvas. The line will have the full height of the canvas.
 *
 * @param canvas the target to draw to
 * @param x the horizontal coordinate
 * @param style the style of the line
 */
function drawVerticalLine(canvas: HTMLCanvasElement | null, x: number, style: string) {
  if (canvas == null) return
  draw(canvas, x, 0, x, canvas.height, style, 0, 0)
}

/**
 * Draws a line on the canvas with two points.
 * Points are in Image coordinate systems with 0/0 top left
 *
 * @param canvas the target to draw to
 * @param x1 x of the first point
 * @param y1 y of the first point
 * @param x2 x of the second point
 * @param y2 y of the second point
 * @param style style of the line
 * @param xOffset will be applied to both points
 * @param yOffset will be applied to both points
 */
function draw(canvas: HTMLCanvasElement | null, x1: number, y1: number, x2: number, y2: number, style: string, xOffset: number, yOffset: number) {
  if (canvas != null) {
    let ctx = canvas.getContext('2d')!
    ctx.beginPath()
    ctx.moveTo(x1 + xOffset, y1 + yOffset)
    ctx.lineTo(x2 + xOffset, y2 + yOffset)
    ctx.strokeStyle = style
    ctx.stroke()
  }
}

/**
 * Scales pixels to the width of the picture 
 * 
 * @param pixels number of pixels
 * @param convertOptions constant values for conversion
 * @returns scaled pixels
 */
function scaleToWidth(pixels: number, convertOptions: Options) {
  return pixels / convertOptions.mask_shape[1] * convertOptions.orig_shape[1]
}

/**
 * Scales pixels to the height of the picture
 * @param pixels number of pixels
 * @param convertOptions constant values for conversion
 * @returns scaled pixels
 */
function scaleToHeight(pixels: number, convertOptions: Options) {
  return pixels / convertOptions.mask_shape[0] * convertOptions.orig_shape[0] / Math.cos(convertOptions.angle * Math.PI / 180)
}

/**
 * Converts pixels to centimeters
 * @param pixels number of pixels
 * @param convertOptions constants for conversion
 * @returns number of centimeters
 */
function pixelsToCm(pixels: number, convertOptions: Options) {
  return pixels / (convertOptions.calibration * convertOptions.calibration_distance / (convertOptions.distance * 100))
  //return pixels / (calibration(convertOptions.distance) / 100)
}

/**
 * Convert measurements to centimeters.
 * @param body_length length of the body in pixels
 * @param shoulder_height height of shoulder in pixels
 * @param rump_height height of rump in pixels
 * @param bodyHeight height of body in the middle
 * @param convertOptions constants for conversion
 * @returns bodyLength, shoulderHeight, rumpHeight, bodyHeight in centimeters
 */
export function convertToCm(body_length: number, shoulder_height: number, rump_height: number, bodyHeight: number, convertOptions: ConvertOptions) {
  const options = { ...DefaultConvertOptions, ...convertOptions }
  return [
    pixelsToCm(scaleToWidth(body_length, options), options),
    pixelsToCm(scaleToHeight(shoulder_height, options), options),
    pixelsToCm(scaleToHeight(rump_height, options), options),
    pixelsToCm(scaleToHeight(bodyHeight, options), options),
  ]
}

/**
 * Options that can be overwritten from default
 */
type ConvertOptions = {
  distance?: number
  calibration?: number
  calibration_distance?: number
  orig_shape?: number[]
  mask_shape?: number[]
  angle?: number
}

/**
 * Options with conversion constants
 */
type Options = {
  distance: number
  calibration: number
  calibration_distance: number
  orig_shape: number[]
  mask_shape: number[]
  angle: number
}

/**
 * Default options that are used
 */
const DefaultConvertOptions: Options = {
  distance: 1.5,
  calibration: 3.375,
  calibration_distance: 200,
  orig_shape: [640, 640],
  mask_shape: [640, 640],
  angle: 20,
}

function calibration(distance: number) {
  const meters = [0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 10]
  const lengths = [1000, 503, 398, 304, 242, 214, 188, 164, 80]
  let counter = 0
  while (meters[counter] < distance) {
    counter += 1
  }

  const diff = meters[counter] - distance
  const factor = diff / (meters[counter] - meters[counter - 1])
  const factor2 = 1 - factor
  return lengths[counter] * factor2 + lengths[counter - 1] * factor
}
