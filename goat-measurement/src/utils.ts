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
  let height = detection.shape[0]
  const sm = detection.sum(0).greater(tf.scalar(0, "int32"))
  const xStart = sm.argMax()
  const xEnd = tf.scalar(detection.shape[1] - 1, "int32").sub(sm.reverse().argMax())
  detection = detection.slice([0, xStart.dataSync()[0]], [height, xEnd.sub(xStart).dataSync()[0]])
  let width = detection.shape[1]
  const x = xStart.dataSync()[0] + box.topX()

  let lastIndices = tf.fill([width], height).sub(detection.reverse(0).argMax(0))
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
    rumpSideStart = 0
    headStart = rumpWidth + shoulderWidth
    tailStart = 0
  }
  drawRect(canvas, x + headStart, box.topY(), headWidth, box.height(), "#3022dd24")
  drawRect(canvas, x + shoulderSideStart, box.topY(), shoulderWidth, box.height(), "#d52a3d24")
  drawRect(canvas, x + rumpSideStart, box.topY(), rumpWidth, box.height(), "#55d32c24")
  drawRect(canvas, x + tailStart, box.topY(), tailWidth, box.height(), "#ffcd4324")

  const shoulderSide = lastIndices.slice(shoulderSideStart, shoulderWidth)
  const lowestShoulderIndex = shoulderSide.argMax()
  drawVerticalLine(canvas, (await lowestShoulderIndex.data())[0] + x + shoulderSideStart, "green")

  const rumpSide = lastIndices.slice(rumpSideStart, rumpWidth)
  const lowestRumpIndex = rumpSide.argMax()
  let hill
  if (direction == "left") {
    hill = firstHill(detection.argMax<tf.Tensor1D>(0).slice(rumpSideStart, rumpWidth))
  } else {
    hill = tf.scalar(rumpWidth).sub(firstHill(detection.argMax<tf.Tensor1D>(0).slice(rumpSideStart, rumpWidth).reverse()))
  }
  drawVerticalLine(canvas, hill.dataSync()[0] + x + rumpSideStart, "grey")
  drawVerticalLine(canvas, (await lowestRumpIndex.data())[0] + x + rumpSideStart, "red")

  const newMiddle = lowestShoulderIndex
    .add(tf.scalar(shoulderSideStart, "int32"))
    .add(lowestRumpIndex.add(tf.scalar(rumpSideStart, "int32")))
    .div(tf.scalar(2, "int32"))
  drawVerticalLine(canvas, newMiddle.dataSync()[0] + x, "purple")

  const middleStart = detection.argMax(0).gather(newMiddle)
  const middleEnd = lastIndices.gather(newMiddle)
  const middleLength = middleEnd.sub(middleStart)
  draw(canvas, newMiddle.dataSync()[0], middleStart.dataSync()[0], newMiddle.dataSync()[0], middleEnd.dataSync()[0], "yellow", x, box.topY())
  const bodyHeight = middleEnd.sub(middleStart)

  const bodyLengthIndex = middleLength.mul(tf.scalar(0.5)).add(middleStart).cast("int32")
  const bodyLengthLine = detection.gather(bodyLengthIndex, 0)
  if (bodyLengthLine.shape.length > 1) {
    console.error("bodyLengthLine is not a Tensor1D")
  }
  // @ts-ignore this is not a Tensor2D no clue why it thinks that we reduce the dimensions by one
  const [bodyLengthStart, bodyLengthEnd, bodyLength] = await binaryRle(bodyLengthLine)
  draw(canvas, bodyLengthStart, bodyLengthIndex.dataSync()[0], bodyLengthEnd, bodyLengthIndex.dataSync()[0], "black", x, box.topY())

  const shoulderIndex = lowestShoulderIndex.add(tf.scalar(shoulderSideStart, "int32"))
  const shoulderStart = detection.argMax(0).gather(shoulderIndex)
  const shoulderEnd = lastIndices.gather(shoulderIndex)
  draw(canvas, shoulderIndex.dataSync()[0], shoulderStart.dataSync()[0], shoulderIndex.dataSync()[0], shoulderEnd.dataSync()[0], "blue", x, box.topY())
  const shoulderHeight = shoulderEnd.sub(shoulderStart)

  const rumpIndex = lowestRumpIndex.add(tf.scalar(rumpSideStart, "int32"))
  const rumpTop = detection.argMax(0).gather(tf.scalar(rumpSideStart, "int32").add(hill).cast("int32"))
  const rumpBottom = lastIndices.gather(rumpIndex)
  draw(canvas, rumpIndex.dataSync()[0], rumpTop.dataSync()[0], rumpIndex.dataSync()[0], rumpBottom.dataSync()[0], "orange", x, box.topY())
  const rumpHeight = rumpBottom.sub(rumpTop)

  return [bodyLength, shoulderHeight.dataSync()[0], rumpHeight.dataSync()[0], bodyHeight.dataSync()[0]]
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
  const leftDiff = left.sub(right).less(0).cast("int32").argMax()
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
  console.log("convert options: ", options)
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
