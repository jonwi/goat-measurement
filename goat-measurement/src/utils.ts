import * as tf from '@tensorflow/tfjs'
import { Box } from './yolotfjs'

async function binaryRle(t: tf.Tensor1D) {
  let l = t.shape[0]
  let indices = tf.range(0, t.shape[0], 1, 'int32')

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
    let begin = nonzero.gather(maxLengthIndex.sub(tf.tensor1d([1], 'int32'))).data()

    let res = await Promise.all([begin, end, length])
    let unwrap = res.map(r => r[0])
    unwrap[0] += 1
    return unwrap
  }
  return [-1, -1, -1]
}

export async function bodyMeasurement(mask: tf.Tensor2D, box: Box, canvas: HTMLCanvasElement | null = null) {
  // mask is hxw 640x640
  let detection = mask.slice([box.topY(), box.topX()], [box.height(), box.width()])
  let height = detection.shape[0]
  const sm = detection.sum(0).greater(tf.scalar(0, "int32"))
  const xStart = sm.argMax()
  const xEnd = tf.scalar(detection.shape[1] - 1, "int32").sub(sm.reverse().argMax())
  console.log("xStart xEnd", xStart.dataSync(), xEnd.dataSync())
  detection = detection.slice([0, xStart.dataSync()[0]], [height, xEnd.sub(xStart).dataSync()[0]])
  console.log("detection shape", detection.shape)
  let width = detection.shape[1]
  const x = xStart.dataSync()[0] + box.topX()

  let lastIndices = tf.fill([width], height).sub(detection.reverse(0).argMax(0))
  console.log("lastIndices", await lastIndices.data())
  const middleSplit = Math.ceil(width * 0.6)
  const headWidth = width * 0.2 // this might need more finetuning
  const leftWidth = width * 0.4
  const rightWidth = width * 0.4
  drawRect(canvas, x, box.topY(), headWidth, box.height(), "#3022dd24")
  drawRect(canvas, x + headWidth, box.topY(), leftWidth, box.height(), "#d52a3d24")
  drawRect(canvas, x + headWidth + leftWidth, box.topY(), rightWidth, box.height(), "#55d32c24")
  const lowestLeft = lastIndices.slice(headWidth, middleSplit).argMax()
  console.log("left", await lowestLeft.data())
  drawVerticalLine(canvas, (await lowestLeft.data())[0] + x + headWidth, "green")
  const lowestRight = lastIndices.slice(middleSplit, width - middleSplit).argMax()
  drawVerticalLine(canvas, (await lowestRight.data())[0] + x + middleSplit, "red")
  console.log(await lastIndices.data())
  const newMiddle = lowestLeft
    .add(tf.scalar(headWidth, "int32"))
    .add(lowestRight.add(tf.scalar(middleSplit, 'int32')))
    .div(tf.scalar(2, 'int32'))
  drawVerticalLine(canvas, newMiddle.dataSync()[0] + x, "purple")

  const middleStart = detection.argMax(0).gather(newMiddle)
  const middleEnd = lastIndices.gather(newMiddle)
  draw(canvas, newMiddle.dataSync()[0], middleStart.dataSync()[0], newMiddle.dataSync()[0], middleEnd.dataSync()[0], "yellow", x, box.topY())

  const leftIndex = lowestLeft.add(tf.scalar(headWidth, 'int32')).add(tf.scalar(1, 'int32'))
  const shoulderStart = detection.argMax(0).gather(leftIndex)
  const shoulderEnd = lastIndices.gather(leftIndex)
  console.log("shoulderEnd", shoulderEnd.dataSync())
  draw(canvas, leftIndex.dataSync()[0], shoulderStart.dataSync()[0], leftIndex.dataSync()[0], shoulderEnd.dataSync()[0], "orange", x, box.topY())
  const shoulder_height = shoulderEnd.sub(shoulderStart)

  const rightIndex = lowestRight.add(tf.scalar(headWidth, 'int32')).add(tf.scalar(leftWidth, 'int32')).add(tf.scalar(2, 'int32'))
  const rumpStart = detection.argMax(0).gather(rightIndex)
  const rumpEnd = lastIndices.gather(rightIndex)
  draw(canvas, rightIndex.dataSync()[0], rumpStart.dataSync()[0], rightIndex.dataSync()[0], rumpEnd.dataSync()[0], "orange", x, box.topY())
  const rump_height = rumpEnd.sub(rumpStart)

  const body_length = 0

  return [body_length, shoulder_height.dataSync()[0], rump_height.dataSync()[0]]
}

function drawRect(canvas: HTMLCanvasElement | null, x: number, y: number, width: number, height: number, style: string) {
  if (canvas == null) return
  let ctx = canvas.getContext("2d")!
  ctx.fillStyle = style
  ctx.fillRect(x, y, width, height)
}

function drawVerticalLine(canvas: HTMLCanvasElement | null, x: number, style: string) {
  if (canvas == null) return
  draw(canvas, x, 0, x, canvas.height, style, 0, 0)
}

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

function scaleToWidth(pixels: number, convertOptions: Options) {
  return pixels / convertOptions.mask_shape[1] * convertOptions.orig_shape[1]
}

function scaleToHeight(pixels: number, convertOptions: Options) {
  return pixels / convertOptions.mask_shape[0] * convertOptions.orig_shape[0] / Math.cos(convertOptions.angle * Math.PI / 180)
}

function pixelsToCm(pixels: number, convertOptions: Options) {
  return pixels / (convertOptions.calibration * convertOptions.calibration_distance / (convertOptions.distance * 100))
}

export function convertToCm(body_length: number, shoulder_height: number, rump_height: number, convertOptions: ConvertOptions) {
  const options = { ...DefaultConvertOptions, ...convertOptions }
  console.log("convert options: ", options)
  return [
    pixelsToCm(scaleToWidth(body_length, options), options),
    pixelsToCm(scaleToHeight(shoulder_height, options), options),
    pixelsToCm(scaleToHeight(rump_height, options), options)
  ]
}

type ConvertOptions = {
  distance?: number
  calibration?: number
  calibration_distance?: number
  orig_shape?: number[]
  mask_shape?: number[]
  angle?: number
}

type Options = {
  distance: number
  calibration: number
  calibration_distance: number
  orig_shape: number[]
  mask_shape: number[]
  angle: number
}

const DefaultConvertOptions: Options = {
  distance: 1.5,
  calibration: 3.375,
  calibration_distance: 200,
  orig_shape: [640, 640],
  mask_shape: [640, 640],
  angle: 20,
}
