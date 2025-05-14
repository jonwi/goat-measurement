import { BoundingBox } from '@huggingface/transformers'
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
  const detection = mask.slice([box.topY(), box.topX()], [box.height(), box.width()])
  let height = detection.shape[0]
  let width = detection.shape[1]
  let horizontalCumsum = detection.cumsum(1)
  let verticalCumsum = detection.cumsum(0)

  let lastLine = horizontalCumsum.slice([0, width - 2], [-1, 1])
  let longestLine = lastLine.squeeze().argMax().add(tf.scalar(20, 'int32'))
  let [start, end, length] = await binaryRle(detection.gather(longestLine).squeeze())
  draw(canvas, start, longestLine.dataSync()[0], end, longestLine.dataSync()[0], "red", box.topX(), box.topY())

  let rump = Math.floor(length * 0.75 + start)
  let sacLine = verticalCumsum.slice([0, rump], [-1, 1]).squeeze()
  let rump_start = sacLine.equal(tf.tensor1d([1], 'int32')).toInt().argMax()
  let rump_end = sacLine.argMax()
  // draw(canvas, rump, rump_start.dataSync()[0], rump, rump_end.dataSync()[0], "green", box.topX(), box.topY())

  let shoulder = Math.floor(length * 0.25) + start
  let shoulderLine = verticalCumsum.slice([0, shoulder], [-1, 1]).squeeze()
  let shoulder_start = shoulderLine.equal(tf.tensor1d([1], 'int32')).toInt().argMax()
  let shoulder_end = shoulderLine.argMax()
  // draw(canvas, shoulder, shoulder_start.dataSync()[0], shoulder, shoulder_end.dataSync()[0], "blue", box.topX(), box.topY())

  let lastIndices = tf.fill([mask.shape[1]], mask.shape[1]).sub(mask.reverse(1).argMax(1))
  // ignore lines where mask.shape[1] because there are not supposed to be any fully filled lines and therefore they are empty
  lastIndices = lastIndices.where(lastIndices.notEqual(tf.scalar(mask.shape[1])), tf.scalar(0))
  console.log(await lastIndices.data())
  let middle = Math.floor(length * 0.5) + start
  let middleLine = verticalCumsum.slice([0, middle], [-1, 1]).squeeze()
  let middle_start = middleLine.equal(tf.tensor1d([1], 'int32')).toInt().argMax().dataSync()[0]
  let middle_end = middleLine.argMax().dataSync()[0]
  draw(canvas, middle, middle_start, middle, middle_end, "orange", box.topX(), box.topY())


  let center = Math.floor((middle_end - middle_start) * 0.50 + middle_start)
  let [center_start, center_end, body_length] = await binaryRle(detection.gather(center).squeeze())
  draw(canvas, center_start, center, center_end, center, "yellow", box.topX(), box.topY())

  let left_section = detection.slice([0, 0], [-1, middle])
  let z = left_section.mul(tf.range(0, height, 1, 'int32').expandDims(-1))
  let front_feed_end = z.max().max()
  let shoulder_height = front_feed_end.sub(shoulder_start)
  draw(canvas, shoulder + 10, shoulder_start.dataSync()[0], shoulder + 10, front_feed_end.dataSync()[0], "darkblue", box.topX(), box.topY())


  let right_section = detection.slice([0, middle], [-1, -1])
  let y = right_section.mul(tf.range(0, height, 1, 'int32').expandDims(-1))
  let back_feed_end = y.max().max()
  let rump_height = back_feed_end.sub(rump_start)
  draw(canvas, rump + 20, rump_start.dataSync()[0], rump + 20, back_feed_end.dataSync()[0], "darkgreen", box.topX(), box.topY())

  return [body_length, shoulder_height.dataSync()[0], rump_height.dataSync()[0]]
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
