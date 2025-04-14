import * as tf from '@tensorflow/tfjs'

async function binary_rle(t: tf.Tensor1D) {
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

export async function body_measurement(mask: tf.Tensor2D, canvas: HTMLCanvasElement | null = null) {
  // mask is hxw 640x640
  let height = mask.shape[0]
  let width = mask.shape[1]
  let horizontalCumsum = mask.cumsum(1)
  let verticalCumsum = mask.cumsum(0)

  let lastLine = horizontalCumsum.slice([0, width - 2], [-1, 1])
  let longestLine = lastLine.squeeze().argMax().add(tf.scalar(20, 'int32'))
  let [start, end, length] = await binary_rle(mask.gather(longestLine).squeeze())
  draw(canvas, start, longestLine.dataSync()[0], end, longestLine.dataSync()[0], "red")

  let rump = Math.floor(length * 0.75 + start)
  let sacLine = verticalCumsum.slice([0, rump], [-1, 1]).squeeze()
  let rump_start = sacLine.equal(tf.tensor1d([1], 'int32')).toInt().argMax()
  let rump_end = sacLine.argMax()
  // draw(canvas, rump, rump_start.dataSync()[0], rump, rump_end.dataSync()[0], "green")

  let shoulder = Math.floor(length * 0.25) + start
  let shoulderLine = verticalCumsum.slice([0, shoulder], [-1, 1]).squeeze()
  let shoulder_start = shoulderLine.equal(tf.tensor1d([1], 'int32')).toInt().argMax()
  let shoulder_end = shoulderLine.argMax()
  // draw(canvas, shoulder, shoulder_start.dataSync()[0], shoulder, shoulder_end.dataSync()[0], "blue")

  let middle = Math.floor(length * 0.5) + start
  let middleLine = verticalCumsum.slice([0, middle], [-1, 1]).squeeze()
  let middle_start = middleLine.equal(tf.tensor1d([1], 'int32')).toInt().argMax().dataSync()[0]
  let middle_end = middleLine.argMax().dataSync()[0]
  draw(canvas, middle, middle_start, middle, middle_end, "orange")


  let center = Math.floor((middle_end - middle_start) * 0.60 + middle_start)
  let [center_start, center_end, body_length] = await binary_rle(mask.gather(center).squeeze())
  draw(canvas, center_start, center, center_end, center, "yellow")

  let left_section = mask.slice([0, 0], [-1, middle])
  let z = left_section.mul(tf.range(0, height, 1, 'int32').expandDims(-1))
  let front_feed_end = z.max().max()
  let shoulder_height = front_feed_end.sub(shoulder_start)
  draw(canvas, shoulder + 10, shoulder_start.dataSync()[0], shoulder + 10, front_feed_end.dataSync()[0], "darkblue")


  let right_section = mask.slice([0, middle], [-1, -1])
  let y = right_section.mul(tf.range(0, height, 1, 'int32').expandDims(-1))
  let back_feed_end = y.max().max()
  let rump_height = back_feed_end.sub(rump_start)
  draw(canvas, rump + 20, rump_start.dataSync()[0], rump + 20, back_feed_end.dataSync()[0], "darkgreen")

  return [body_length, shoulder_height.dataSync()[0], rump_height.dataSync()[0]]
}

function draw(canvas: HTMLCanvasElement | null, x1: number, y1: number, x2: number, y2: number, style: string) {
  if (canvas != null) {
    let ctx = canvas.getContext('2d')!
    ctx.beginPath()
    ctx.moveTo(x1, y1)
    ctx.lineTo(x2, y2)
    ctx.strokeStyle = style
    ctx.stroke()
  }
}

function scale_to_width(pixels: number, convertOptions: Options) {
  return pixels / convertOptions.mask_shape[1] * convertOptions.orig_shape[1]
}

function scale_to_height(pixels: number, convertOptions: Options) {
  return pixels / convertOptions.mask_shape[0] * convertOptions.orig_shape[0] / Math.cos(convertOptions.angle * Math.PI / 180)
}

function pixels_to_cm(pixels: number, convertOptions: Options) {
  return pixels / (convertOptions.calibration * convertOptions.calibration_distance / (convertOptions.distance * 100))
}

export function convert_to_cm(body_length: number, shoulder_height: number, rump_height: number, convertOptions: ConvertOptions) {
  const options = { ...DefaultConvertOptions, ...convertOptions }
  console.log("convert options: ", options)
  return [
    pixels_to_cm(scale_to_width(body_length, options), options),
    pixels_to_cm(scale_to_height(shoulder_height, options), options),
    pixels_to_cm(scale_to_height(rump_height, options), options)
  ]
}

type ConvertOptions = {
  distance?: number;
  calibration?: number;
  calibration_distance?: number;
  orig_shape?: number[];
  mask_shape?: number[];
  angle?: number;
}

type Options = {
  distance: number;
  calibration: number;
  calibration_distance: number;
  orig_shape: number[];
  mask_shape: number[];
  angle: number;
}

const DefaultConvertOptions: Options = {
  distance: 1.5,
  calibration: 165.85,
  calibration_distance: 20,
  orig_shape: [4032, 3024],
  mask_shape: [640, 480],
  angle: 20,
}
