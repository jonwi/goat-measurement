import { DepthEstimationPipelineOutput, max, min, pipeline, RawImage } from "@huggingface/transformers"

export interface DistanceProvider {
  distance(image: HTMLImageElement, debugCanvas: HTMLCanvasElement): Promise<number>
}

export class DistanceProviderDepth implements DistanceProvider {
  pipe

  constructor() {
    this.pipe = pipeline(
      "depth-estimation",
      "onnx-community/depth-anything-v2-small",
      { device: "webgpu" },
    )
  }

  async distance(image: HTMLImageElement, debugCanvas: HTMLCanvasElement) {
    const width = image.width
    const height = image.height
    const tempCanvas = document.createElement("canvas")
    tempCanvas.width = width
    tempCanvas.height = height
    const tmpCtx = tempCanvas.getContext("2d")
    tmpCtx?.drawImage(image, 0, 0, width, height)
    document.body.append(tempCanvas)
    const imageData = RawImage.fromCanvas(tempCanvas)
    console.log("imageData", imageData)
    const pipe = await this.pipe
    const result: DepthEstimationPipelineOutput = await pipe(imageData)
    console.log(result)
    const index = height * width / 2 + width / 2
    console.log(index)
    console.log(result.predicted_depth.data[index])
    console.log(result.predicted_depth.slice([height / 2, height / 2 + 1], [width / 2, width / 2 + 1]))
    console.log(max(result.predicted_depth.data))
    console.log(min(result.predicted_depth.data))
    const ctx = debugCanvas.getContext("2d")
    ctx?.drawImage(result.depth.toCanvas(), 0, 0)


    return 1.64
  }
}

export class DistanceProviderStatic implements DistanceProvider {

  async distance(image: HTMLImageElement | HTMLVideoElement, debugCanvas: HTMLCanvasElement) {
    const imageName = image.src
    if (imageName.includes("Diego")) {
      return 2.3
    }
    if (imageName.includes("Zara")) {
      return 1.62
    }
    if (imageName.includes("Carina")) {
      return 1.59
    }
    if (imageName.includes("Twentyone")) {
      return 1.64
    }
    return 1.5
  }
}

