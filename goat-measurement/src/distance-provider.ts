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
  _distance: number
  constructor(distance: number = 1.5) {
    this._distance = distance
  }

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
    return this._distance
  }
}

export class DistanceProviderInput implements DistanceProvider {
  async distance(image: HTMLImageElement | HTMLVideoElement, debugCanvas: HTMLCanvasElement) {
    const inputContainer = document.createElement("div")
    const input = document.createElement("input")
    const acceptButton = document.createElement("button")
    acceptButton.innerText = "Accept"
    inputContainer.appendChild(input)
    inputContainer.appendChild(acceptButton)
    inputContainer.classList.add("input")
    document.body.appendChild(inputContainer)

    const imageName = image.src
    let initValue = 1.5
    if (imageName.includes("Diego")) {
      initValue = 2.3
    }
    if (imageName.includes("Zara")) {
      initValue = 1.62
    }
    if (imageName.includes("Carina")) {
      initValue = 1.59
    }
    if (imageName.includes("Twentyone")) {
      initValue = 1.64
    }
    input.value = `${initValue}`

    const distance = await new Promise<number>((resolve, _) => {
      acceptButton.addEventListener("click", () => {
        const n = Number.parseFloat(input.value)
        if (n != Number.NaN) {
          resolve(+input.value)
        }
      })
    })
    document.body.removeChild(inputContainer)
    return distance
  }
}
