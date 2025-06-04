
export interface AngleProvider {
  angle(image: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement): Promise<number>
}

export class AngleProviderSensor implements AngleProvider {
  lastAngle: number | null = null

  constructor() {
    window.addEventListener("deviceorientation", (event) => {
      if (event.gamma)
        this.lastAngle = 90 + event.gamma
      else
        this.lastAngle = null
    })
  }

  async angle(image: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement) {
    if (this.lastAngle)
      return this.lastAngle
    return 20
  }
}

export class AngleProviderStatic implements AngleProvider {
  _angle: number
  constructor(angle: number) {
    this._angle = angle
  }

  async angle(image: HTMLImageElement | HTMLVideoElement) {
    const imageName = image.src
    const number = imageName.split("_")[0]
    try {
      const req = await fetch(`${number}_data.json`)
      const data = await req.json()
      return data["Angle"]
    } catch {
      return this.angle
    }
  }
}

