
export interface AngleProvider {
  angle(): Promise<number>
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

  async angle() {
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

  async angle() {
    return this._angle
  }
}

