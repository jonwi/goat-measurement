export class AngleProvider {
  lastAngle: number | null = null;

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
