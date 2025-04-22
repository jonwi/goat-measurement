
export function predictWeight(bodyLength: number, shoulderHeight: number, rumpHeight: number, heartGirth: number) {
  return bodyLength * 0.45287999 + shoulderHeight * 1.30813392 + rumpHeight * 0.55532975 - 111.45145379928671
}
