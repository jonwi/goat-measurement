
export function predictWeight(bodyLength: number, shoulderHeight: number, rumpHeight: number, heartGirth: number) {
  console.log("bodyLength", bodyLength, "shoulderHeight", shoulderHeight, "rumpHeight", rumpHeight)
  return bodyLength * 0.45287999 + rumpHeight * 1.30813392 + shoulderHeight * 0.55532975 - 111.45145379928671
}
