import * as tf from '@tensorflow/tfjs'
import { Box } from './yolotfjs'

export interface GoatPredictor {
  loadModel(): Promise<void>
  predict(source: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement, imageCanvas: HTMLCanvasElement, debugCanvas: HTMLCanvasElement): Promise<[tf.Tensor2D | null, Box | null]>
}
