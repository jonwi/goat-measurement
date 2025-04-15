import { SamModel, AutoProcessor, RawImage } from '@huggingface/transformers'


const modelProm = SamModel.from_pretrained('Xenova/slimsam-77-uniform', { device: "webgpu" })
const processorProm = AutoProcessor.from_pretrained('Xenova/slimsam-77-uniform', {})

export async function createMask(image: HTMLImageElement, debugCavnas: HTMLCanvasElement) {
  // Load model and processor
  const model = await modelProm
  const processor = await processorProm

  // Prepare image and input points
  const img_url = image.src
  const raw_image = await RawImage.read(img_url)
  const input_points = [[[340, 250]]]

  // Process inputs and perform mask generation
  const inputs = await processor(raw_image, { input_points })
  const outputs = await model(inputs)

  // Post-process masks
  const masks = await processor.post_process_masks(outputs.pred_masks, inputs.original_sizes, inputs.reshaped_input_sizes)
  console.log("sam2", masks)
}
