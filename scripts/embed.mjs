// Shared embedding (index build). Browser does the same: pad to white square 224 -> DINOv2-small -> CLS token, L2-normalized.
// NOTE: quantized (q8) ONNX models gave corrupted embeddings on onnxruntime-node (~1/3 of images); use fp32 here.
import { AutoProcessor, AutoModel, RawImage } from '@huggingface/transformers';
import sharp from 'sharp';
export const MODEL = 'Xenova/dinov2-small';
const proc = await AutoProcessor.from_pretrained(MODEL);
const model = await AutoModel.from_pretrained(MODEL, { dtype: 'fp32', session_options: { intraOpNumThreads: 1 } });
export async function embedBuffer(buf) {
  const { data } = await sharp(buf).flatten({ background: '#ffffff' }).resize(224, 224, { fit: 'contain', background: '#ffffff' }).raw().toBuffer({ resolveWithObject: true });
  const o = await model(await proc(new RawImage(new Uint8ClampedArray(data), 224, 224, 3)));
  const D = o.last_hidden_state.dims[2], v = o.last_hidden_state.data.slice(0, D);
  let n = 0; for (const x of v) n += x * x; n = Math.sqrt(n);
  return Float32Array.from(v, x => x / n);
}
