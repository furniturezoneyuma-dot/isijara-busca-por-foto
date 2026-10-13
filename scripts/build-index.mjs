// Builds public/index.json + public/emb.bin (int8, 512-d CLIP ViT-B/32 image embeddings).
// Incremental: reuses embeddings for images already in the previous index.
import { pipeline, RawImage, AutoProcessor, CLIPVisionModelWithProjection } from '@huggingface/transformers';
import fs from 'fs';
const STORE = 'https://isijaradresses.com', PER = 4, D = 512, Q = 400;
const MODEL = 'Xenova/clip-vit-base-patch32';
const products = [];
for (let p = 1; ; p++) {
  const r = await fetch(`${STORE}/products.json?limit=250&page=${p}`, { headers: { 'user-agent': 'Mozilla/5.0 isijara-busca-por-foto' } });
  const j = await r.json(); if (!j.products?.length) break; products.push(...j.products);
}
console.log('products', products.length);
if (products.length < 100) throw new Error('catalog fetch looks wrong, aborting');
const key = u => u.split('?')[0];
let prev = new Map();
try {
  const pi = JSON.parse(fs.readFileSync('public/index.json')); const pb = new Int8Array(fs.readFileSync('public/emb.bin').buffer.slice(0));
  pi.keys.forEach((k, i) => prev.set(k, pb.slice(i * D, (i + 1) * D)));
} catch { }
const processor = await AutoProcessor.from_pretrained(MODEL);
const model = await CLIPVisionModelWithProjection.from_pretrained(MODEL, { dtype: 'q8' });
const P = [], keys = [], owner = [], vecs = [];
let fresh = 0;
const jobs = [];
products.forEach(pr => {
  const imgs = pr.images.slice(0, PER); if (!imgs.length) return;
  const pi = P.length; P.push([pr.handle, pr.title, imgs[0].src.split('?')[0] + '?width=300', pr.variants?.[0]?.price || '']);
  imgs.forEach(im => jobs.push({ pi, k: key(im.src), src: im.src }));
});
async function embed(src) {
  const u = src + (src.includes('?') ? '&' : '?') + 'width=400';
  const img = await RawImage.fromURL(u);
  const { image_embeds } = await model(await processor(img));
  const v = image_embeds.data; let n = 0; for (const x of v) n += x * x; n = Math.sqrt(n);
  return Int8Array.from(v, x => Math.max(-127, Math.min(127, Math.round(x / n * Q))));
}
let done = 0;
for (let i = 0; i < jobs.length; i += 8) {
  const batch = jobs.slice(i, i + 8);
  const res = await Promise.all(batch.map(async j => {
    if (prev.has(j.k)) return prev.get(j.k);
    try { const v = await embed(j.src); fresh++; return v; } catch (e) { console.warn('fail', j.src, e.message); return null; }
  }));
  batch.forEach((j, t) => { if (res[t]) { keys.push(j.k); owner.push(j.pi); vecs.push(res[t]); } });
  done += batch.length; if (done % 400 < 8) console.log(done, '/', jobs.length, 'new', fresh);
}
const buf = new Int8Array(vecs.length * D); vecs.forEach((v, i) => buf.set(v, i * D));
fs.writeFileSync('public/emb.bin', Buffer.from(buf.buffer));
fs.writeFileSync('public/index.json', JSON.stringify({ model: MODEL, dtype: 'q8', dim: D, scale: Q, updated: new Date().toISOString(), products: P, owner, keys }));
console.log('images', vecs.length, 'new', fresh, 'bytes', buf.length);
