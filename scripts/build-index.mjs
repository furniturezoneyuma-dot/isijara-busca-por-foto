// Builds public/index.json + public/emb.bin (int8 DINOv2-small CLS embeddings of every product photo).
// Incremental: reuses embeddings for images already in the previous index (same model). Usage: node scripts/build-index.mjs [workers]
import fs from 'fs'; import { fork } from 'child_process';
const STORE = 'https://isijaradresses.com', PER = 20, D = 384, Q = 127;
if (process.argv[2] === '--worker') {
  const { embedBuffer } = await import('./embed.mjs');
  process.on('message', async ({ src }) => {
    try { const b = Buffer.from(await (await fetch(src + (src.includes('?') ? '&' : '?') + 'width=400')).arrayBuffer());
      process.send({ src, v: Array.from(await embedBuffer(b)) }); } catch (e) { process.send({ src, err: e.message }); }
  });
  process.send({ ready: true });
} else {
  const { MODEL } = await import('./embed-meta.mjs');
  const products = [];
  for (let p = 1; ; p++) {
    const j = await (await fetch(`${STORE}/products.json?limit=250&page=${p}`, { headers: { 'user-agent': 'Mozilla/5.0 isijara-busca-por-foto' } })).json();
    if (!j.products?.length) break; products.push(...j.products);
  }
  console.log('products', products.length);
  if (products.length < 100) throw new Error('catalog fetch looks wrong, aborting');
  const key = u => u.split('?')[0];
  const prev = new Map();
  try {
    const pi = JSON.parse(fs.readFileSync('public/index.json')); const pb = new Int8Array(fs.readFileSync('public/emb.bin').buffer.slice(0));
    if (pi.model === MODEL && pi.dim === D) pi.keys.forEach((k, i) => prev.set(k, pb.slice(i * D, (i + 1) * D)));
  } catch { }
  const P = [], jobs = [];
  for (const pr of products) {
    const imgs = pr.images.slice(0, PER); if (!imgs.length) continue;
    const pi = P.length; P.push([pr.handle, pr.title, imgs[0].src.split('?')[0] + '?width=300', pr.variants?.[0]?.price || '']);
    imgs.forEach(im => jobs.push({ pi, k: key(im.src), src: im.src }));
  }
  const todo = [...new Map(jobs.filter(j => !prev.has(j.k)).map(j => [j.k, j.src])).values()];
  console.log('images', jobs.length, 'reused', jobs.length - todo.length, 'new', todo.length);
  const W = Math.max(1, Math.min(+process.argv[2] || 2, todo.length));
  const fresh = new Map(); let done = 0;
  if (todo.length) await new Promise(res => {
    let next = 0, alive = W;
    for (let w = 0; w < W; w++) {
      const c = fork(new URL(import.meta.url).pathname, ['--worker']);
      const feed = () => { if (next < todo.length) c.send({ src: todo[next++] }); else { c.kill(); if (--alive === 0) res(); } };
      c.on('message', m => {
        if (m.v) fresh.set(key(m.src), Int8Array.from(m.v, x => Math.max(-127, Math.min(127, Math.round(x * Q * 2.5)))));
        else if (m.err) console.warn('fail', m.src, m.err);
        if (!m.ready && ++done % 500 === 0) console.log(done, '/', todo.length);
        feed();
      });
    }
  });
  const keys = [], owner = [], vecs = [];
  for (const j of jobs) { const v = prev.get(j.k) || fresh.get(j.k); if (v) { keys.push(j.k); owner.push(j.pi); vecs.push(v); } }
  const buf = new Int8Array(vecs.length * D); vecs.forEach((v, i) => buf.set(v, i * D));
  fs.writeFileSync('public/emb.bin', Buffer.from(buf.buffer));
  fs.writeFileSync('public/index.json', JSON.stringify({ model: MODEL, dim: D, updated: new Date().toISOString(), products: P, owner, keys }));
  console.log('wrote', vecs.length, 'embeddings,', buf.length, 'bytes');
}
