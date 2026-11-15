// Print every figures/html/*.html page to a vector PDF of exactly its own size (figures/pdf).
// Tables are typeset by LaTeX (tables/tex). Usage, from the replication folder:
//   node figures/render.js            # everything
//   node figures/render.js fig_size   # only the named pages
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

(async () => {
  const root = path.join(__dirname, '..');
  const only = process.argv.slice(2);
  const browser = await chromium.launch();
  const page = await browser.newPage({ deviceScaleFactor: 1 });
  for (const sub of ['figures']) {
    const dir = path.join(root, sub, 'html');
    const out = path.join(root, sub, 'pdf');
    if (!fs.existsSync(dir)) continue;
    fs.mkdirSync(out, { recursive: true });
    const files = fs.readdirSync(dir).filter(f => f.endsWith('.html'))
      .filter(f => only.length === 0 || only.includes(path.basename(f, '.html')));
    for (const f of files) {
      await page.setViewportSize({ width: 800, height: 600 });
      await page.goto('file://' + path.join(dir, f));
      await page.evaluate(() => document.fonts.ready);
      const box = await page.$eval('#fig', el => {
        const r = el.getBoundingClientRect();
        return { w: Math.ceil(r.width), h: Math.ceil(r.height) };
      });
      const pdf = path.join(out, f.replace(/\.html$/, '.pdf'));
      await page.pdf({ path: pdf, width: box.w + 'px', height: (box.h + 1) + 'px',
                       printBackground: true, pageRanges: '1' });
      console.log(`${sub}/${f} -> pdf (${box.w}x${box.h}px)`);
    }
  }
  await browser.close();
})();
