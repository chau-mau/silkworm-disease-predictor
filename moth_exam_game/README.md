# मादा शलभ परीक्षण मार्गदर्शिका (Mother Moth Examination Guide)

An interactive Hindi web game that teaches the **mother moth examination** procedure
for pebrine disease (*Nosema sps.*) in tasar silkworms.

**Standalone website** — completely separate from the Silkworm Disease Predictor app.
Single self-contained `index.html` (no build step, no server code, no API keys).

## Run locally

Just open `index.html` in a browser, or serve the folder:

```bash
cd moth_exam_game
python3 -m http.server 8080
# open http://localhost:8080
```

## Deploy as a separate website

- **GitHub Pages**: push this folder to a repository → Settings → Pages → deploy from branch.
- **Netlify / Vercel / Render static site**: point the project root to this folder.
- Any static web server (Apache/Nginx/IIS) can host it directly.

## Game flow (mirrors the real lab procedure)

1. **शलभ चयन** — identify the female moth (heavier body, thin antennae); the male has a
   smaller body and thick feathery antennae. Animated SVG illustrations make the difference clear.
2. **औज़ार चयन** — collect the 6 required tools from a randomly shuffled tray.
3. **कटाई-पिसाई** — cut abdomen segments 4-7 with scissors, place tissue in mortar-pestle, grind.
4. **स्लाइड तैयारी** — add PVS drops, place the ground sample on the slide, smear, then cover slip.
5. **माइक्रोस्कोप (600x)** — use the circular **coarse** and **fine** focus knobs, drag the slide
   right-to-left and up-down under the lens, find Brownian-moving rice-grain shaped pebrine spores, avoid debris.
6. **निपटान** — pebrine-positive slides are discarded into the ethanol/propanol tank.

## Feedback style

- No points or penalties are shown.
- Correct actions play a short "ding" sound and show a green ✓.
- Wrong actions play a "buzzer" sound and show a red ✗.
- Every wrong answer offers **"फिर से कोशिश करें"** (Try again) and **"आगे बढ़ें"** (Continue anyway),
  so learners are never stuck.

## Assets

- `spore_field.jpg` — clear pebrine spore microscope field supplied by the user.
- Moth illustrations are animated SVGs drawn to match the proportions of the reference plate,
  avoiding direct use of copyrighted photographs.

## Credits

Developed by **CSB-Central Tasar Research and Training Institute (CTRTI), Ranchi, Jharkhand**.
Tasar silkworm imagery and logo sourced from https://ctrti.res.in/.
