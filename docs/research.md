# How to identify a Riftbound card from a photo

Commi photographs new Riftbound pack pulls and wants each card named so a later inventory can update itself. This note does not design that inventory. It records where identification already exists, what the open weights did on two real photos, and which prototype to build first.

The open weights people are pointing at are RiftSight, published on 19 October 2026. Rift Atlas is the fan-made simulator and deck site. The model card names that simulator as one of the two digital clients the weights were trained to read. Rift Atlas did not publish the model.

## The open weights

The repository is [ryangswe/riftsight-vision](https://github.com/ryangswe/riftsight-vision). GitHub reports the license as Apache-2.0. The README limits that grant to the project's own code and weights. Card names and card art stay with Riot. The bank and catalog arrangement are Apache-2.0 for the project's own work, and the card data inside the catalog is not licensed there. Label manifests are CC BY 4.0. The trademark page says a fork may say it is based on RiftSight's open vision model and must use its own name.

The only GitHub release that attaches weight files is [v0.1.0](https://github.com/ryangswe/riftsight-vision/releases/tag/v0.1.0), published 19 October 2026. Later tags through v0.6.0rc2, the tip on 30 October 2026, ship code and docs. `cardvision fetch-weights` still downloads the v0.1.0 set and checks each SHA-256. That set is release id `201478a00d6ef6ba`.

| File | What it is | Size |
| --- | --- | --- |
| `detector.onnx` | YOLOX-Nano, card face and card back | 3.7 MB |
| `recognizer.onnx` | MobileNetV3-Small features, projected to 128 values | 4.0 MB |
| `bank.json` | One 128-value vector per card, 1,492 cards | 629 KB |
| `catalog.json` | Ids, English names, kinds, and which printing reprints which | 164 KB |
| `identity-groups.json` | Printings that share a name | 20 KB |
| `accept.json` | tau 0.62, delta 0.05, rotation retry on | 57 B |

The command-line package on PyPI is [cardvision](https://pypi.org/project/cardvision/). The install used here was 0.5.1, which asks for Python 3.11 or newer. The ONNX files run on CPU. This machine has no NVIDIA device. The shell command that converted photo 1 and ran `cardvision infer` on both photos finished in 1.9 seconds.

The announcement that matches "open-sourced recently" is the 19 October 2026 post on the [JuicyKaraageTCG feed](https://nitter.rdrama.net/JuicyKaraageTCG), which links the repository and the [browser demo](https://ryangswe.github.io/riftsight-vision/). A [Shadowgraph page](https://shadowgraph.io/ryangswe/riftsight-vision) quotes the same day's Reddit posts and a Chrome extension listing. This research opened the demo page and the repository. It did not open the Chrome listing, so the extension's price is unverified. The demo page says a dropped image stays on the computer.

## How a frame becomes a name

[How it works](https://github.com/ryangswe/riftsight-vision/blob/main/docs/how-it-works.md) is the pipeline. Two models do the work, and a rule decides whether to speak.

1. The frame is scaled so its width sits between 720 and 1920 pixels.
2. The detector, YOLOX-Nano, draws boxes for a face or a back. Overlapping boxes are thinned at an overlap of 0.45. A box under 24 pixels is dropped. A back is reported and never named. The default score cutoff is 0.30.
3. Each face is pooled to a 128 by 128 grid and the recognizer emits a 128-value unit vector.
4. That vector is a dot product against every row of the bank. The score is the best similarity. The margin is that score minus the best similarity of a different identity. Printings that share a name are one identity, so another printing of the same name does not shrink the margin.
5. The look is accepted when similarity is at least 0.62 and the margin is at least 0.05. Otherwise the result is nothing. If the upright look fails, the three quarter turns are tried.
6. The name that is shown is the bank row with the best similarity. The [model card](https://github.com/ryangswe/riftsight-vision/blob/main/docs/model-card.md) says that row is not always the printing on the table.

The bank is built from card images, 16 views each. The repository does not contain those images. Adding a card means adding a bank row. The recognizer is not retrained for that.

There is no OCR step and no perceptual hash. Foil is a training-scene rate in `catalog.json`, 0.10 for runes and 0.60 for legends. It is not a class the model returns. The catalog locale is `en`. Each card record is an id, an English name, a kind, `printingOf`, and an art file name. The set code is the id prefix, as in `VEN-R04B` or `OGN-136`.

The model card's intended use is gameplay video from Rift Atlas, from TCG Arena, and from physical play filmed near overhead. Counting cards across time is listed as out of scope. On the author's public evaluation set, which the card says was looked at during development, the released pair named 92.5 percent of readable faces right. On a private held-out set of near-overhead physical play, the same page reports 322 of 382 readable faces right, 84.3 percent. Those figures are the author's. The photos below are the check that matters for pack pulls.

## Variants and languages

Alt-art runes are separate bank rows that share a name. On the three rune cards the detector found, the winning row was the printed collector code. `RAD-R01A`, `RAD-R03A`, and `VEN-R04B` are in the 1,492-card catalog, with the other rune letters through `UNL` and `SFD`.

A Chinese printing is not a catalog row. [languages.md](https://github.com/ryangswe/riftsight-vision/blob/main/docs/languages.md) is a recipe for showing a translated picture after the English id is known. It says recognition does not change. The photo of a Chinese Pit Rookie came back as English `OGN-136`. That is the behavior the docs describe. Whether the Chinese title on that card reads 竞技场新人 was not checked character by character. [OpenRift](https://openrift.app/api/v1/search?q=lang:sc%20pit%20rookie) stores that Simplified Chinese printing under the same public code `OGN-136/298`.

## Rift Atlas as a product

[riftatlas.com](https://riftatlas.com/) is a card lookup, a deck builder, and a browser simulator at [play.riftatlas.com](https://play.riftatlas.com/). The simulator page says it is free to play, unofficial, and not endorsed by Riot. The [12 to 18 October 2026 changelog](https://riftatlas.com/changelog/2026/week/42) says memberships are billed monthly through Link, with tiers named Supporter, Champion, Legend, and Signature Legend, and that more official French and Chinese card images were added. A fetch of `https://riftatlas.com/membership` on 30 October 2026 returned 404. No price was found on the changelog or on the [Metafy community page](https://metafy.gg/@riftatlas), which lists what each tier includes and does not list an amount. No public card-identification API was found. Deck export exists as a wallpaper and, from the same changelog, as JSON on a replay. A physical-collection inventory export was not found.

## Other scanners and where card data lives

### OpenRift

[OpenRift](https://github.com/openriftapp/openrift) is the closest thing to the long-term product. The README says the app is free, AGPL-3.0, and already tracks a collection with import and export. The [developers page](https://openrift.app/developers) documents a no-key JSON API. `enums.languages` lists English, Simplified Chinese, French, Korean, and Traditional Chinese. Images are at `/media/cards/{imageId}-{size}.webp`. The same page says the art belongs to Riot, asks clients not to mirror images in bulk, and sets no fixed rate limit beyond "keep traffic light." Responses are cacheable for an hour.

[docs/scanner.md](https://github.com/openriftapp/openrift/blob/main/docs/scanner.md) describes the scanner at `/scan`. Recognition runs in the browser. The camera frame stays on the device. There is no per-scan fee. The public manifest at `https://openrift.app/api/v1/scan/manifest` on 30 October 2026 listed a card detector, a board detector, a 256-float encoder, and a bank of 3,859 reference renders, built 9 October 2026. Training code is not in the repository.

The pipeline finds corners, unwarps the card, embeds it, checks the shortlist against the real card image, then reads the name band, the collector-code strip, and the stamp to pick the printing. A miss becomes a picker. The same doc says an earlier OCR and perceptual-hash scanner named the right card in 0 to 10 percent of frames and kept naming wrong ones, and that a 16-pixel color grid could not separate right matches from wrong ones. Their own clip set, as of October 2026, found 114 of 116 single-card clips with 0 wrong locks. That number is theirs, measured on recordings that are not in the repository, and it is not a score on the two photos here.

Their bench for a still photo is `bun scripts/scan/run-boards.ts`. It expects Bun, a local Postgres, and a cache of card images. That full bench was not started here. The same shared `readBoard` path was run offline instead: public ONNX files and bank from `https://openrift.app/media/scan/`, Bun, `onnxruntime-node`, and on-demand `https://openrift.app/media/cards/{uuid}-400w.webp` references. The script and JSON are under `spike/openrift/` and `spike/results/openrift-report.json`. That is their board-read code, not a reimplemented unwarp.

On quantity, the same page is explicit. Each copy counts once, including when the card stays in view. The next copy counts after the guide is empty, after a different card, or after a placement watcher sees a new card laid on the pile. The Identify button forces another add of the card still in hand. Tap mode reads every card in one still, and a second tap adds them again. Stacked and overlapping cards are listed under known gaps as best effort. The offline still-photo run did not count copies. It returned one row per detected face.

### Ximilar

[Ximilar's collectibles docs](https://docs.ximilar.com/collectibles/recognition) require a Business 100K plan or higher. The page says to email tech@ximilar.com for temporary access. The [pricing page](https://www.ximilar.com/pricing/) lists a free plan at 1,000 credits a month with no credit card, and Business 100K at €59 or $69 a month for 100,000 credits. Identify a TCG card costs 10 credits. Default `tcg_id` names the largest card in each image, up to 10 images. `analyze_all` reads every card in one image and charges per card. The [taxonomy](https://docs.ximilar.com/taxonomy/collectibles) lists Riftbound in English only, with fields name, full name, set, set code, card number, year, and card id. Those pages do not describe a stack count. This research did not call the API. [Gibl TCG](https://gibltcg.com/docs/api-riftbound) documents a keyed `predict-card` call that claims Riftbound. No price was found on that page, and the endpoint was not called.

### Apps that already make the person type the count

[TCGplayer Quicklist](https://help.tcgplayer.com/hc/en-us/articles/115004326088-How-to-Begin-Scanning-Cards-with-Quicklist) refuses to scan the same card twice in a row and tells the seller to set quantity with a hotkey or Edit. It also says not to stack cards under the camera, and that sleeves make glare worse. Whether that scanner lists Riftbound was not on the page. [ManaBox](https://manabox.app/guides/scanner/getting-started/) identifies from artwork and has a control for more than one copy, plus a tap that scans the same card again. It tells the user to avoid overlap. [Delver Lens](https://www.reddit.com/r/delverlens/comments/xtsw76/app_suggestion_for_bulk_scanning/) scans one card at a time. The developer, in that thread, describes merging rows so the list reads as "3x Island." None of these three was checked for a Riftbound catalog. The searches for this note did not find a Dragon Shield photo-identification API.

### Data, and the DIY routes that already lost a measurement

The official gallery the [riftbound-scanner](https://github.com/Nekoraru22/riftbound-scanner) README scrapes is `https://riftbound.leagueoflegends.com/en-us/card-gallery/`. That repository is MIT. Its README specifies YOLO11 pose for corners and a 16 by 16 color grid matched by cosine similarity. A listing of the git tree found no ONNX file and no hash file, so the weights are produced by the scraper and a training run, not downloaded. OpenRift's scanner doc is the reason not to start there. They measured that color grid and could not set a threshold that rejects a wrong card.

[RiftHunt](https://rifthunt.com/api/) serves a no-key catalog. The page fetched for this note says the text language is only `en`, and it hosts WebP images. [CollectorVision](https://github.com/HanClinto/CollectorVision) lists a Riftbound catalog as an experimental preview beside its Magic embeddings. [Wardeye](https://github.com/effe-exe/Wardeye/blob/main/ml/README.md) studies cards in stream video, using color grids, dHash, and DINOv2.

A from-scratch YOLO plus CLIP or DINOv2 index is the same shape as the two systems above, without their banks. OpenRift tried MobileCLIP, kept a smaller trained MobileNet because the CLIP encoder was too slow on a phone, and still verifies with pixels. Starting there would repeat that work.

## One photo does not count four copies

A squared stack shows one face. The copies underneath have no pixels in the frame. No embedding, hash, or OCR of that face can see them. A side photo of the edges can, in principle, count sheets. Same-color sleeves make those edges hard to separate, and none of the tools above do that count. A fan can show several borders. If the borders overlap enough that the detector emits one box, the count is still one, and the name can get worse because the crop is several cards at once.

Existing tools do not infer the count from a pile. They identify a face and then either wait for the next physical copy or let the person type the number. RiftSight's model card lists counting as out of scope. OpenRift re-arms when a new card is placed. TCGplayer, ManaBox, and Delver Lens all keep quantity as an edit on a single scan.

The paste test in `spike/results/report.json` is four copies of the Fury Rune crop from photo 1, laid on a flat green field. It is not a physical stack. There is no sleeve thickness and no new glare.

| Scene | Boxes | Names |
| --- | --- | --- |
| One copy | 1 | Fury Rune `RAD-R01A` |
| Four copies, 4 pixel steps | 1 | Fury Rune `RAD-R01A` |
| Four copies, 90 pixel fan | 0 | none |
| Four copies, 260 pixel fan | 0 | none |
| Four copies with a gap | 2 | Fury Rune twice |

The squared paste is one card, named correctly. The fans produced no box. Four separated copies produced two. A repeated paste of the same pixels is a harsh input, and a real photo of four physical cards with gaps may do better. It does not rescue the squared stack. That layout has nothing to count.

## What the two photos returned

Release `201478a00d6ef6ba`, package cardvision 0.5.1, CPU only. The command and the JSON are under `spike/`. Photo 1 arrived as a 3024 by 4032 HEIC and was saved as `examples/photos/photo1-runes.jpg`. Photo 2 arrived as a 480 by 360 JPEG, `examples/photos/photo2-fabric.jpg`. The detector scales a narrow frame up to a width of 720, so photo 2 was identified from that small file.

### Photo 1, five runes on a green mat

The collector line was read by eye from crops. No OCR engine was run.

| Card on the table | Printed line | Model |
| --- | --- | --- |
| Calm Rune, purple sleeve | `VEN · R02b · P · EN` | no box |
| Mind Rune, purple sleeve | `VEN · R03a · EN` | no box |
| Body Rune, purple sleeve | `VEN · R04b · P · EN` | Body Rune `VEN-R04B`, similarity 0.862, detector 0.321 |
| Mind Rune, toploader | `RAD · R03a · EN` | Mind Rune `RAD-R03A`, similarity 0.695, detector 0.491 |
| Fury Rune, toploader | `RAD · R01a · EN` | Fury Rune `RAD-R01A`, similarity 0.803, detector 0.590 |

Three of five cards were found. Each of those three matched the printed code, including the promo letter, despite sleeve edges and toploader glare. The two misses are the green bamboo Calm Rune and the dark green Mind Rune, both in purple sleeves on the green mat. Lowering the detector cutoff to 0.05 still produced no face box on them. The best face score in the whole frame was 0.586, on the Fury Rune. Two back boxes in the top-middle of the frame scored 0.104 and 0.069, under the 0.30 cutoff, and neither sits on the Calm Rune. This is a miss, not a near miss. The Body Rune was found at 0.321, just over the cutoff, and its art is the high-contrast yellow cube pattern.

### Photo 2, nine cards on patterned fabric

Nine faces, zero backs. Eight names agree with text that is readable on the photo or with the official art.

Cloud Drake `VEN-048`, Otterpus `VEN-053`, Dredge Up `VEN-049`, Shock Blast `VEN-059`, Grumpy Rockbear `VEN-050`, Overloaded Traveler `RAD-072`, and Concert Poster `RAD-116` match the English titles visible on those cards. Pit Rookie `OGN-136` is the card with Chinese text. Its figure matches the [OpenRift English image](https://openrift.app/media/cards/f14b643a-dd5d-4383-9ce8-99628efa3c95-240w.webp) for `OGN-136/298`. Similarity was 0.766. The model returned the English id, which is all the bank contains.

Hunters' Circle `RAD-163` is the sideways card at the upper left. The catalog kind is Battlefield, and the match used the rotation retry. Similarity was 0.910. The title is not readable on a 480 by 360 file, so this name is unverified by eye. The task note expected one face-down card. This file shows a printed face there, and the run emitted no back.

### OpenRift `readBoard` on the same two files

Bank `scan-bank-50d08713bd8b8fa4` (3,859 keys, 256-d ArcFace), encoder and board detector `v2`, Bun, CPU. Photo 1 was resized to a long side of 2,048. Photo 2 stayed 480 by 360. JSON: `spike/results/openrift-report.json`.

| Photo | cardvision faces | OpenRift cards | Confident OpenRift |
| --- | --- | --- | --- |
| photo1-runes | 3 named of 5 | 3 | 3 |
| photo2-fabric | 9 named | 6 | 5 |

Photo 1 returned Body Rune `VEN-R04b-P-EN`, Mind Rune `RAD-R03a-SC`, and Fury Rune `RAD-R01a-EN`. The same three faces cardvision named. The purple-sleeved Calm and Mind runes still had no outline. The toploader Mind Rune came back as the Simplified Chinese printing key even though the sleeve line read `RAD · R03a · EN` by eye. Printing resolution did not lock (`printingResolved: false`), so that is an art-level hit with the wrong language row still on top.

Photo 2 confident hits: Otterpus, Shock Blast, Dredge Up, Grumpy Rockbear, Pit Rookie. Concert Poster was proposed and marked not confident. Missed relative to cardvision: Cloud Drake, Overloaded Traveler, and Hunters' Circle. Pit Rookie was confident under the same English image id cardvision used. No quantity came out of either photo. Six faces in one still are six rows of count 1.

## Four ways to build the first identifier

| Approach | What you get | What it costs |
| --- | --- | --- |
| OpenRift `/scan`, and its collection if the export is enough | Printing, language, collector code, and a free on-device scanner that already abstains when unsure | Measured on these stills via `readBoard`: same 3 of 5 runes as cardvision, 5 confident of 9 fabric faces, one wrong-language Mind key. Stacks remain a documented best-effort gap. AGPL-3.0. Phone UX still untested here. |
| `cardvision infer` on a photo of separated faces | The spike above. Apache-2.0 weights. Several cards per still. Printing-level ids when the art differs. Runs on CPU. | Missed 2 of 5 sleeved runes. English ids only. No foil flag. No quantity. A card that fills the frame can score 0, as the first bad paste did before the card was shrunk onto a mat. Found three fabric cards OpenRift's board detector skipped on the 480 px file. |
| Ximilar `tcg_id` | A hosted multi-card API, with Riftbound English fields and `analyze_all` billed per card | Business plan at $69 a month, or an emailed trial. Taxonomy lists English only for Riftbound. Not called here. No stack count. |
| OCR of the bottom code, then a catalog lookup | The line `VEN · R02b · P · EN` is the printing, the promo mark, and the language, and it is readable on the full-size rune photo | OpenRift measured a text-and-hash scanner at 0 to 10 percent of frames and dropped it. Glare and a 480 pixel photo hide the line. Use the line as a check on an embedding hit, which is what OpenRift's printing stage already does. |

## Recommendation

OpenRift's still-photo path already names the same three runes cardvision names and the Chinese Pit Rookie, and it already abstains on Concert Poster. It still misses the two purple-sleeved green runes and three fabric faces on the small photo. Prefer OpenRift's phone `/scan` for day-to-day collection work before writing a second model here. The AGPL license means a private fork that you run for yourself is a different decision from pasting their scanner into an app you ship. Read the license before copying.

If phone capture still drops sleeved runes, or if the photos have to stay in this repository under Apache-2.0 tooling, the first code to write here is a folder watcher around `cardvision infer`. One JSON row per accepted face, count 1, card id taken from the bank row. Ask for a typed count when the person has several copies of that face. Do not add a stack counter. The paste test says a squared pile is one card, and a fan can be zero. Add the printed collector line as a second opinion only after the embedding hit exists, and only to confirm the rune letter and the language. Do not start from a color grid, from CLIP, or from a paid API until this pair has failed on a larger set of Commi's own photos.

## A real stack test

Use four physical copies of one card Commi can spare. Sleeves on, because that is how the cards are stored. Photograph each layout once, from the same height as photo 1, and run `cardvision infer` plus one pass through OpenRift's scanner.

1. Squared stack, edges flush, camera above. Expect one face and a count of 1 from both tools. A count of 4 here would be a surprise worth keeping the photo for.
2. The same stack from the side, edges toward the camera, against a contrasting background. Count the visible edges by hand. This is the only layout that can see hidden copies. Record whether a person can count them. Do not expect either model to.
3. Fan, each card shifted by about a quarter of its width. Record how many boxes `cardvision infer` emits and how many locks OpenRift adds from one still. Then place the copies one at a time in OpenRift's guide and record whether the placement watcher adds four.
4. Four copies separated on the mat, no overlap. This is the capture convention the first prototype should require. Expect four names. Photo 1 says sleeves on a similar-colored mat can still drop a card, so keep one miss as a reason to retake, not as a reason to invent a count.

Keep the photos. A later change to the threshold or the bank should be scored against this set, with wrong names counted separately from misses.
