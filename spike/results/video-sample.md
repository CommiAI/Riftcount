# Sample video — cardvision frame sample

Source: `examples/videos/sample-riftbound.mp4` (Commi upload `sample_video_62ce.mp4`).  
Clip: ~4.5 s, 1080×1920, ~30 fps.

## Sampling

- Wanted: `cardvision infer --every 0.5` on the mp4.
- Blocker: optional OpenCV video extra was not installed (`pip install …[video]`).
- Workaround: `ffmpeg -vf fps=2` → 9 JPEGs (one every 0.5 s), then `cardvision infer --release …` on that folder.
- Release: `201478a00d6ef6ba`. Accept: tau 0.62, delta 0.05.
- Full JSON: `spike/results/video-sample-report.json`. One annotated middle frame: `spike/results/video-sample-frames/frame-005-annotated.jpg`.

## What it named

Across the 9 frames it locked **24 distinct card names** (101 named boxes total, 32 unsure).

Names that locked at least once:

Applied Researchers, Apprentice Mage, Blade Twirler, Clairvoyance, Cloud Drake, Concert Poster, Covert Informant, Dredge Up, Eager Apprentice, Forsaken Baccai, Gemcraft Seer, Grumpy Rockbear, Hunters' Circle, Iterative Design, Mesmerize, Oasis Raider, Otterpus, Overloaded Traveler, Patched Porobot, Pit Crew, Pit Rookie, Questionable Tome, Shadow Assassin, Shock Blast.

## Copies / physical-card guess

This is a phone pan over cards on the floor, not a stack. Estimate = for each name, take the **most copies of that name in any one frame**, then sum:

| Name | Max in one frame |
| --- | ---: |
| Iterative Design | 2 |
| Cloud Drake | 2 |
| Dredge Up | 2 |
| Grumpy Rockbear | 2 |
| Forsaken Baccai | 2 |
| (19 other names) | 1 each |

**About 29 physical cards** named this way. Richest single frame named 14 faces (frame-005). Cards only half in frame, or only seen as unsure, can sit outside that 29.

## Unsure / misses

Unsure = detector saw a face, match stayed under the accept rule (or margin failed). Common cases:

- Cut-off cards on the left/right edges of the pan.
- One tilted bottom-left card often guessed weakly as Riptide Rex / On the Hunt / Master Yi and left unnamed.
- Occasional near-miss on a card that *is* named in a neighboring frame (e.g. Cloud Drake / Concert Poster / Pit Rookie / Shock Blast with similarity just under or barely over tau but rejected on margin).

No card backs were reported.
