"""Real-weight test matrix: several resolutions / durations / modes, one mp4 + timings each.

~/.venvs/pih3/bin/python scripts/real_matrix.py [--only name1,name2] [--out outputs/matrix]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from pih3_mlx.pipeline import PiH3Engine, save_mp4  # noqa: E402

FOX = "A red fox trotting through a snowy pine forest, snow crunching underfoot, birds chirping in the distance"
CITY = ("A street musician plays an upbeat melody on a saxophone on a rainy neon-lit city street at night, "
        "people walking past with umbrellas, the sound of rain and the saxophone")
WAVES = "Ocean waves crash against black volcanic rocks at sunset, seagulls calling overhead, slow camera pan"
TALK = ("A young woman with curly brown hair sits in a cozy cafe, looks into the camera and says warmly: "
        "\"Hi everyone, welcome back! Today I'm going to show you my favourite coffee spot.\" Soft cafe chatter "
        "in the background")
KITCHEN = ("Two friends cook dinner together in a bright home kitchen. The man chops vegetables and says: \"Can you "
           "pass me the salt?\" The woman laughs, hands it to him and replies: \"Only if I get the first taste!\"")
CHEF = ("A chef in a white uniform tosses vegetables in a flaming wok in a busy restaurant kitchen, sizzling sounds, "
        "close-up, cinematic lighting")
DANCE = ("A group of young people dance energetically to hip-hop music on a sunny city plaza, a crowd claps "
         "along, handheld camera circling the dancers")
CAFE_SAY = ("The woman smiles and says to the camera: \"Hi everyone, welcome back! Today I'll show you my favourite "
            "coffee.\" Then she lifts her latte and takes a sip. Soft cafe chatter and clinking cups in the background")
CAFE_BYE = ("The woman talks cheerfully to the camera, then laughs and waves goodbye, saying: \"See you next time, "
            "bye!\" Soft cafe chatter in the background")
CHESTNUT_SAY = ("The old man scoops hot roasted chestnuts into a paper bag, holds it out toward the camera and says with "
                "a smile: \"Careful, they're still hot!\" Crackling fire and evening street ambience")
DANCER_DO = ("The dancer performs a graceful contemporary dance, spinning and gliding across the sunlit studio floor, "
             "soft piano music")
OFFICE_DO = ("The man explains the quarterly results to his colleagues, turns to the whiteboard and draws a rising bar "
             "chart with an arrow, then points at the tallest bar and says: \"And this is where we are heading.\"")

# name, prompt, height, width, frames, extra (steps, sparse_attention, image_from, last_image_from)
CASES = [
    # text-to-video, 4 steps
    ("t2v_512x896_5s", FOX, 512, 896, 121, {}),
    ("t2v_512x896_5s_dense", FOX, 512, 896, 121, {"sparse_attention": False}),
    ("t2v_512x896_5s_talk", TALK, 512, 896, 121, {}),
    ("t2v_896x512_5s_portrait", CITY, 896, 512, 121, {}),
    ("t2v_512x896_10s_dialogue", KITCHEN, 512, 896, 243, {}),
    ("t2v_512x896_14s", WAVES, 512, 896, 345, {}),  # longest 17n+5 frame count within 15 s
    ("t2v_768x1344_5s_chef", CHEF, 768, 1344, 121, {}),
    ("t2v_768x1344_10s_dance", DANCE, 768, 1344, 243, {}),
    # image-to-video with people; keyframes generated with Gemini (Nano Banana Pro) in outputs/keyframes, the last
    # frames as edits of the first so identity and scene match
    ("i2v_cafe_5s", CAFE_SAY, None, None, 121, {"image_path": "cafe_first_generated_1.jpg"}),
    ("i2v_chestnut_portrait_5s", CHESTNUT_SAY, None, None, 121, {"image_path": "chestnut_generated_1.jpg"}),
    ("i2v_dancer_768_5s", DANCER_DO, 768, 1344, 121, {"image_path": "dancer_generated_1.jpg"}),
    ("i2v_cafe_10s", CAFE_SAY, None, None, 243, {"image_path": "cafe_first_generated_1.jpg"}),
    ("flf2v_cafe_5s", CAFE_BYE, None, None, 121,
     {"image_path": "cafe_first_generated_1.jpg", "last_image_path": "cafe_last_generated_1.jpg"}),
    ("flf2v_office_5s", OFFICE_DO, None, None, 121,
     {"image_path": "office_first_generated_1.jpg", "last_image_path": "office_last_generated_1.jpg"}),
    # 2 steps, each paired with a 4-step case above (same prompt, size and seed)
    ("t2v_512x896_5s_2step", FOX, 512, 896, 121, {"steps": 2}),
    ("t2v_512x896_5s_talk_2step", TALK, 512, 896, 121, {"steps": 2}),
    ("t2v_768x1344_5s_chef_2step", CHEF, 768, 1344, 121, {"steps": 2}),
    ("i2v_cafe_5s_2step", CAFE_SAY, None, None, 121, {"image_path": "cafe_first_generated_1.jpg", "steps": 2}),
    ("flf2v_cafe_5s_2step", CAFE_BYE, None, None, 121,
     {"image_path": "cafe_first_generated_1.jpg", "last_image_path": "cafe_last_generated_1.jpg", "steps": 2}),
    ("flf2v_office_5s_2step", OFFICE_DO, None, None, 121,
     {"image_path": "office_first_generated_1.jpg", "last_image_path": "office_last_generated_1.jpg", "steps": 2}),
]

ap = argparse.ArgumentParser()
ap.add_argument("--only", default="")
ap.add_argument("--out", default="outputs/matrix")
ap.add_argument("--steps", type=int, default=4)
ap.add_argument("--redo", action="store_true", help="rerun cases already in summary.json")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
summary_path = os.path.join(a.out, "summary.json")
summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
only = set(filter(None, a.only.split(",")))
engine = PiH3Engine()  # models stay loaded across cases

for name, prompt, h, w, frames, extra in CASES:
    if only and name not in only:
        continue
    if name in summary and "error" not in summary[name] and not a.redo:
        print(f"===== {name} already done, skipping", flush=True)
        continue
    extra = dict(extra)
    image = last_image = None
    kf = os.path.join(os.path.dirname(a.out.rstrip("/")), "keyframes")
    if "image_path" in extra:
        image = Image.open(os.path.join(kf, extra.pop("image_path")))
    if "last_image_path" in extra:
        last_image = Image.open(os.path.join(kf, extra.pop("last_image_path")))
    if "image_from" in extra:
        image = Image.open(os.path.join(a.out, extra.pop("image_from") + "_first.png"))
    if "last_image_from" in extra:
        last_image = Image.open(os.path.join(a.out, extra.pop("last_image_from") + "_last.png"))
    steps = extra.pop("steps", a.steps)
    print(f"\n===== {name}", flush=True)
    mx.reset_peak_memory()
    t = time.time()
    try:
        res = engine.generate(prompt, image=image, last_image=last_image, height=h, width=w, num_frames=frames, steps=steps,
                       seed=42, **extra)
    except Exception as e:  # keep going through the matrix
        print(f"FAILED {name}: {e!r}", flush=True)
        summary[name] = {"error": repr(e)}
        json.dump(summary, open(summary_path, "w"), indent=2)
        continue
    path = os.path.join(a.out, name + ".mp4")
    save_mp4(res, path)
    Image.fromarray(res.video[0]).save(os.path.join(a.out, name + "_first.png"))
    Image.fromarray(res.video[-1]).save(os.path.join(a.out, name + "_last.png"))
    np.save(os.path.join(a.out, name + "_frames_sub.npy"), res.video[::8])  # for sparse-vs-dense comparison
    v = res.video.astype(np.float32)
    summary[name] = {
        "steps": steps, "shape": list(res.video.shape), "audio_s": res.audio.shape[1] / res.sampling_rate,
        "wall_s": round(time.time() - t, 1), "peak_gb": round(mx.get_peak_memory() / 1e9, 1),
        "timings": {k: (round(x, 1) if isinstance(x, float) else x) for k, x in res.timings.items()},
        "video_mean": round(float(v.mean()), 2), "video_std": round(float(v.std()), 2),
        "temporal_diff": round(float(np.abs(np.diff(v[::4], axis=0)).mean()), 2),
        "audio_rms": round(float(np.sqrt((res.audio ** 2).mean())), 4),
    }
    json.dump(summary, open(summary_path, "w"), indent=2)
    print(f"===== {name} done: {summary[name]}", flush=True)
