"""Soft-VAD segmenter with pre-roll (no hard gate that clips word starts).

The front end emits a smoothed speech probability per frame. This segmenter
turns it into speech segments: a segment opens when p >= on_threshold and is
extended backwards by ``pre_roll`` frames (200 ms); it closes after p stays
below off_threshold for ``hangover`` frames. Features are never zeroed --
the VAD only decides which span to decode / finalise.
"""
from __future__ import annotations


class VadSegmenter:
    def __init__(self, on_threshold=0.6, off_threshold=0.3, pre_roll=20, hangover=30):
        self.on, self.off = on_threshold, off_threshold
        self.pre_roll, self.hangover = pre_roll, hangover
        self.t = 0
        self.active = False
        self.start = 0
        self.quiet = 0

    def push(self, probs) -> list[tuple[str, int]]:
        """Returns events ('start', frame) / ('end', frame) in global frame indices."""
        ev = []
        for p in probs:
            if not self.active and p >= self.on:
                self.active = True
                self.start = max(0, self.t - self.pre_roll)
                self.quiet = 0
                ev.append(("start", self.start))
            elif self.active:
                self.quiet = self.quiet + 1 if p < self.off else 0
                if self.quiet >= self.hangover:
                    self.active = False
                    ev.append(("end", self.t + 1))
            self.t += 1
        return ev


def segments_from_probs(probs, **kw) -> list[tuple[int, int]]:
    seg = VadSegmenter(**kw)
    evs = seg.push(probs)
    out, s = [], None
    for kind, t in evs:
        if kind == "start":
            s = t
        else:
            out.append((s, t))
            s = None
    if s is not None:
        out.append((s, len(probs)))
    return out
