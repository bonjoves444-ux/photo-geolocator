# photo-geolocator-01 entry (Python 3.13, standard library only).
#
# Model-free: no model calls at all, so it needs no API key. For each of
# the instance's 25 street-level photos it reads the JPEG's DC coefficients
# (one average colour per 8x8 block, no full decode) and places the photo
# with two hand-reasoned rules:
#
#   1. A yellow centre line on the road means the Americas. Yellow centre
#      lines are the marking convention of the Americas (US, Canada,
#      Mexico, Brazil, ...); most of the rest of the world marks the centre
#      white. A photo whose road surface is clearly yellow-marked is
#      placed at the Americas' centre, (0, -70).
#   2. Otherwise the photo is placed at (57, 28), the haversine centre of
#      the OpenStreetView-5M photo distribution: the single point that
#      minimises total great-circle error over the kit's 75 practice
#      photos (and the same point, within a degree or two, on any two of
#      the three practice trials held out in turn).
#
# Nothing else is used: no trained weights, no embeddings, no table of
# photos or places, no network but the env, no clock, no randomness, so
# two runs on the same instance give the same answers.
#
# Run as the grader runs it: python3 -P -s /bot/main.py with
# MAKEFUN_ENV_URL set to the env and MAKEFUN_PROMPT_FILE to the brief.

import json
import os
import struct
import sys
import urllib.request

# The single point that minimises total haversine error over the kit's
# 75 practice photos (see the module docstring).
PRIOR_LAT = 57.0
PRIOR_LON = 28.0
# The centre of the Americas, for yellow-marked roads (rule 1).
AMERICAS_LAT = 0.0
AMERICAS_LON = -70.0
# Bottom-zone fraction of yellow-marked road blocks that triggers rule 1.
YELLOW_MIN = 0.05

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")


class JpegError(Exception):
    pass


class _Restart(Exception):
    pass


class BitReader:
    """Bits of the entropy-coded scan, with 0xFF00 unstuffing."""

    def __init__(self, data):
        self.d = data
        self.i = 0
        self.bitbuf = 0
        self.bitcnt = 0

    def _next_byte(self):
        while True:
            if self.i >= len(self.d):
                raise JpegError("scan data exhausted")
            b = self.d[self.i]
            self.i += 1
            if b != 0xFF:
                return b
            if self.i >= len(self.d):
                raise JpegError("scan data exhausted")
            nb = self.d[self.i]
            if nb == 0x00:
                self.i += 1
                return 0xFF
            if 0xD0 <= nb <= 0xD7:
                self.i += 1
                return None  # restart marker
            if nb == 0xFF:
                continue  # fill byte: the 0xFF just read was padding
            raise JpegError("unexpected marker in scan")

    def bit(self):
        if self.bitcnt == 0:
            b = self._next_byte()
            if b is None:
                raise _Restart()
            self.bitbuf = b
            self.bitcnt = 8
        self.bitcnt -= 1
        return (self.bitbuf >> self.bitcnt) & 1

    def bits(self, n):
        v = 0
        for _ in range(n):
            v = (v << 1) | self.bit()
        return v


def _huff_table(lengths, symbols):
    table = {}
    code = 0
    k = 0
    for ln in range(1, 17):
        for _ in range(lengths[ln - 1]):
            table[(ln, code)] = symbols[k]
            k += 1
            code += 1
        code <<= 1
    return table


def _decode_huff(br, table):
    code = 0
    for ln in range(1, 17):
        code = (code << 1) | br.bit()
        v = table.get((ln, code))
        if v is not None:
            return v
    raise JpegError("bad huffman code")


def _receive_extend(br, n):
    if n == 0:
        return 0
    v = br.bits(n)
    if v < (1 << (n - 1)):
        return v - (1 << n) + 1
    return v


def _parse_jpeg(data):
    """Return (width, height, components, quant, dc, ac, sos, scan)."""
    if data[:2] != b"\xff\xd8":
        raise JpegError("not a JPEG")
    i = 2
    quant = {}
    dc_tables = {}
    ac_tables = {}
    comps = []
    width = height = 0
    while i < len(data):
        if data[i] != 0xFF:
            raise JpegError("bad marker at %d" % i)
        m = data[i + 1]
        if m == 0xD9:
            break
        if m == 0xDA:
            hdr_len = struct.unpack(">H", data[i + 2:i + 4])[0]
            hdr = data[i + 4:i + 2 + hdr_len]
            sos = []
            for c in range(hdr[0]):
                cid = hdr[1 + 2 * c]
                tb = hdr[2 + 2 * c]
                sos.append((cid, tb >> 4, tb & 0x0F))
            return width, height, comps, quant, dc_tables, ac_tables, sos, \
                data[i + 2 + hdr_len:]
        ln = struct.unpack(">H", data[i + 2:i + 4])[0]
        seg = data[i + 4:i + 2 + ln]
        if m == 0xDB:  # DQT
            j = 0
            while j < len(seg):
                pq, tq = seg[j] >> 4, seg[j] & 0x0F
                j += 1
                if pq == 0:
                    quant[tq] = list(seg[j:j + 64])
                    j += 64
                else:
                    quant[tq] = [struct.unpack(">H", seg[j + 2 * k:j + 2 * k + 2])[0]
                                 for k in range(64)]
                    j += 128
        elif m == 0xC4:  # DHT
            j = 0
            while j < len(seg):
                tc, th = seg[j] >> 4, seg[j] & 0x0F
                lengths = list(seg[j + 1:j + 17])
                total = sum(lengths)
                symbols = list(seg[j + 17:j + 17 + total])
                (dc_tables if tc == 0 else ac_tables)[th] = \
                    _huff_table(lengths, symbols)
                j += 17 + total
        elif m == 0xC0:  # SOF0 baseline
            height = struct.unpack(">H", seg[1:3])[0]
            width = struct.unpack(">H", seg[3:5])[0]
            for c in range(seg[5]):
                cid = seg[6 + 3 * c]
                hv = seg[7 + 3 * c]
                comps.append({"id": cid, "h": hv >> 4, "v": hv & 0x0F,
                              "tq": seg[8 + 3 * c]})
        elif m == 0xC2:
            raise JpegError("progressive JPEG not supported")
        i += 2 + ln
    raise JpegError("no SOS found")


def _dc_grids(data):
    """Per-component block-average grids (luma first).

    Each cell is one 8x8 block's mean value 0..255. The DC coefficient of
    a block is (1/8) * sum(pixel - 128), so mean = DC / 8 + 128.
    """
    width, height, comps, quant, dc_t, ac_t, sos, scan = _parse_jpeg(data)
    by_id = {c["id"]: c for c in comps}
    for cid, td, ta in sos:
        c = by_id[cid]
        c["dc"] = dc_t[td]
        c["ac"] = ac_t[ta]
    hmax = max(c["h"] for c in comps)
    vmax = max(c["v"] for c in comps)
    mcux = (width + 8 * hmax - 1) // (8 * hmax)
    mcuy = (height + 8 * vmax - 1) // (8 * vmax)
    grids = []
    for c in comps:
        bw = (width * c["h"] + 8 * hmax - 1) // (8 * hmax)
        bh = (height * c["v"] + 8 * vmax - 1) // (8 * vmax)
        grids.append([[0.0] * bw for _ in range(bh)])
    br = BitReader(scan)
    preds = [0] * len(comps)
    for my in range(mcuy):
        for mx in range(mcux):
            for ci, c in enumerate(comps):
                for by in range(c["v"]):
                    for bx in range(c["h"]):
                        try:
                            cat = _decode_huff(br, c["dc"])
                        except _Restart:
                            preds = [0] * len(comps)
                            cat = _decode_huff(br, c["dc"])
                        preds[ci] += _receive_extend(br, cat)
                        mean = preds[ci] * quant[c["tq"]][0] / 8.0 + 128.0
                        gy = my * c["v"] + by
                        gx = mx * c["h"] + bx
                        if gy < len(grids[ci]) and gx < len(grids[ci][0]):
                            grids[ci][gy][gx] = mean
                        # Skip the block's AC coefficients: a block holds
                        # 63 AC positions, and once they are all filled
                        # there is no EOB symbol, so stop on the count.
                        ncoef = 0
                        while ncoef < 63:
                            try:
                                rs = _decode_huff(br, c["ac"])
                            except _Restart:
                                preds = [0] * len(comps)
                                break
                            run, size = rs >> 4, rs & 0x0F
                            if rs == 0xF0:
                                ncoef += 16
                                continue  # ZRL
                            if size == 0:
                                break  # EOB
                            ncoef += run + 1
                            _receive_extend(br, size)
    return grids


def _rgb_blocks(grids):
    """Resample chroma to the luma grid; one (r, g, b) tuple per block."""
    y = grids[0]
    cb = grids[1] if len(grids) > 1 else None
    cr = grids[2] if len(grids) > 2 else None
    rows = []
    for j, yrow in enumerate(y):
        out = []
        for i, yv in enumerate(yrow):
            if cb is not None:
                cj = min(len(cb) - 1, j * len(cb) // len(y))
                ci = min(len(cb[0]) - 1, i * len(cb[0]) // len(yrow))
                cbv = cb[cj][ci] - 128.0
                crv = cr[cj][ci] - 128.0
            else:
                cbv = crv = 0.0
            r = yv + 1.402 * crv
            g = yv - 0.34414 * cbv - 0.71414 * crv
            b = yv + 1.772 * cbv
            out.append((max(0.0, min(255.0, r)), max(0.0, min(255.0, g)),
                        max(0.0, min(255.0, b))))
        rows.append(out)
    return rows


def _yellow_fraction(rows):
    """Fraction of road-surface blocks that are yellow-marked.

    The road is the bottom 35% of the frame. A block counts when it is
    clearly yellow: bright red and green, dim blue, warm by a wide margin.
    A yellow centre line covers only a few blocks, so the fraction is
    small even when the line is unmistakable.
    """
    h = len(rows)
    cells = [c for r in rows[int(h * 0.65):] for c in r]
    if not cells:
        return 0.0
    n = sum(1 for (r, g, b) in cells
            if r > 150 and g > 120 and b < 110 and r - b > 60)
    return n / len(cells)


def guess(jpeg):
    """Where a photo was taken, from its JPEG bytes alone."""
    try:
        yellow = _yellow_fraction(_rgb_blocks(_dc_grids(jpeg)))
    except Exception:
        return PRIOR_LAT, PRIOR_LON
    if yellow > YELLOW_MIN:
        return AMERICAS_LAT, AMERICAS_LON
    return PRIOR_LAT, PRIOR_LON


def _get(path):
    # Proxies off, like the kit's SDK: the trial reaches only the env.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(ENV_URL + path)
    with opener.open(req, timeout=60) as res:
        return res.read()


def _post_answers(answers):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    body = json.dumps({"answers": answers}).encode("utf-8")
    req = urllib.request.Request(ENV_URL + "/answers", data=body,
                                 headers={"content-type": "application/json"},
                                 method="POST")
    with opener.open(req, timeout=60) as res:
        return res.status, res.read()


def main():
    if not ENV_URL:
        raise SystemExit("MAKEFUN_ENV_URL is not set")
    listed = json.loads(_get("/photos").decode("utf-8"))
    photos = [p for p in listed["photos"] if not p.get("answered")]
    answers = []
    for p in photos:
        try:
            jpeg = _get(p["url"])
        except Exception:
            jpeg = b""
        lat, lon = guess(jpeg) if jpeg else (PRIOR_LAT, PRIOR_LON)
        answers.append({"id": p["id"],
                        "lat": round(lat, 6), "lon": round(lon, 6)})
    if not answers:
        return 0
    status, _ = _post_answers(answers)
    sys.stderr.write("photo-geolocator: %d of %d photos answered: HTTP %d\n"
                     % (len(answers), len(photos), status))
    return 0 if status == 200 else 1


if __name__ == "__main__":
    raise SystemExit(main())
