"""Minimal site-geometry registry for the on-the-fly projection path.

Data copied verbatim from lutgen.sites.SITES — the two packages must stay
independent, so we do NOT import from lut-gen.
"""

from __future__ import annotations

from dataclasses import dataclass

from polar_reader.exceptions import SiteNotFoundError


@dataclass(frozen=True, slots=True)
class SiteGeometry:
    lat: float
    lon: float
    first_range_gate_m: float
    range_gate_spacing_m: float
    max_range_m: float
    n_azimuths: int


def _wsr(lat: float, lon: float) -> SiteGeometry:
    return SiteGeometry(lat, lon, 2125.0, 250.0, 230_000.0, 720)


def _tdwr(lat: float, lon: float) -> SiteGeometry:
    return SiteGeometry(lat, lon, 2125.0, 250.0, 230_000.0, 360)


# fmt: off
# Source: lutgen.sites.SITES — coordinates from official NWS/ROC/FAA records.
SITE_GEOMETRY: dict[str, SiteGeometry] = {
    # ── CONUS WSR-88D ──────────────────────────────────────────────────────────
    "KABR": _wsr(45.4558,  -98.4131),
    "KABX": _wsr(35.1497,  -106.8240),
    "KAKQ": _wsr(36.9839,  -77.0078),
    "KAMA": _wsr(35.2333,  -101.7092),
    "KAMX": _wsr(25.6111,  -80.4128),
    "KAPX": _wsr(44.9072,  -84.7197),
    "KARX": _wsr(43.8228,  -91.1912),
    "KATX": _wsr(48.1945,  -122.4958),
    "KBBX": _wsr(39.4956,  -121.6317),
    "KBGM": _wsr(42.1997,  -75.9847),
    "KBHX": _wsr(40.4981,  -124.2919),
    "KBIS": _wsr(46.7708,  -100.7603),
    "KBLX": _wsr(45.8539,  -108.6067),
    "KBMX": _wsr(33.1722,  -86.7697),
    "KBOX": _wsr(41.9558,  -71.1369),
    "KBRO": _wsr(25.9158,  -97.4189),
    "KBUF": _wsr(42.9489,  -78.7369),
    "KBYX": _wsr(24.5975,  -81.7033),
    "KCAE": _wsr(33.9489,  -81.1183),
    "KCBW": _wsr(46.0392,  -67.8067),
    "KCBX": _wsr(43.4911,  -116.2344),
    "KCCX": _wsr(40.9228,  -78.0039),
    "KCLE": _wsr(41.4133,  -81.8597),
    "KCLX": _wsr(32.6553,  -81.0422),
    "KCRP": _wsr(27.7839,  -97.5111),
    "KCYS": _wsr(41.1519,  -104.8061),
    "KDAX": _wsr(38.5011,  -121.6778),
    "KDDC": _wsr(37.7208,  -99.9689),
    "KDFX": _wsr(29.2731,  -100.2806),
    "KDGX": _wsr(32.2800,  -89.9844),
    "KDLH": _wsr(46.8369,  -92.2097),
    "KDMX": _wsr(41.7311,  -93.7228),
    "KDOX": _wsr(38.8258,  -75.4400),
    "KDTX": _wsr(42.6997,  -83.4717),
    "KDVN": _wsr(41.6117,  -90.5808),
    "KDYX": _wsr(32.5386,  -99.2544),
    "KEAX": _wsr(38.8103,  -94.2644),
    "KEMX": _wsr(31.8933,  -110.6303),
    "KENX": _wsr(42.5864,  -74.0642),
    "KEOX": _wsr(31.4603,  -85.4594),
    "KEPZ": _wsr(31.8731,  -106.6981),
    "KESX": _wsr(35.7011,  -114.8917),
    "KEVX": _wsr(30.5644,  -85.9217),
    "KEWX": _wsr(29.7039,  -98.0286),
    "KEYX": _wsr(35.0978,  -117.5608),
    "KFCX": _wsr(37.0242,  -80.2742),
    "KFDR": _wsr(34.3622,  -98.9764),
    "KFDX": _wsr(34.6353,  -103.6297),
    "KFFC": _wsr(33.3636,  -84.5658),
    "KFSD": _wsr(43.5878,  -96.7294),
    "KFSX": _wsr(34.5742,  -111.1983),
    "KFTG": _wsr(39.7867,  -104.5458),
    "KFWS": _wsr(32.5728,  -97.3028),
    "KGGW": _wsr(48.2064,  -106.6250),
    "KGJX": _wsr(39.0622,  -108.2139),
    "KGLD": _wsr(39.3669,  -101.7006),
    "KGRB": _wsr(44.4986,  -88.1111),
    "KGRK": _wsr(30.7217,  -97.3831),
    "KGRR": _wsr(42.8939,  -85.5447),
    "KGSP": _wsr(34.8833,  -82.2203),
    "KGWX": _wsr(33.8969,  -88.3289),
    "KGYX": _wsr(43.8914,  -70.2564),
    "KHDX": _wsr(33.0769,  -106.1228),
    "KHGX": _wsr(29.4719,  -95.0789),
    "KHNX": _wsr(36.3142,  -119.6322),
    "KHPX": _wsr(36.7369,  -87.2847),
    "KHTX": _wsr(34.9306,  -86.0833),
    "KICT": _wsr(37.6544,  -97.4428),
    "KICX": _wsr(37.5908,  -112.8625),
    "KILN": _wsr(39.4703,  -83.8217),
    "KILX": _wsr(40.1506,  -89.3367),
    "KIND": _wsr(39.7075,  -86.2803),
    "KINX": _wsr(36.1753,  -95.5647),
    "KIWA": _wsr(33.2892,  -111.6700),
    "KIWX": _wsr(41.3589,  -85.7000),
    "KJAX": _wsr(30.4847,  -81.7019),
    "KJGX": _wsr(32.6750,  -83.3514),
    "KJKL": _wsr(37.5908,  -83.3131),
    "KLBB": _wsr(33.6542,  -101.8142),
    "KLCH": _wsr(30.1253,  -93.2161),
    "KLGX": _wsr(47.1158,  -124.1069),
    "KLIX": _wsr(30.3367,  -89.8256),
    "KLNX": _wsr(41.9578,  -100.5761),
    "KLOT": _wsr(41.6044,  -88.0847),
    "KLRX": _wsr(40.7397,  -116.8028),
    "KLSX": _wsr(38.6989,  -90.6828),
    "KLTX": _wsr(33.9892,  -78.4292),
    "KLVX": _wsr(37.9753,  -85.9439),
    "KLWX": _wsr(38.9753,  -77.4778),
    "KLZK": _wsr(34.8364,  -92.2619),
    "KMAF": _wsr(31.9433,  -102.1894),
    "KMAX": _wsr(42.0811,  -122.7172),
    "KMBX": _wsr(48.3931,  -100.8644),
    "KMHX": _wsr(34.7761,  -76.8764),
    "KMLB": _wsr(28.1133,  -80.6542),
    "KMOB": _wsr(30.6794,  -88.2397),
    "KMPX": _wsr(44.8489,  -93.5653),
    "KMQT": _wsr(46.5311,  -87.5483),
    "KMRX": _wsr(36.1683,  -83.4017),
    "KMSX": _wsr(47.0411,  -113.9861),
    "KMTX": _wsr(41.2628,  -112.4481),
    "KMUX": _wsr(37.1553,  -121.8983),
    "KMVX": _wsr(47.5281,  -97.3253),
    "KMXX": _wsr(32.5367,  -85.7897),
    "KNKX": _wsr(32.9189,  -117.0419),
    "KNQA": _wsr(35.3447,  -89.8733),
    "KOAX": _wsr(41.3203,  -96.3664),
    "KOHX": _wsr(36.2472,  -86.5625),
    "KOKX": _wsr(40.8656,  -72.8644),
    "KOTX": _wsr(47.6803,  -117.6267),
    "KPAH": _wsr(37.0683,  -88.7717),
    "KPBZ": _wsr(40.5317,  -80.2181),
    "KPDT": _wsr(45.6906,  -118.8525),
    "KPOE": _wsr(31.1556,  -92.9761),
    "KPUX": _wsr(38.4595,  -104.1811),
    "KRAX": _wsr(35.6656,  -78.4897),
    "KRGX": _wsr(39.7542,  -119.4611),
    "KRIW": _wsr(43.0661,  -108.4772),
    "KRLX": _wsr(38.3111,  -81.7228),
    "KSFX": _wsr(43.1056,  -112.6861),
    "KSGF": _wsr(37.2353,  -93.4006),
    "KSHV": _wsr(32.4508,  -93.8411),
    "KSJT": _wsr(31.3711,  -100.4922),
    "KSOX": _wsr(33.8178,  -117.6361),
    "KSRX": _wsr(35.2906,  -94.3619),
    "KTBW": _wsr(27.7056,  -82.4017),
    "KTFX": _wsr(47.4597,  -111.3856),
    "KTLH": _wsr(30.3978,  -84.3289),
    "KTLX": _wsr(35.3331,  -97.2778),
    "KTWX": _wsr(38.9969,  -96.2325),
    "KTYX": _wsr(43.7556,  -75.6800),
    "KUDX": _wsr(44.1247,  -102.8297),
    "KUEX": _wsr(40.3211,  -98.4417),
    "KVAX": _wsr(30.8903,  -83.0019),
    "KVBX": _wsr(34.8381,  -120.3981),
    "KVNX": _wsr(36.7408,  -98.1278),
    "KVTX": _wsr(34.4117,  -119.1794),
    "KVWX": _wsr(38.2600,  -87.7242),
    "KYUX": _wsr(32.4953,  -114.6558),
    # ── Alaska WSR-88D ─────────────────────────────────────────────────────────
    "PABC": _wsr(60.7922,  -161.8764),
    "PACG": _wsr(56.8528,  -135.5292),
    "PAEC": _wsr(64.5114,  -165.2950),
    "PAHG": _wsr(60.7258,  -151.3514),
    "PAIH": _wsr(59.4614,  -146.3006),
    "PAKC": _wsr(58.6794,  -156.6294),
    "PAPD": _wsr(65.0353,  -147.5008),
    # ── Hawaii WSR-88D ─────────────────────────────────────────────────────────
    "PHKI": _wsr(21.8944,  -159.5525),
    "PHKM": _wsr(20.1256,  -155.7783),
    "PHMO": _wsr(21.1328,  -157.1800),
    "PHWA": _wsr(19.0950,  -155.5689),
    # ── Puerto Rico WSR-88D ────────────────────────────────────────────────────
    "TJUA": _wsr(18.1156,  -66.0781),
    # ── Guam WSR-88D ──────────────────────────────────────────────────────────
    "PGUA": _wsr(13.4544,  144.8114),
    # ── TDWR (n_azimuths=360) ─────────────────────────────────────────────────
    "TATL": _tdwr(33.6367,  -84.4281),
    "TBNA": _tdwr(36.0739,  -86.7503),
    "TBOS": _tdwr(42.1675,  -71.0033),
    "TBWI": _tdwr(39.1019,  -76.8914),
    "TBUF": _tdwr(42.9400,  -78.7319),
    "TCLE": _tdwr(41.5078,  -81.6831),
    "TCRW": _tdwr(35.2144,  -80.9431),
    "TCVG": _tdwr(39.0481,  -84.6681),
    "TADW": _tdwr(38.6953,  -76.8453),
    "TDAY": _tdwr(39.9019,  -84.2192),
    "TDEN": _tdwr(39.8561,  -104.6561),
    "TDET": _tdwr(42.2119,  -83.3539),
    "TDFW": _tdwr(32.8978,  -97.0311),
    "TIAD": _tdwr(38.9444,  -77.4569),
    "TIDS": _tdwr(39.7325,  -86.2942),
    "TJFK": _tdwr(40.6214,  -73.7789),
    "TKCI": _tdwr(39.2978,  -94.7139),
    "TLAS": _tdwr(36.0828,  -115.1525),
    "TLAX": _tdwr(33.9425,  -118.4081),
    "TMCO": _tdwr(28.4292,  -81.3089),
    "TMDW": _tdwr(41.7858,  -87.7517),
    "TMEM": _tdwr(35.0425,  -89.9767),
    "TMIA": _tdwr(25.9075,  -80.2789),
    "TMKE": _tdwr(42.9469,  -87.8975),
    "TMSP": _tdwr(44.8819,  -93.2289),
    "TMSY": _tdwr(29.9933,  -90.2589),
    "TOMA": _tdwr(41.3025,  -96.0119),
    "TORD": _tdwr(41.9742,  -87.9075),
    "TPHL": _tdwr(39.8714,  -75.2411),
    "TPHX": _tdwr(33.4289,  -112.0156),
    "TPIT": _tdwr(40.4914,  -80.2325),
    "TPVD": _tdwr(41.7239,  -71.4281),
    "TRDU": _tdwr(35.8778,  -78.7875),
    "TSDF": _tdwr(38.1742,  -85.7361),
    "TSEA": _tdwr(47.4431,  -122.3014),
    "TSJC": _tdwr(37.3625,  -121.9281),
    "TSJU": _tdwr(18.4394,  -66.0014),
    "TSLC": _tdwr(40.7789,  -111.9619),
    "TSTL": _tdwr(38.7489,  -90.3703),
    "TTPA": _tdwr(27.9742,  -82.5331),
}
# fmt: on


def get_site_geometry(site_id: str) -> SiteGeometry:
    """Return SiteGeometry for site_id, raising SiteNotFoundError if missing."""
    geom = SITE_GEOMETRY.get(site_id)
    if geom is None:
        raise SiteNotFoundError(f"Site '{site_id}' not in site registry")
    return geom
