"""
uhd_io.py -- thin, version-tolerant UHD wrapper used by tx_usrp.py / rx_usrp.py.
================================================================================

UHD's Python bindings moved around between 3.15 / 4.0 / 4.1 / 4.6 (module names,
`get_streamer` vs `set_tx_streamer`, `stream_args` as a string vs an object).
Everything Radio-side that our project needs is: "put these complex samples on
the air at rate R" and "give me N complex samples at rate R".  This module does
that and nothing else, so a version difference never breaks the LPI logic.

No GNU Radio required: `pip install uhd` is not a thing -- the bindings come with
the Ettus UHD installer or the `radioconda` / `conda-forge::uhd` package.
"""
from __future__ import annotations

import time

import numpy as np

SC16_MIN_FS = 208_334.0        # B200/B210 UHD minimum sample rate


def pick_rate(want: float = 245_760.0, master: float = 61.44e6) -> float:
    """Round `want` up to a value the B210 can synthesise exactly (integer
    decimation of the master clock) and that is above the USB limit."""
    r = max(float(want), SC16_MIN_FS)
    div = max(1, int(round(master / r)))
    return master / div


def iq_to_sc16(iq: np.ndarray, headroom: float = 0.9) -> np.ndarray:
    """complex64 -> packed 'sc16' int32 (real low, imag high), peak-normalised."""
    m = float(np.abs(iq).max()) + 1e-12
    s = headroom * 32767.0 / m
    re = np.clip(np.round(iq.real * s), -32768, 32767).astype(np.int16)
    im = np.clip(np.round(iq.imag * s), -32768, 32767).astype(np.int16)
    return (re.astype(np.uint16).astype(np.int32)
            | (im.astype(np.uint16).astype(np.int32) << 16)).astype(np.int32)


def sc16_to_iq(raw: np.ndarray) -> np.ndarray:
    v = np.asarray(raw).astype(np.uint32 if np.asarray(raw).dtype.itemsize >= 4
                               else np.uint16)
    if v.dtype == np.uint32:
        re = ((v & 0xFFFF).astype(np.uint16)).view(np.int16)
        im = ((v >> 16) & 0xFFFF).astype(np.uint16)
        im = im.view(np.int16)
        return (re.astype(np.float32) + 1j * im.astype(np.float32)) / 32768.0
    re = v.view(np.int16)                       # interleaved int16 pairs
    z = re.reshape(-1, 2)
    return (z[:, 0].astype(np.float32) + 1j * z[:, 1].astype(np.float32)) / 32768.0


class Radio:
    """One USRP, one direction, one channel (0).  Use as a context manager."""

    def __init__(self, addr: str = "", rate: float = 245_760.0, freq: float = 2.484e9,
                 gain: float = 0.0, ant: str = "", clock: str = "", time_source: str = "",
                 lo_export: bool = False, direction: str = "tx", timeout: float = 2.0):
        self.direction = direction.lower()
        self.want_rate = float(rate)
        self.rate = float(rate)
        self.freq = float(freq)
        self.timeout = timeout
        self.last_overruns = 0
        try:
            import uhd                                # noqa: F401
        except ImportError as e:
            raise SystemExit(
                "\n[uhd] the python bindings are not importable (%s).\n"
                "      Windows : conda install -c radioconda uhd   (or the Ettus installer)\n"
                "      Linux   : sudo apt install libuhd-dev uhd-host  &&  python3 -c 'import uhd'\n"
                "      check   : uhd_find_devices / uhd_fft\n"
                "      no radio? use --file / --capture .npz round-trips instead.\n" % e)
        import uhd
        self.uhd = uhd
        self.usrp = uhd.usrp.MultiUSRP("addr=%s" % addr if addr and ":" not in addr
                                       and "/" not in addr and "." in addr else addr)
        self._setup(freq, gain, ant, clock, time_source, lo_export)

    # ------------------------------------------------------------------
    def _setup(self, freq, gain, ant, clock, time_source, lo_export):
        uhd, usrp, d = self.uhd, self.usrp, self.direction
        setter = getattr(usrp, f"set_{d}_rate")
        try:
            setter(self.want_rate, 0)
        except Exception as e:
            print(f"[uhd] set_{d}_rate({self.want_rate:.0f}) refused ({e})")
        self.rate = float(getattr(usrp, f"get_{d}_rate")(0))
        if self.rate < 0.98 * self.want_rate:
            alt = pick_rate(self.want_rate)
            print(f"[uhd] radio chose {self.rate:.0f} S/s; retrying with {alt:.0f} S/s")
            try:
                setter(alt, 0)
                self.rate = float(getattr(usrp, f"get_{d}_rate")(0))
            except Exception:
                pass
        if clock:
            usrp.set_clock_source(clock, 0)
        if time_source:
            usrp.set_time_source(time_source, 0)
        tr = uhd.libpyuhd.types.tune_request(self.freq)
        try:
            r = getattr(usrp, f"set_{d}_freq")(tr, 0)
            got = getattr(usrp, f"get_{d}_freq")(0)
            if abs(got - self.freq) > 2e5:
                print(f"[uhd] WARNING tuned to {got/1e6:.4f} MHz (wanted "
                      f"{self.freq/1e6:.4f}); the radio cannot reach that frequency")
        except Exception as e:
            print(f"[uhd] set_{d}_freq failed ({e}); trying bare float")
            getattr(usrp, f"set_{d}_freq")(self.freq, 0)
        if lo_export:
            for k in ("set_tx_lo_export_enabled", "set_rx_lo_export_enabled"):
                try:
                    getattr(usrp, k)(True, 0)
                except Exception:
                    pass
        try:
            if d == "rx":
                usrp.set_rx_agc(False, 0)
        except Exception:
            pass
        try:
            getattr(usrp, f"set_{d}_gain")(float(gain), 0)
        except Exception as e:
            print(f"[uhd] set_{d}_gain failed: {e}")
        if ant:
            try:
                getattr(usrp, f"set_{d}_antenna")(ant, 0)
            except Exception as e:
                print(f"[uhd] antenna '{ant}' not accepted ({e}); "
                      f"options: {getattr(usrp, f'get_{d}_antennas')(0)}")
        self.gain = float(getattr(usrp, f"get_{d}_gain")(0))
        self.ant = ""
        try:
            self.ant = getattr(usrp, f"get_{d}_antenna")(0)
        except Exception:
            pass

    def summary(self) -> str:
        return (f"{self.direction.upper()} rate={self.rate:.0f} S/s  "
                f"f={self.freq/1e6:.4f} MHz  gain={self.gain:.1f} dB  "
                f"ant={self.ant or '-'}  clock={self.usrp.get_clock_source(0)}")

    # ------------------------------------------------------------------
    def send(self, iq: np.ndarray, start_time: float | None = None) -> int:
        """iq: complex64 (any length).  Returns the number of samples accepted."""
        pkt = iq_to_sc16(iq / (np.abs(iq).max() + 1e-12), 0.95)
        usrp = self.usrp
        kw = {}
        if start_time is not None:
            kw["start_time"] = float(start_time)
        for name in ("send_num_samples", "send"):
            fn = getattr(usrp, name, None)
            if fn is None:
                continue
            try:
                if name == "send_num_samples":
                    return int(fn(pkt, self.rate, _stream_args(self.uhd, 1),
                                  timeout=max(1.0, iq.size / self.rate + 2.0), **kw))
                return int(self._streamer_send(pkt, start_time))
            except TypeError as e:
                print(f"[uhd] {name} signature mismatch: {e}")
            except Exception as e:
                print(f"[uhd] {name} failed: {e}")
        raise RuntimeError("no usable UHD transmit API -- run `python -c "
                           "\"import uhd; print([m for m in dir(uhd.usrp.MultiUSRP) if 'send' in m])\"`")

    def _streamer_send(self, pkt, start_time=None):
        usrp = self.usrp
        if hasattr(usrp, "set_tx_streamer"):
            st = usrp.set_tx_streamer(1, "sc16")
        else:
            st = usrp.get_streamer(["sc16"], self.uhd.stream_dirs.USB_TX)
        meta = self.uhd.libpyuhd.types.tx_metadata()
        if start_time is not None:
            meta.start_of_burst = True
            meta.end_of_burst = True
            meta.time_spec = usrp.get_time_now().get_timed_spec(start_time)
        n = st.send(pkt, meta, pkt.size, 1.0)
        st.flush()
        return n

    def recv(self, n_samples: int, timeout: float | None = None,
             start_time: float | None = None) -> np.ndarray:
        """-> complex64 array of <= n_samples samples."""
        usrp = self.usrp
        to = self.timeout if timeout is None else timeout
        kw = {}
        if start_time is not None:
            kw["start_time"] = float(start_time)
        for name in ("recv_num_samples", "recv"):
            fn = getattr(usrp, name, None)
            if fn is None:
                continue
            try:
                if name == "recv_num_samples":
                    out = fn(n_samples, self.rate, _stream_args(self.uhd, 1),
                             timeout=max(0.5, n_samples / self.rate + 1.0), **kw)
                    out = np.asarray(out)
                    if out.dtype == np.complex64 or out.dtype == np.complex128:
                        return out.astype(np.complex64)
                    return sc16_to_iq(out).astype(np.complex64)
                return self._streamer_recv(n_samples, to)
            except TypeError as e:
                print(f"[uhd] {name} signature mismatch: {e}")
            except Exception as e:
                print(f"[uhd] {name} failed: {e}")
        raise RuntimeError("no usable UHD receive API")

    def _streamer_recv(self, n, timeout):
        usrp = self.usrp
        if hasattr(usrp, "set_rx_streamer"):
            st = usrp.set_rx_streamer(1, "sc16")
        else:
            st = usrp.get_streamer(["sc16"], self.uhd.stream_dirs.USB_RX)
        buf = np.zeros(n, np.int32)
        meta = self.uhd.libpyuhd.types.rx_metadata()
        cmd = self.uhd.libpyuhd.types.stream_cmd(
            self.uhd.stream_dirs.USB_RX, self.uhd.stream_cmds.START)
        issue = getattr(usrp, "issue_stream_cmd", None) or getattr(st, "issue_stream_cmd")
        issue(cmd)
        got = 0
        t0 = time.perf_counter()
        while got < n and time.perf_counter() - t0 < timeout:
            k = st.recv(buf[got:], meta, n - got, 0.5)
            if meta.error_flags and getattr(meta.error_flags, "rOVR", False):
                self.last_overruns += 1
            got += int(k)
        issue(self.uhd.libpyuhd.types.stream_cmd(self.uhd.stream_dirs.USB_RX,
                                                 self.uhd.stream_cmds.STOP))
        return sc16_to_iq(buf[:got]).astype(np.complex64)

    def close(self):
        try:
            self.usrp.set_rx_enable(False, 0)
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
        return False


def _stream_args(uhd, nchan):
    try:
        sa = uhd.libpyuhd.types.stream_args()
        sa.channels = list(range(nchan))
        sa.otw = "sc16"
        return sa
    except Exception:
        return "sc16"


def find_devices() -> str:
    try:
        import subprocess
        return subprocess.run(["uhd_find_devices"], capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except Exception as e:
        return f"(uhd_find_devices unavailable: {e})"
