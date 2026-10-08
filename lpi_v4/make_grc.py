#!/usr/bin/env python3
"""Generate the v4 GNU Radio Companion flowgraphs from lpi_grc.py's block sources.

Run after editing `make_grc_sources()` in lpi_grc.py:

    python make_grc.py            # writes tx_lpi_v4.grc and rx_lpi_v4.grc

The point of generating them instead of hand-editing YAML: the embedded
`epy_block` `_source` strings stay byte-identical to what
`python lpi_grc.py --print-sources` shows and to what the pytest loopback
exercises, so the GUI flowgraphs cannot drift away from the tested code.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import yaml  # noqa: E402


def q(s: str) -> str:
    """YAML single-line double-quoted scalar (what GRC writes for _source)."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def st(x, y, state="enabled"):
    return {"bus_sink": False, "bus_source": False, "bus_structure": None,
            "coordinate": [int(x), int(y)], "rotation": 0, "state": state}


def variable(name, value, comment=""):
    return {"name": name, "id": "variable",
            "parameters": {"alias": "", "comment": comment, "value": str(value)},
            "states": st(8, 8)}


def options(flow_id, title, desc):
    return {"options": {"parameters": {
        "author": "PriyalSingh23 / ShadowComm", "catch_exceptions": "True",
        "category": "[GRC Hier Blocks]", "cmake_opt": "", "comment": "",
        "copyright": "", "description": desc, "gen_cmake": "On",
        "gen_linking": "dynamic", "generate_options": "qt_gui",
        "hier_block_src_path": ".:", "id": flow_id, "max_nouts": "0",
        "output_language": "python", "placement": "(0,0)", "qt_qss_theme": "",
        "realtime_scheduling": "", "run": "True", "run_command": "{python} -u {filename}",
        "run_options": "prompt", "sizing_mode": "fixed", "thread_safe_setters": "",
        "title": title, "window_size": "(1100,700)"}, "states": st(8, 8)}}


# ------------------------------------------------------------------ TX ----
TX_BLOCKS = [
    options("tx_lpi_v4", "ShadowComm LPI v4 Transmitter",
            "Message -> AES/RS -> keyed spread frames -> CGAN waveform -> USRP. "
            "Needs the lpi_v4 folder (lpi_grc.py, lpi_core.py, exported .pt)."),
    variable("samp_rate", "245760",
             "B210 minimum is ~208.3 kS/s; 245760 = 61.44 MHz/250 (exact integer)"),
    variable("center_freq", "2.484e9", "same on TX and RX; the LPI gain is in the "
                                        "waveform, not in a low carrier"),
    variable("tx_gain", "0", "start at 0 dB and raise only until the RX locks"),
    variable("lpi_dir", '"' + HERE.replace("\\", "/") + '"',
             "folder containing lpi_core.py / lpi_grc.py / the exported weights"),
    variable("tx_bits_file", '"/tmp/lpi_bits.dat"',
             "raw uint8 file; each byte becomes 8 payload bits (see make_bits_file.py)"),
    variable("ckpt", '"' + os.path.join(HERE, "run", "lpi_v4.best.pt").replace("\\", "/") + '"'),
    variable("cfg_json", '"' + os.path.join(HERE, "run", "lpi_v4.best.json").replace("\\", "/") + '"'),
    variable("session_key", '"SESSION-KEY"', "must equal the RX key"),
]


def tx_grc(source: str):
    blocks = list(TX_BLOCKS)
    blocks += [
        {"name": "blocks_file_source_0", "id": "blocks_file_source", "parameters": {
            "alias": "", "comment": "packed bytes -> 8 bits each", "file": tx_bits_file_ref,
            "length": "0", "maxoutbuf": "0", "minoutbuf": "0", "repeat": "True",
            "type": "byte", "vlen": "1"}, "states": st(40, 200)},
        {"name": "blocks_unpack_k_bits_bb_0", "id": "blocks_unpack_k_bits_bb", "parameters": {
            "alias": "", "comment": "", "k": "8", "maxoutbuf": "0", "minoutbuf": "0"},
            "states": st(260, 200)},
        {"name": "digital_chunks_to_symbols_bc_0", "id": "digital_chunks_to_symbols_bc",
         "parameters": {"alias": "", "comment": "0 -> +1, 1 -> -1 (the block thresholds "
                         "the real part, so either mapping decodes)",
                         "constellation": "[-1, 1]", "dim": "1", "label": "",
                         "maxoutbuf": "0", "minoutbuf": "0", "num_rows": "1",
                         "num_ports": "1", "precision": "32", "symbol_table": ""},
         "states": st(470, 200)},
        {"name": "blocks_stream_to_vector_0", "id": "blocks_stream_to_vector", "parameters": {
            "alias": "", "comment": "one frame = 128 symbols", "maxoutbuf": "0",
            "minoutbuf": "0", "num_items": "128", "type": "complex", "vlen": "1"},
            "states": st(680, 200)},
        {"name": "epy_block_0", "id": "epy_block", "parameters": {
            "_source": source, "affinity": "", "alias": "",
            "comment": "LPI v4 keyed spread + pilot framing", "amp": "0.05",
            "ckpt": ckpt_ref, "cfg_json": cfg_ref, "key": key_ref, "lpi_dir": lpi_dir_ref,
            "maxoutbuf": "0", "minoutbuf": "0"}, "states": st(880, 200)},
        {"name": "blocks_vector_to_stream_0", "id": "blocks_vector_to_stream", "parameters": {
            "alias": "", "comment": "576 samples/frame = 64 pilot + 512 payload",
            "maxoutbuf": "0", "minoutbuf": "0", "num_items": "576", "type": "complex",
            "vlen": "1"}, "states": st(1090, 200)},
        {"name": "uhd_usrp_sink_0", "id": "uhd_usrp_sink", "parameters": {
            "affinity": "", "alias": "", "ant0": "TX/RX", "bw0": "0",
            "center_freq0": center_freq_ref, "clock_rate": "0e0", "clock_source0": "",
            "comment": "", "dev_addr": '""', "dev_args": "", "gain0": tx_gain_ref,
            "gain_type0": "default", "len_tag_name": '""', "lo_export0": "False",
            "lo_source0": "internal", "maxoutbuf": "0", "minoutbuf": "0", "nchan": "1",
            "num_mboards": "1", "otw": "", "samp_rate": samp_rate_ref, "sd_spec0": "",
            "show_lo_controls": "False", "start_time": "-1.0", "stream_args": "",
            "stream_chans": "[]", "sync": "none", "time_source0": "", "type": "fc32"},
            "states": st(1290, 200)},
        {"name": "qtgui_time_sink_x_0", "id": "qtgui_time_sink_x", "parameters": {
            "alias": "", "autoscale": "True", "axislabels": "['Time (s)', 'Amplitude']",
            "comment": "the waveform should *look* like noise on the scope",
            "ctrlpanel": "False", "entags": "True", "grid": "True", "gui_hint": "",
            "label1": "LPI IQ (I)", "label2": "LPI IQ (Q)", "label3": "", "label4": "",
            "legend": "True", "maxoutbuf": "0", "minoutbuf": "0", "name": "LPI v4 TX",
            "npoints": "2048", "srate": samp_rate_ref, "tr_chan": "0", "tr_delay": "0",
            "tr_level": "0.0", "tr_mode": "tr_auto", "tr_out_skipped": "False",
            "tr_tags": "False", "type": "complex", "update_time": "0.10",
            "width": "800", "yaxislabel": "Amplitude"}, "states": st(1290, 400)},
    ]
    return blocks


REFS = {}


def _refs():
    g = globals()
    for k, v in [("tx_bits_file_ref", "${tx_bits_file}"), ("ckpt_ref", "${ckpt}"),
                 ("cfg_ref", "${cfg_json}"), ("key_ref", "${session_key}"),
                 ("lpi_dir_ref", "${lpi_dir}"), ("center_freq_ref", "${center_freq}"),
                 ("tx_gain_ref", "${tx_gain}"), ("samp_rate_ref", "${samp_rate}"),
                 ("rx_gain_ref", "${rx_gain}"), ("cap_file_ref", "${capture_file}"),
                 ("out_file_ref", "${out_file}")]:
        g[k] = v


# ------------------------------------------------------------------ RX ----
def rx_grc(source: str):
    blocks = [
        options("rx_lpi_v4", "ShadowComm LPI v4 Receiver",
                "USRP IQ -> keyed-pilot sync (timing/CFO/phase) -> MF+ZF decode -> "
                "RS(42,34) + CRC-8 + AES -> text.  Prints to the console GRC runs in."),
        variable("samp_rate", "245760", "MUST equal the TX rate (lpi_config.json)"),
        variable("center_freq", "2.484e9", "same as TX"),
        variable("rx_gain", "30", "start ~30 dB; the LPI signal is *below* the noise floor"),
        variable("lpi_dir", '"' + HERE.replace("\\", "/") + '"'),
        variable("ckpt", '"' + os.path.join(HERE, "run", "lpi_v4.best.pt").replace("\\", "/") + '"'),
        variable("cfg_json", '"' + os.path.join(HERE, "run", "lpi_v4.best.json").replace("\\", "/") + '"'),
        variable("session_key", '"SESSION-KEY"'),
        variable("capture_file", '""',
                 'fill in a .cs16/.fc32 file to replay a recording instead of the radio'),
        variable("out_file", '"lpi_rx.log"', "one line per decoded burst"),
        {"name": "blocks_file_source_0", "id": "blocks_file_source", "parameters": {
            "alias": "", "comment": "OPTIONAL replay input -- leave the path empty and "
            "use the UHD source (double-click this block and set file= to replay)",
            "file": '""', "length": "0", "maxoutbuf": "0", "minoutbuf": "0",
            "repeat": "False", "type": "complex", "vlen": "1"},
            "states": st(40, 200, "disabled")},
        {"name": "uhd_usrp_source_0", "id": "uhd_usrp_source", "parameters": {
            "affinity": "", "alias": "", "ant0": "RX2", "bw0": "2e5", "dc_offs_0": "[0]",
            "dc_offset_mode0": "0", "gain0": rx_gain_ref, "gain_type0": "default",
            "gain_mode0": "Manual", "iq_imbal_0": "[0]", "iq_balance_mode0": "0",
            "lo_source0": "internal", "maxoutbuf": "0", "minoutbuf": "0", "nchan": "1",
            "num_mboards": "1", "otw": "", "radio0": '["addr=\\"\\""]',
            "rx_freq0": center_freq_ref, "rate0": samp_rate_ref,
            "rx_agc0": "Disabled", "samp_rate": samp_rate_ref, "sd_spec0": "",
            "show_lo_controls": "False", "stream_args": "", "stream_chans": "[]",
            "time_source0": "", "trigger_rate": "1e3", "tune_mame0": "Manual",
            "type": "fc32"}, "states": st(40, 400)},
        {"name": "epy_block_0", "id": "epy_block", "parameters": {
            "_source": source, "affinity": "", "alias": "",
            "comment": "sync + decode + CRC text (prints to stdout)",
            "ckpt": ckpt_ref, "cfg_json": cfg_ref, "key": key_ref, "lpi_dir": lpi_dir_ref,
            "maxoutbuf": "0", "minoutbuf": "0", "mode": '"ctr"', "out_file": out_file_ref,
            "verbose": "True", "window": "0"}, "states": st(320, 400)},
    ]
    return blocks


def _stringify_params(blk):
    """GRC stores every *parameter* as a string (and every state as a real bool /
    int), so normalise exactly that way."""
    out = dict(blk)
    par = blk.get("parameters")
    if isinstance(par, dict):
        p2 = {}
        for k, v in par.items():
            if isinstance(v, str) or v is None:
                p2[k] = "" if v is None else v
            elif isinstance(v, bool):
                p2[k] = "True" if v else "False"
            elif isinstance(v, (list, tuple)):
                p2[k] = list(v)
            elif isinstance(v, float):
                p2[k] = repr(v)
            else:
                p2[k] = str(v)
        out["parameters"] = p2
    return out


def main():
    _refs()
    from lpi_grc import make_grc_sources
    src_tx, src_rx = make_grc_sources()
    tx_blocks = tx_grc(src_tx)
    rx_blocks = rx_grc(src_rx)
    tx_conn = [["blocks_file_source_0", "'0'", "blocks_unpack_k_bits_bb_0", "'0'"],
               ["blocks_unpack_k_bits_bb_0", "'0'", "digital_chunks_to_symbols_bc_0", "'0'"],
               ["digital_chunks_to_symbols_bc_0", "'0'", "blocks_stream_to_vector_0", "'0'"],
               ["blocks_stream_to_vector_0", "'0'", "epy_block_0", "'0'"],
               ["epy_block_0", "'0'", "blocks_vector_to_stream_0", "'0'"],
               ["blocks_vector_to_stream_0", "'0'", "uhd_usrp_sink_0", "'0'"],
               ["blocks_vector_to_stream_0", "'0'", "qtgui_time_sink_x_0", "'0'"]]
    rx_conn = [["uhd_usrp_source_0", "'0'", "epy_block_0", "'0'"]]
    for fname, blocks, conn in (("tx_lpi_v4.grc", tx_blocks, tx_conn),
                                ("rx_lpi_v4.grc", rx_blocks, rx_conn)):
        opts = blocks[0]["options"]
        body = [_stringify_params(b) for b in blocks[1:]]
        txt = ("options:\n" + _indent(_dump(opts)) + "\nblocks:\n" + _dump(body)
               + "\nconnections:\n" + "\n".join(
                   f"- [{a}, {b}, {c}, {d}]" for a, b, c, d in conn) + "\n")
        with open(os.path.join(HERE, fname), "w", newline="\n") as f:
            f.write(txt)
        print("wrote", fname, os.path.getsize(os.path.join(HERE, fname)), "bytes")


def _dump(o):
    return yaml.dump(o, sort_keys=False, width=100000,
                     default_flow_style=None).rstrip("\n")


def _indent(txt, n=2):
    pad = " " * n
    return "\n".join(pad + l if l.strip() else l for l in txt.splitlines())


if __name__ == "__main__":
    main()
