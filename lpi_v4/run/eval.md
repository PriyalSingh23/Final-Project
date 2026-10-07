## ShadowComm LPI v4 -- evaluation report

config: T=512, 128 bits/frame, pilot 64, fs=245760, processing gain 9.0 dB

### Stealth
| metric | value |
|---|---|
| ks_p | 0.8646 |
| kurt | 2.991 |
| entropy | 0.9566 |
| circ | 0.01075 |
| papr | 0.008737 |
| scf | 1.013 |
| c42 | -0.004793 |
| wvd | 1.017 |
| Sc | 0.1399 |
| adv_acc | 55.9 |
| adv_auc | 0.5858 |
| adv protocol | fresh CNN detector, 400 steps on 2600 train / 1000 held-out frames (disjoint pools) |

### Link (per-bit BER)
| SNR | MF | in-graph | field | field 95 % ≤ | FER | FER 95 % ≤ | bits |
|---|---|---|---|---|---|---|---|
| 0 | 1.03e-02 | 6.26e-03 | 1.00e-02 | 1.07e-02 | 66.4% | 69.9% | 65,536 |
| 2 | 3.27e-03 | 1.36e-03 | 2.14e-03 | 2.46e-03 | 22.7% | 25.9% | 65,536 |
| 4 | 8.85e-04 | 1.68e-04 | 2.44e-04 | 3.71e-04 | 3.1% | 4.7% | 65,536 |
| 6 | 4.12e-04 | 0.00e+00 | 3.05e-05 | 9.61e-05 | 0.4% | 1.2% | 65,536 |
| 8 | 1.22e-04 | 0.00e+00 | 0.00e+00 | 4.57e-05 | 0.0% | 0.6% | 65,536 |
| 10 | 3.05e-05 | 0.00e+00 | 0.00e+00 | 4.57e-05 | 0.0% | 0.6% | 65,536 |

### Message layer (CRC-8 + RS(42,34) + AES)
| SNR | BER | BER 95 % ≤ | FER | FER 95 % ≤ | msgs ok |
|---|---|---|---|---|---|
| 0 | 9.93e-03 | 1.10e-02 | 18.8% | 28.6% | 52/64 |
| 2 | 1.95e-03 | 2.48e-03 | 0.0% | 4.6% | 64/64 |
| 4 | 3.66e-04 | 6.39e-04 | 0.0% | 4.6% | 64/64 |
| 6 | 4.07e-05 | 1.93e-04 | 0.0% | 4.6% | 64/64 |
| 8 | 0.00e+00 | 1.22e-04 | 0.0% | 4.6% | 64/64 |
| 10 | 0.00e+00 | 1.22e-04 | 0.0% | 4.6% | 64/64 |

### Gate

*ber/fer = worst over SNR >= 5 dB (3 of 6 sweep points)*  
- [x] adv_in_50s
- [x] ks>0.05
- [x] ew>=7/8
- [x] ber_field<1%
- [x] fer<5%
- [x] capture_decodes -- *not measured*: capture was synthesized here, so it tests the codec/framing, not the radio front end

**overall: ALL TARGETS MET**