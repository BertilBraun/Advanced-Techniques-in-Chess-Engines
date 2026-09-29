import glob
import json

for path in sorted(glob.glob('/workspace/matrix/results/*.json')):
    report = json.load(open(path))
    float16 = report['float16']['timing']['median_positions_per_second']
    row = [path.rsplit('/', 1)[-1][:-5].ljust(18), f'fp16 {float16:9,.0f}']
    for candidate in report['candidates']:
        fidelity = candidate['fidelity_against_float16']
        row.append(
            f"{candidate['precision']} {candidate['timing']['median_positions_per_second']:9,.0f} "
            f"({candidate['speedup_over_float16']:.2f}x, top1 {fidelity['policy_top1_agreement']:.3f}, "
            f"KL {fidelity['mean_policy_kl_divergence']:.4f}, WDL {fidelity['wdl_mean_absolute_error']:.4f}, "
            f"EV {fidelity['expected_value_mean_absolute_error']:.4f}, nodes {candidate['quantized_node_count']})"
        )
    print(' | '.join(row))
print(report['hardware'], report['fidelity_positions'], report['calibration_positions'], report['calibration_fidelity_overlap_positions'])
