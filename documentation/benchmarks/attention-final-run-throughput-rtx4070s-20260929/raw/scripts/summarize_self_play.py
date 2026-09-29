import glob
import json

for path in sorted(glob.glob('/workspace/matrix/self-play/*/*/summary.json')):
    summary = json.load(open(path))
    measurement, inference = summary['measurement'], summary['inference']
    name = path.split('/')[4]
    extra = {key: summary[key] for key in summary if key in ('gpu', 'cpu', 'resources', 'utilization')}
    print(f"{name:18s} searches/s {measurement['searches_per_second']:9,.0f}  evaluations/s "
          f"{inference['evaluations'] / measurement['makespan_seconds']:9,.0f}  batch {inference['average_batch_size']:.1f}  "
          f"{str(extra)[:160]}")
print(sorted(summary.keys()))
