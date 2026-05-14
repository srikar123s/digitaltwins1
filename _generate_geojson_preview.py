import json
import math
from pathlib import Path
import random

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
DECODER = json.JSONDecoder()
MARKER = '"features":'
SAMPLE_SIZE = 1800


def iter_features(path: Path):
    with path.open('r', encoding='utf-8') as handle:
        buffer = ''
        found_features = False
        found_array = False

        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk and not buffer:
                return
            buffer += chunk

            if not found_features:
                marker_index = buffer.find(MARKER)
                if marker_index == -1:
                    if len(buffer) > len(MARKER):
                        buffer = buffer[-len(MARKER):]
                    continue
                buffer = buffer[marker_index + len(MARKER):]
                found_features = True

            if found_features and not found_array:
                bracket_index = buffer.find('[')
                if bracket_index == -1:
                    continue
                buffer = buffer[bracket_index + 1:]
                found_array = True

            while True:
                buffer = buffer.lstrip()
                if not buffer:
                    break

                if buffer[0] == ']':
                    return

                if buffer[0] == ',':
                    buffer = buffer[1:]
                    continue

                try:
                    feature, end = DECODER.raw_decode(buffer)
                except json.JSONDecodeError:
                    break

                yield feature
                buffer = buffer[end:]


def normalize(value, minimum, maximum):
    span = maximum - minimum
    if span == 0:
        return 0.0
    return (value - minimum) / span


def categorize(values):
    total = len(values)
    low = sum(1 for v in values if v < 0.25)
    medium = sum(1 for v in values if 0.25 <= v < 0.5)
    high = sum(1 for v in values if 0.5 <= v < 0.75)
    extreme = sum(1 for v in values if v >= 0.75)

    def item(count):
        return {
            'count': count,
            'percent': round((count / total) * 100, 2) if total else 0.0,
        }

    return {
        'low': item(low),
        'medium': item(medium),
        'high': item(high),
        'extreme': item(extreme),
    }


def process_file(path: Path):
    preview_path = path.with_name(path.stem + '_summary.geojson')
    stats_path = path.with_name(path.stem + '_stats.json')

    print(f'Processing {path.name}')

    # First pass: count, ranges, and reservoir sample.
    count = 0
    ranges = {
        'probability': [math.inf, -math.inf],
        'flood': [math.inf, -math.inf],
        'landslide': [math.inf, -math.inf],
        'risk': [math.inf, -math.inf],
    }
    sample = []

    for feature in iter_features(path):
        count += 1
        props = feature.get('properties') or {}
        for key in ranges:
            value = props.get(key)
            if value is None:
                continue
            if value < ranges[key][0]:
                ranges[key][0] = value
            if value > ranges[key][1]:
                ranges[key][1] = value

        if len(sample) < SAMPLE_SIZE:
            sample.append(feature)
        else:
            replacement = random.randint(0, count - 1)
            if replacement < SAMPLE_SIZE:
                sample[replacement] = feature

    if count == 0:
        print('  empty file, skipped')
        return

    # Second pass: exact category counts with the final ranges.
    prob_values = []
    flood_norm_values = []
    landslide_norm_values = []

    flood_min, flood_max = ranges['flood']
    landslide_min, landslide_max = ranges['landslide']
    prob_min, prob_max = ranges['probability']
    risk_min, risk_max = ranges['risk']

    for feature in iter_features(path):
        props = feature.get('properties') or {}
        prob = props.get('probability')
        flood = props.get('flood')
        landslide = props.get('landslide', props.get('stress'))

        if prob is not None:
            prob_values.append(prob)
        if flood is not None:
            flood_norm_values.append(normalize(flood, flood_min, flood_max))
        if landslide is not None:
            landslide_norm_values.append(normalize(landslide, landslide_min, landslide_max))

    stats = {
        'gridSize': count,
        'probRange': {'min': prob_min, 'max': prob_max},
        'floodRange': {'min': flood_min, 'max': flood_max},
        'stressRange': {'min': landslide_min, 'max': landslide_max},
        'riskRange': {'min': risk_min, 'max': risk_max},
        'floodRisk': categorize(flood_norm_values),
        'landslideRisk': categorize(landslide_norm_values),
        'overallProbability': categorize(prob_values),
    }

    preview = {
        'type': 'FeatureCollection',
        'features': sample,
    }

    with preview_path.open('w', encoding='utf-8') as handle:
        json.dump(preview, handle, ensure_ascii=False)

    with stats_path.open('w', encoding='utf-8') as handle:
        json.dump(stats, handle, ensure_ascii=False)

    print(f'  wrote {preview_path.name} and {stats_path.name}')


def main():
    for path in sorted(OUTPUTS_DIR.glob('*.geojson')):
        if path.name.endswith('_summary.geojson') or path.name.endswith('_stats.json'):
            continue
        process_file(path)


if __name__ == '__main__':
    main()
