import time
from contextlib import contextmanager

class Timer:
    _UNIT_FACTORS = {
        's': 1.0,
        'ms': 1e3,
        'us': 1e6,
        'ns': 1e9,
    }

    def __init__(self, unit: str = 's', precision: int = 6):
        """
        Args:
            unit: one of 's', 'ms', 'us', 'ns' for seconds, milliseconds, microseconds, or nanoseconds.
            precision: number of decimal places when printing.
        """
        if unit not in self._UNIT_FACTORS:
            raise ValueError(f"Unsupported unit '{unit}'. Choose from {list(self._UNIT_FACTORS)}.")
        self.unit = unit
        self._factor = self._UNIT_FACTORS[unit]
        self.precision = precision
        self._records: dict[str, dict[str, float | int]] = {}

    @contextmanager
    def name(self, section: str):
        """Time a block and accumulate total & count for `section`."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            rec = self._records.setdefault(section, {'total': 0.0, 'count': 0})
            rec['total'] += elapsed
            rec['count'] += 1

    def print_results(self):
        """Print total, count, and average in the configured unit."""
        if not self._records:
            print("No timings recorded.")
            return

        unit_label = self.unit
        header_fmt = f"{{:<20}} {{:>12}} {{:>8}} {{:>14}}"
        row_fmt    = f"{{:<20}} {{:12.{self.precision}f}} {{:8d}} {{:14.{self.precision}f}}"

        print(header_fmt.format("Section", f"Total ({unit_label})", "Count", f"Average ({unit_label})"))
        print("-" * (20 + 1 + 12 + 1 + 8 + 1 + 14))
        for section, rec in self._records.items():
            total_in_unit = rec['total'] * self._factor
            count        = rec['count']
            avg_in_unit  = (rec['total'] / count) * self._factor
            print(row_fmt.format(section, total_in_unit, count, avg_in_unit))
