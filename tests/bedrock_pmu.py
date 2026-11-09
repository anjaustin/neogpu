"""Fail-closed parsing for non-interval perf stat CSV evidence."""
import math

if not __debug__:
    raise RuntimeError('Bedrock evidence requires Python assertions enabled')


def counter_rows(rows):
    result = {}
    for row in rows:
        if not row or not row[0] or row[0].startswith('#'):
            continue
        if len(row) < 5 or row[2] in result:
            raise ValueError(f'malformed or duplicate perf row: {row!r}')
        try:
            value, runtime, running = float(row[0]), float(row[3]), float(row[4])
            valid = all(math.isfinite(v) for v in (value, runtime, running)) and value >= 0 and runtime > 0 and 0 <= running <= 100
        except ValueError:
            valid = False
        if not valid:
            result[row[2]] = {'available': False, 'raw': row}
            continue
        result[row[2]] = {'available': True, 'value': value, 'unit': row[1],
                         'runtime_ns': runtime, 'percent_running': running,
                         # This is reported precision, not an assertion about
                         # an unrounded kernel scheduling fraction.
                         'unmultiplexed': running == 100.0}
    return result


def selftest():
    def row(value='1', runtime='100', running='100.00'):
        return [value, '', 'cycles', runtime, running]
    assert counter_rows([row()])['cycles']['unmultiplexed']
    for running in ('99.50', '99.99'):
        assert not counter_rows([row(running=running)])['cycles']['unmultiplexed']
    for value, runtime, running in [('<not counted>', '0', '100'), ('nan', '1', '100'),
                                     ('inf', '1', '100'), ('-1', '1', '100'),
                                     ('1', '0', '100'), ('1', '1', '101')]:
        assert not counter_rows([row(value, runtime, running)])['cycles']['available']
    for rows in ([['malformed']], [row(), row()]):
        try:
            counter_rows(rows)
        except ValueError:
            continue
        raise AssertionError('malformed/duplicate evidence was accepted')


if __name__ == '__main__':
    selftest()
    print('PMU parser: scaling, unsupported, nonfinite, zero-runtime and malformed controls passed')
