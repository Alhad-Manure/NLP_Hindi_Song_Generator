import unittest
from ptmlib.cpu import CpuCount


class CpuCountTestCase(unittest.TestCase):

    def test_cpu_count_success(self):
        cpu_count = CpuCount()
        cpu_count._cpu_count = 8  # override for testing

        self.assertEqual(8, cpu_count.total_count(), 'total cpu count should be 8')
        self.assertEqual(7, cpu_count.adjusted_count(), 'adjusted cpu count should be 7')
        self.assertEqual(5, cpu_count.adjusted_count(3), 'adjusted cpu count should be 5')
        self.assertEqual(6, cpu_count.adjusted_count_by_percent(), 'adjusted cpu count should be 6')
        self.assertEqual(2, cpu_count.adjusted_count_by_percent(0.75), 'adjusted cpu count should be 2')

    def test_cpu_exclude_all_success(self):
        cpu_count = CpuCount()
        total_cpu_count = cpu_count.total_count()
        self.assertEqual(1, cpu_count.adjusted_count(total_cpu_count), 'adjusted cpu count should be 1 at a minimum')
        self.assertEqual(1, cpu_count.adjusted_count_by_percent(1.0), 'adjusted cpu count should be 1 at a minimum')

    @staticmethod
    def test_print_stats_success():
        cpu_count = CpuCount()
        cpu_count.print_stats()


if __name__ == '__main__':
    unittest.main()
