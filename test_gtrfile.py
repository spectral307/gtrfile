from gtrfile import GtrFile
import numpy as np
from os import SEEK_SET, SEEK_CUR, SEEK_END
import unittest


class TestGtrFile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.__gtrfile = GtrFile("test_record.gtr")
        cls.__target_dtype = np.dtype(
            [("вход 1", np.float32), ("вход 3", np.float32)])

    @classmethod
    def tearDownClass(cls):
        cls.__gtrfile.close()

    def setUp(self):
        self.__gtrfile.seek_item(0)

    def test_init_should_read_header(self):
        target_header = {
            "device": "AP6300",
            "rate": 5000,
            "time": 5,
            "inputs": [
                {
                    "n": 0,
                    "name": "вход 1",
                    "sensitivity": 1.0,
                    "unit": "",
                    "offset": 0.0,
                    "iepe": False,
                    "coupling": "1"
                },
                {
                    "n": 1,
                    "name": "вход 3",
                    "sensitivity": 1.0,
                    "unit": "",
                    "offset": 0.0,
                    "iepe": False,
                    "coupling": "1"
                }
            ]
        }
        self.assertDictEqual(self.__gtrfile.header, target_header)
        self.assertEqual(self.__gtrfile.inputs_number, 2)

    def test_init_should_calculate_items_section_parameters(self):
        self.assertEqual(self.__gtrfile.items_number, 25000)
        self.assertEqual(self.__gtrfile.remainder_length, 25000)
        target_items_section = {
            "start": 424,
            "end": 200424,
            "length": 200000
        }
        self.assertDictEqual(self.__gtrfile.items_section,
                             target_items_section)

    def test_init_should_construct_item_dtype(self):
        self.assertEqual(self.__gtrfile.item_dtype, self.__target_dtype)

    def test_should_seek_and_read_item_from_start_position(self):
        whence = SEEK_SET

        offset = 0
        self.__gtrfile.seek_item(offset, whence)
        self.assertEqual(self.__gtrfile.current_item_index,
                         offset)
        self.assertEqual(self.__gtrfile.remainder_length,
                         25000-offset)
        item = self.__gtrfile.read_items(1)
        self.assertEqual(item["s"], np.array(
            [(7.670629024505615234e-01, 4.544281959533691406e-01)],
            dtype=self.__target_dtype))

        offset = 25
        self.__gtrfile.seek_item(offset, whence)
        self.assertEqual(self.__gtrfile.current_item_index,
                         offset)
        self.assertEqual(self.__gtrfile.remainder_length,
                         25000-offset)
        item = self.__gtrfile.read_items(1)
        self.assertEqual(item["s"], np.array(
            [(1.473498344421386719e+00, 4.495286941528320312e-01)],
            dtype=self.__target_dtype))

        offset = 24999
        self.__gtrfile.seek_item(offset, whence)
        self.assertEqual(self.__gtrfile.current_item_index,
                         offset)
        self.assertEqual(self.__gtrfile.remainder_length,
                         25000-offset)
        item = self.__gtrfile.read_items(1)
        self.assertEqual(item["s"], np.array(
            [(6.803703308105468750e-01, 4.548680782318115234e-01)],
            dtype=self.__target_dtype))

        offset = 25000
        self.__gtrfile.seek_item(offset, whence)
        self.assertEqual(self.__gtrfile.current_item_index,
                         offset)
        self.assertEqual(self.__gtrfile.remainder_length,
                         25000-offset)
        with self.assertRaises(ValueError) as error:
            item = self.__gtrfile.read_items(1)
        self.assertEqual(str(error.exception),
                         "\"count\" is greater than remainder length")

    def test_should_seek_and_read_item_from_current_position(self):
        whence = SEEK_SET

        start_offset = 6724
        self.__gtrfile.seek_item(start_offset, whence)

        whence = SEEK_CUR
        offset = 25
        self.__gtrfile.seek_item(offset, whence)
        self.assertEqual(self.__gtrfile.current_item_index,
                         offset+start_offset)
        self.assertEqual(self.__gtrfile.remainder_length,
                         25000-offset-start_offset)
        item = self.__gtrfile.read_items(1)
        self.assertEqual(item["s"], np.array(
            [(-1.562879085540771484e+00, 4.425144195556640625e-01)],
            dtype=self.__target_dtype))

    def test_should_seek_and_read_item_from_end_position(self):
        whence = SEEK_END
        offset = 0
        self.__gtrfile.seek_item(offset, whence)
        self.assertEqual(self.__gtrfile.current_item_index,
                         25000)
        self.assertEqual(self.__gtrfile.remainder_length,
                         0)
        offset = -25
        self.__gtrfile.seek_item(offset, whence)
        self.assertEqual(self.__gtrfile.current_item_index,
                         25000+offset)
        self.assertEqual(self.__gtrfile.remainder_length,
                         0-offset)
        offset = -25000
        self.__gtrfile.seek_item(offset, whence)
        self.assertEqual(self.__gtrfile.current_item_index,
                         25000+offset)
        self.assertEqual(self.__gtrfile.remainder_length,
                         0-offset)

    def test_should_seek_and_read_items_from_start_position(self):
        whence = SEEK_SET
        self.__gtrfile.seek_item(25, whence)
        items = self.__gtrfile.read_items(10)
        dt = np.dtype([("вход 1", np.float32), ("вход 3", np.float32)])
        target_s = np.array([(1.473498344421386719e+00, 4.495286941528320312e-01),
                             (1.488910913467407227e+00, 4.494071006774902344e-01),
                             (1.503187417984008789e+00, 4.489576816558837891e-01),
                             (1.516067981719970703e+00, 4.489469528198242188e-01),
                             (1.527853012084960938e+00, 4.483985900878906250e-01),
                             (1.538616418838500977e+00, 4.486179351806640625e-01),
                             (1.548420190811157227e+00, 4.478216171264648438e-01),
                             (1.556791067123413086e+00, 4.485678672790527344e-01),
                             (1.564111709594726562e+00, 4.470252990722656250e-01),
                             (1.570063829421997070e+00, 4.486751556396484375e-01)], dtype=dt)
        self.assertTrue(np.all(items["s"] == target_s))


if __name__ == "__main__":
    unittest.main()
