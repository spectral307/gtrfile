import numpy as np
from os import SEEK_SET, SEEK_CUR, SEEK_END
from rich import print as rich_print
import xml.etree.ElementTree as ET


class GtrFile:
    #############################################
    # Member Variables #
    #############################################
    __file = None
    __header = {
        "device": None,
        "inputs": [],
        "rate": None,
        "time": None
    }
    __header_encoding = None
    __header_raw_text = None
    __inputs_number = None
    __item_dtype = None
    __item_size = None
    __items_number = None
    __items_section = {
        "start": None,
        "end": None,
        "length": None
    }
    __sample_size = None

    #############################################
    # Magic Methods #
    #############################################
    def __init__(self, path):
        self.__header_encoding = "cp1251"
        # sample - один отсчёт одного канала;
        # размер sample в файле gtr - 4 байта (тип float32)
        self.__sample_size = 4

        self.__file = open(path, "rb")
        self.__file.seek(4, SEEK_SET)

        self.__read_header_section()

        # посчитать размер секции с отсчётами сигнала по количеству байт
        self.__items_section["start"] = self.__file.tell()
        self.__file.seek(0, SEEK_END)
        self.__items_section["end"] = self.__file.tell()
        self.__items_section["length"] = self.__items_section["end"] - \
            self.__items_section["start"]

        # вернуться в начало секции с отсчётами сигнала
        self.__file.seek(self.__items_section["start"], SEEK_SET)

        # количество записывающих каналов
        self.__inputs_number = len(self.__header["inputs"])
        # item - один отсчёт со всех записывающих каналов
        self.__item_size = self.__sample_size * self.__inputs_number
        # число item-ов, посчитанное по количеству байт
        items_number = self.__items_section["length"] / self.__item_size

        # убедиться, что число item-ов - целое
        if items_number.is_integer():
            self.__items_number = int(items_number)
        else:
            raise BaseException("Items number is not integer")

        self.__construct_item_dtype()

    def __del__(self):
        if self.__file is not None:
            self.__file.close()
    #############################################
    # Public Methods #
    #############################################

    def close(self):
        self.__file.close()

    def read_items(self, count, until_eof=False):
        if until_eof:
            count = self.__get_remainder_length()
        else:
            if count > self.__get_remainder_length():
                raise ValueError(
                    "\"count\" is greater than remainder length")

        dtype = np.dtype([("s", self.__item_dtype), ("t", "f8"), ("n", "i4")])
        ret = np.empty(count, dtype)

        start = self.current_item_index
        ret["s"] = np.fromfile(
            self.__file, dtype=self.__item_dtype, count=count)
        ret["t"] = self.__get_time(start, count)
        ret["n"] = self.__get_ordinals(start, count)

        return ret

    def read_items_until_eof(self):
        return self.read_items(0, True)

    def seek_item(self, offset, whence=SEEK_SET):
        if whence == SEEK_SET:
            if offset < 0:
                raise ValueError("\"offset\" is less than zero")
            if offset > self.__items_number:
                raise ValueError(
                    "\"offset\" is greater than items number")
            self.__file.seek(offset * self.__item_size + self.__items_section["start"],
                             whence)
        elif whence == SEEK_CUR:
            if offset < 0:
                raise ValueError("\"offset\" is less than zero")
            if offset > self.__get_remainder_length():
                raise ValueError(
                    "\"offset\" is greater than remainder length")
            self.__file.seek(offset * self.__item_size, whence)
        elif whence == SEEK_END:
            if offset > 0:
                raise ValueError("\"offset\" is greater than zero")
            if abs(offset) > self.__items_number:
                raise ValueError(
                    "\"offset\" absolute value is greater than items number")
            self.__file.seek(offset * self.__item_size, whence)
        else:
            raise ValueError("\"whence\" is invalid")

    def print_items_section_summary(self):
        rich_print(self.__items_section)

    def print_header(self):
        rich_print(self.__header)

    #############################################
    # Properties #
    #############################################
    @property
    def item_dtype(self):
        return self.__item_dtype

    @property
    def closed(self):
        return self.__file.closed

    @property
    def current_item_index(self):
        current_position = self.__file.tell()
        current_item_index = (
            current_position - self.__items_section["start"]) / self.__item_size
        if not current_item_index.is_integer():
            raise BaseException("Current item index is not integer")
        if current_item_index > self.__items_number:
            raise BaseException(
                "Current item index is greater than items number")
        return int(current_item_index)

    @property
    def header(self):
        return self.__header

    @property
    def inputs_number(self):
        return self.__inputs_number

    @property
    def items_number(self):
        return self.__items_number

    @property
    def items_section(self):
        return self.__items_section

    @property
    def remainder_length(self):
        return self.__get_remainder_length()

    #############################################
    # Private Methods #
    #############################################
    def __construct_item_dtype(self):
        obj = []
        for i, inp in enumerate(self.__header["inputs"]):
            obj.append((f"{inp['name']}", np.float32))
        self.__item_dtype = np.dtype(obj)

    def __get_remainder_length(self):
        return self.__items_number - self.current_item_index

    def __get_ordinals(self, start, count):
        return np.arange(start, start + count)

    def __get_time(self, start, count):
        ts = 1 / self.__header["rate"]
        return ts * np.arange(start, start + count)

    def __parse_header_text(self):
        tree = ET.ElementTree(ET.fromstring(self.__header_raw_text))
        root = tree.getroot()
        self.__header["time"] = int(root.attrib["time"])
        self.__header["rate"] = int(root.attrib["rate"])
        self.__header["device"] = root.attrib["device"]
        recorder = root.find("recorder")
        inputs = recorder.findall("input")

        for i, inp in enumerate(inputs):
            self.__header["inputs"].append({
                "n": i,
                "name": inp.attrib["name"],
                "sensitivity": float(inp.attrib["sensitivity"]),
                "unit": inp.attrib["unit"],
                "offset": float(inp.attrib["offset"]),
                "iepe": True if inp.attrib["iepe"] == 1 else False,
                "coupling": inp.attrib["coupling"]
            })

    def __read_header_section(self):
        self.__read_header_section_as_text()
        self.__parse_header_text()

    def __read_header_section_as_text(self):
        line = self.__file.readline().decode(self.__header_encoding)
        self.__header_raw_text = line
        while line != "</gtr_header>\n":
            line = self.__file.readline().decode(self.__header_encoding)
            self.__header_raw_text += line

def main():
    print("=> gtrfile main")

if __name__ == "__main__":
    main()