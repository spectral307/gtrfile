import numpy as np
import xml.etree.ElementTree as ET
from datetime import timedelta


class GtrFile:

    __file = None
    __bin_section_start = None
    __bin_section_end = None
    __bin_section_size = None
    __samples_number_per_input = None
    __record_duration = None
    __inputs_number = None
    __header = {
        "encoding": "cp1251",
        "device": None,
        "rate": None,
        "time": None,
        "text": None,
        "formatted_text": None,
        "inputs": []
    }
    __dtype = None
    __size_of_float = 4

    def __init__(self, path: str):
        self.__file = open(path, "rb")
        self.__file.seek(4, 0)

        self.__read_header()
        self.__calculate_bin_section_parameters()
        self.__parse_header()

        self.__record_duration = timedelta(seconds=self.__header["time"])

        self.__inputs_number = len(self.__header["inputs"])

        self.__size_of_float = 4
        self.__samples_number_per_input = self.__bin_section_size \
            // self.__inputs_number // self.__size_of_float

        obj = []
        for i, inp in enumerate(self.__header["inputs"]):
            obj.append((f"{inp['name']}", np.float32))
        self.__dtype = np.dtype(obj)

    @property
    def closed(self):
        return self.__file.closed

    @property
    def header(self):
        return self.__header

    @property
    def samples_number_per_input(self):
        return self.__samples_number_per_input

    @property
    def dtype(self):
        return self.__dtype

    def __calculate_bin_section_parameters(self):
        self.__bin_section_start = self.__file.tell()
        self.__file.seek(0, 2)
        self.__bin_section_end = self.__file.tell()
        self.__bin_section_size = self.__bin_section_end - self.__bin_section_start
        self.__file.seek(self.__bin_section_start, 0)

    def __read_header(self):
        line = self.__file.readline().decode(self.__header["encoding"])
        self.__header["text"] = line.rstrip()
        self.__header["formatted_text"] = line
        while line != "</gtr_header>\n":
            line = self.__file.readline().decode(self.__header["encoding"])
            self.__header["text"] += line.rstrip()
            self.__header["formatted_text"] += line

    def __parse_header(self):
        tree = ET.ElementTree(ET.fromstring(self.__header["text"]))
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

    def __seek_sample(self, offset):
        if offset < 0 or offset > self.__samples_number_per_input:
            raise Exception("Wrong offset value")

        self.__size_of_float = 4
        bin_offset = self.__bin_section_start + \
            offset * self.__inputs_number * self.__size_of_float

        if bin_offset == self.__file.tell():
            return

        self.__file.seek(bin_offset, 0)

    def __get_remainder_size(self):
        self.__size_of_float = 4
        return ((self.__bin_section_end - self.__file.tell())
                // self.__inputs_number // self.__size_of_float)

    def get_samples(self, start, arr: np.ndarray):
        self.__seek_sample(start)

        if arr.dtype != self.dtype:
            raise Exception("Wrong array dtype")

        if arr.size > self.__get_remainder_size():
            raise Exception("Array too large for the data remainder")

        np.copyto(arr, np.fromfile(
            self.__file, dtype=self.dtype, count=arr.size))

    def close(self):
        if not self.closed:
            self.__file.close()

    def __str__(self):
        ind = "  "
        inputs_str = "["
        for i, inp in enumerate(self.__header["inputs"]):
            inputs_str += f"{',' if i != 0 else ''}\n{ind}{ind}{ind}{{"
            inputs_str += f"\n{ind}{ind}{ind}{ind}n: {inp['n']}"
            inputs_str += f"\n{ind}{ind}{ind}{ind}name: {inp['name']},"
            inputs_str += f"\n{ind}{ind}{ind}{ind}iepe: {inp['iepe']},"
            inputs_str += f"\n{ind}{ind}{ind}{ind}coupling: {inp['coupling']},"
            inputs_str += f"\n{ind}{ind}{ind}{ind}sensitivity: {inp['sensitivity']},"
            inputs_str += f"\n{ind}{ind}{ind}{ind}unit: {inp['unit']},"
            inputs_str += f"\n{ind}{ind}{ind}{ind}offset: {inp['offset']},"
            inputs_str += f"\n{ind}{ind}{ind}}}"
        inputs_str += f"\n{ind}{ind}]"
        return ("<GtrFile instance:"
                + f"\n{ind}samples_number_per_input: {self.__samples_number_per_input},"
                + f"\n{ind}bin_section_start: {self.__bin_section_start},"
                + f"\n{ind}bin_section_end: {self.__bin_section_end},"
                + f"\n{ind}bin_section_size: {self.__bin_section_size},"
                + f"\n{ind}record_duration: {self.__record_duration},"
                + f"\n{ind}inputs_number: {self.__inputs_number},"
                + f"\n{ind}dtype: {self.__dtype},"
                + f"\n{ind}header: {{"
                + f"\n{ind}{ind}encoding: {self.__header['encoding']},"
                + f"\n{ind}{ind}device: {self.__header['device']},"
                + f"\n{ind}{ind}rate: {self.__header['rate']},"
                + f"\n{ind}{ind}time: {self.__header['time']},"
                + f"\n{ind}{ind}inputs: {inputs_str}"
                + "\n>")

    def __del__(self):
        if not self.closed:
            self.close()
