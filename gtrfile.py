from datetime import timedelta
import numpy as np
import xml.etree.ElementTree as ET


class GtrFile:

    #############################################
    # Member Variables #
    #############################################
    __bin_section_end = None
    __bin_section_size = None
    __bin_section_start = None
    __samples_dtype = None
    __file = None
    __header = {
        "device": None,
        "encoding": "cp1251",
        "formatted_text": None,
        "inputs": [],
        "rate": None,
        "text": None,
        "time": None
    }
    __inputs_number = None
    __record_duration = None
    __samples_number_per_input = None
    __size_of_float = 4

    #############################################
    # Magic Methods #
    #############################################
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

        self.__infer_samples_dtype()

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
                + f"\n{ind}dtype: {self.__samples_dtype},"
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

    #############################################
    # Public Methods #
    #############################################
    def get_samples(self, start: int, size: int):
        self.__seek_sample(start)

        if size > self.__get_remainder_size():
            raise Exception("Size is greater than remainder size")

        s = np.fromfile(self.__file, dtype=self.dtype, count=size)
        t = self.__calc_time_vector_in_seconds(start, size)

        return (t, s)

    def close(self):
        if not self.closed:
            self.__file.close()

    #############################################
    # Properties #
    #############################################
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
        return self.__samples_dtype

    #############################################
    # Private Methods #
    #############################################
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

    def __infer_samples_dtype(self):
        obj = []
        for i, inp in enumerate(self.__header["inputs"]):
            obj.append((f"{inp['name']}", np.float32))
        self.__samples_dtype = np.dtype(obj)

    def __seek_sample(self, offset):
        if offset < 0 or offset > self.__samples_number_per_input:
            raise Exception("Wrong offset value")

        self.__size_of_float = 4
        bin_offset = self.__bin_section_start + offset * \
            self.__inputs_number * self.__size_of_float

        if bin_offset == self.__file.tell():
            return

        self.__file.seek(bin_offset, 0)

    def __get_remainder_size(self):
        self.__size_of_float = 4
        return ((self.__bin_section_end - self.__file.tell())
                // self.__inputs_number // self.__size_of_float)

    def __calc_time_vector_in_seconds(self, start, size):
        sampling_interval = 1 / self.header["rate"]
        return sampling_interval * np.arange(start, start+size)
