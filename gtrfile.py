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
    __itemsize = None
    __items_dtype = None
    __items_number = None
    __items_size_in_bytes = None
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
    __sample_size = 4
    __str_repr = None

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

        self.__items_number = int(
            self.__bin_section_size / self.__inputs_number / self.__sample_size)

        self.__itemsize = self.__sample_size * self.__inputs_number

        self.__items_size_in_bytes = self.__items_number * self.__itemsize

        self.__construct_items_dtype()

    def __str__(self):
        if self.__str_repr is None:
            self.__construct_str_repr()
        return self.__str_repr

    def __del__(self):
        if not self.closed:
            self.close()

    #############################################
    # Public Methods #
    #############################################
    def get_items(self, start: int, size: int, until_eof=False, include_time_vector=True, include_sample_numbers=True):
        self.__seek_item(start)

        if until_eof:
            size = self.__get_remainder_size()
        else:
            if size > self.__get_remainder_size():
                raise ValueError("size is greater than the remainder size")

        ret = [None] * 3

        ret[0] = np.fromfile(
            self.__file, dtype=self.__items_dtype, count=size)

        if include_time_vector:
            ret[1] = self.__get_time_vector_in_seconds(start, size)

        if include_sample_numbers:
            ret[2] = self.__get_sample_numbers(start, size)

        return ret

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
    def items_number(self):
        return self.__items_number

    @property
    def inputs_number(self):
        return self.__inputs_number

    @property
    def items_dtype(self):
        return self.__items_dtype

    @property
    def sample_size(self):
        return self.__sample_size

    @property
    def itemsize(self):
        return self.__itemsize

    @property
    def duration(self):
        return self.__record_duration

    @property
    def rate(self):
        return self.__header["rate"]

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

    def __construct_items_dtype(self):
        obj = []
        for i, inp in enumerate(self.__header["inputs"]):
            obj.append((f"{inp['name']}", np.float32))
        self.__items_dtype = np.dtype(obj)

    def __construct_str_repr(self):
        ind = "  "
        inputs_str = "["
        for i, inp in enumerate(self.__header["inputs"]):
            inputs_str += f"{',' if i != 0 else ''}\n{ind*3}{{"
            inputs_str += f"\n{ind*4}n: {inp['n']},"
            inputs_str += f"\n{ind*4}name: {inp['name']},"
            inputs_str += f"\n{ind*4}iepe: {inp['iepe']},"
            inputs_str += f"\n{ind*4}coupling: {inp['coupling']},"
            inputs_str += f"\n{ind*4}sensitivity: {inp['sensitivity']},"
            inputs_str += f"\n{ind*4}unit: {inp['unit']},"
            inputs_str += f"\n{ind*4}offset: {inp['offset']}"
            inputs_str += f"\n{ind*3}}}"
        inputs_str += f"\n{ind*2}]"
        self.__str_repr = ("<GtrFile instance:"
                           + f"\n{ind}binary section start: {self.__bin_section_start},"
                           + f"\n{ind}binary section end: {self.__bin_section_end},"
                           + f"\n{ind}binary section size: {self.__bin_section_size},"
                           + f"\n{ind}record duration: {self.__record_duration},"
                           + f"\n{ind}item size: {self.__itemsize},"
                           + f"\n{ind}items size in mbytes: {'{:.2f}'.format(self.__items_size_in_bytes/1024/1024)},"
                           + f"\n{ind}items number per input: {self.__items_number},"
                           + f"\n{ind}items dtype: {self.__items_dtype},"
                           + f"\n{ind}inputs number: {self.__inputs_number},"
                           + f"\n{ind}header: {{"
                           + f"\n{ind*2}device: {self.__header['device']},"
                           + f"\n{ind*2}rate: {self.__header['rate']},"
                           + f"\n{ind*2}time: {self.__header['time']},"
                           + f"\n{ind*2}encoding: {self.__header['encoding']},"
                           + f"\n{ind*2}inputs: {inputs_str}"
                           + f"\n{ind}}}"
                           + "\n>")

    def __seek_item(self, offset):
        if offset < 0:
            raise ValueError("offset is less than 0")
        if offset > self.__items_number:
            raise ValueError("offset is greater than the record length")

        bin_offset = self.__bin_section_start + offset * self.__itemsize

        if bin_offset == self.__file.tell():
            return

        self.__file.seek(bin_offset, 0)

    def __get_remainder_size(self):
        return ((self.__bin_section_end - self.__file.tell())
                // self.__inputs_number // self.__sample_size)

    def __get_time_vector_in_seconds(self, start, size):
        sampling_interval = 1 / self.header["rate"]
        return sampling_interval * np.arange(start, start+size)

    def __get_sample_numbers(self, start, size):
        return np.arange(start, start+size)
