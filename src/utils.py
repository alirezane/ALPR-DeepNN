import os
import datetime


digit_dictionary = {
    'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
    'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'zero': '0',
    '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five',
    '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine', '0': 'zero', 'noise' : 'noise', 'Noise' : 'noise',
    'ALEF': 'ALEF', 'BEH': 'BEH', 'DAL': 'DAL',
    'EIN': 'EIN', 'GHAF': 'GHAF', 'HEH': 'HEH', 'JIM': 'JIM',
    'LAAM': 'LAAM', 'MIM': 'MIM', 'NOON': 'NOON', 'SAAD': 'SAAD',
    'SIN': 'SIN', 'TA': 'TA', 'TEH': 'TEH', 'YEH': 'YEH', 'VAV': 'VAV',
    'ZHE': 'ZHE', 'KAF': 'KAF', 'PEH': 'PEH', 'D': 'D', 'S': 'S',
    'SHIN': 'SHIN', 'FEH': 'FEH', 'SEH': 'SEH', 'ZEH': 'ZEH'
}

char_vector = [
    'ALEF', 'BEH', 'DAL',
    'EIN', 'GHAF', 'HEH', 'JIM',
    'LAAM', 'MIM', 'NOON', 'SAAD',
    'SIN', 'TA', 'TEH', 'YEH', 'VAV',
    'ZHE', 'KAF', 'PEH', 'D', 'S',
    'SHIN', 'FEH', 'SEH', 'ZEH'
]

digit_vector = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

digit_char_vector = digit_vector + char_vector


class AFile:
    def __init__(self, file_path, file_name):
        self.path = file_path
        self.name = file_name
        self.lines = {}
        self.address_lines = {}
        f = open(os.path.join(self.path, self.name), 'r')
        lines = f.readlines()
        f.close()
        for line in lines:
            new_line = Line(line, 'a')
            if new_line.index != 'Address':
                self.lines[new_line.index] = new_line
            else:
                if new_line.tag in self.address_lines.keys():
                    self.address_lines[new_line.tag].append(new_line)
                else:
                    self.address_lines[new_line.tag] = [new_line]
    def update(self):
        f = open(os.path.join(self.path, self.name), 'w')
        for index in ['PlateType'] + [str(i) for i in range(1, 9)]:
            f.write(self.lines[index].generate_string())
        for key in self.address_lines.keys():
            for address_line in self.address_lines[key]:
                f.write(address_line.generate_string())
        f.close()
    def label_checked(self):
        label_checked_flag = False
        for key in self.lines.keys():
            if self.lines[key].username != 'Crawler':
                label_checked_flag = True
                break
        return label_checked_flag

    def get_label(self):
        label = []
        for index in [str(i) for i in range(1, 3)]:
            label.append(digit_dictionary[self.lines[index].tag])
        label.append(self.lines['3'].tag)
        for index in [str(i) for i in range(4, 9)]:
            label.append(digit_dictionary[self.lines[index].tag])
        return label


class LFile:
    def __init__(self, file_path, file_name):
        self.path = file_path
        self.name = file_name
        self.lines = {}
        f = open(os.path.join(self.path, self.name), 'r')
        lines = f.readlines()
        lines.reverse()
        f.close()
        for line in lines:
            new_line = Line(line, 'l')
            if new_line.index not in self.lines.keys():
                self.lines[new_line.index] = new_line
            else:
                if new_line.get_time() > self.lines[new_line.index].get_time():
                    self.lines[new_line.index] = new_line


class Line:
    def __init__(self, line, type):
        self.string = line
        self.type = type
        self.username = self.get_username()
        self.index = self.get_index()
        self.tag = self.get_tag()
        self.time_str = self.get_time_str()
        self.file = self.get_file()
        self.crawltime_str = self.get_crawltime_str()
    def get_username(self):
        return self.string.split('Username:')[1].split(';')[0]
    def get_index(self):
        return self.string.split('Index:')[1].split(';')[0]
    def get_tag(self):
        return self.string.split('Tag:')[1].split(';')[0]
    def get_time_str(self):
        return self.string.split('Time:')[1].split(';')[0]
    def get_time(self):
        return datetime.datetime.strptime(self.time_str, '%d/%m/%Y %H-%M-%S')
    def get_file(self):
        if self.type == 'a':
            return self.string.split('File:')[1].split(';')[0]
        else:
            return ''
    def get_crawltime_str(self):
        if self.type == 'a':
            return self.string.split('CrawlTime:')[1].split(';')[0]
        else:
            return ''
    def get_crawltime(self):
        return  datetime.datetime.strptime(self.crawltime_str, '%d/%m/%Y %H-%M-%S')
    def generate_string(self):
        if self.type == 'a':
            return 'Username:{};Index:{};Tag:{};Time:{};File:{};CrawlTime:{};\n'.format(self.username,
                                                                                        self.index,
                                                                                        self.tag,
                                                                                        self.time_str,
                                                                                        self.file,
                                                                                        self.crawltime_str)
        else:
            return 'Username:{};Index:{};Tag:{};Time:{};\n'.format(self.username,
                                                                   self.index,
                                                                   self.tag,
                                                                   self.time_str)
