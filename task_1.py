"""Решение задачи 1.

Пусть IP-адрес некоторого узла подсети равен 198.65.12.67, а значение
маски для подсети равно 255.255.255.240. Определите ID подсети, к которой относится
этот узел (адрес подсети). Какое максимальное число узлов может быть в этой подсети?
Определите пул адресов этой подсети
"""

CLASSES = {
    'A': range(1, 126 + 1),
    'B': range(128, 191 + 1),
    'C': range(192, 223 + 1),
    'D': range(224, 239 + 1),
    'E': range(240, 255 + 1)
}


def set_len_8_binary(binary):
    """
    Метод удлинняет нулями слева бинарное число до 8 символов
    :param binary:
    :type binary:
    :return:
    :rtype:
    """
    zeros = ''.join(['0' for _ in range(8 - len(binary))])
    return zeros + binary


def get_binary(number):
    """
    Возвращает число в двоичной системе.
    :param number:
    :type number:
    :return:
    :rtype:
    """
    if isinstance(number, str):
        number = int(number)
    binary = bin(number)[2:]
    binary = set_len_8_binary(binary)
    return binary


def get_octets(ip_addr):
    """
    Метод возвращает список числовых октетов ip адреса
    :param ip_addr: 
    :type ip_addr: 
    :return: 
    :rtype: 
    """
    return [int(octet) for octet in ip_addr.split('.')]


def get_binary_ip(ip_addr):
    """
    Метод возвращает ip адрес по октетам в двоичном виде
    :param ip_addr:
    :type ip_addr:
    :return:
    :rtype:
    """
    octets = get_octets(ip_addr)
    result_bool_and = []
    for octet in octets:
        result_bool_and.append(get_binary(octet))
    return result_bool_and


def get_class_ip(ip_addr):
    """
    По ip адресу определяет класс сети
    :param ip_addr:
    :type ip_addr:
    :return:
    :rtype:
    """
    result_class = None
    octet_1 = get_octets(ip_addr)[0]
    for ip_class, octets in CLASSES.items():
        if octet_1 in octets:
            result_class = ip_class
            break
    return result_class


def get_adress_net(ip_addr):
    """
    По ip адресу узла определяет адрес сети
    :param ip_addr:
    :type ip_addr:
    :return:
    :rtype:
    """
    octets = get_octets(ip_addr)
    return f'{octets[0]}.{octets[1]}.{octets[2]}.0'

#
# def grouper(iterable, n):
#     """
#     принимает на вход итерируемый объект и число, обозначающее размер последовательностей,
#     на которые будет разбит исходный объект
#     :param iterable:
#     :type iterable:
#     :param n:
#     :type n:
#     :return:
#     :rtype:
#     """
#     args = [iter(iterable)] * n
#     return zip(*args)

#
# def bool_and_ips(ip_1, ip_2):
#     """
#     Метод делает побитовое И для ip адресов
#     :param ip_1:
#     :type ip_1:
#     :param ip_2:
#     :type ip_2:
#     :return:
#     :rtype:
#     """
#     octets_1 = get_octets(ip_1)
#     octets_2 = get_octets(ip_2)
#     result_bool_and = ''
#     for nom, _ in enumerate(octets_1):
#         result_bool_and += get_binary(octets_1[nom] & octets_2[nom])
#     return [''.join(slice_str) for slice_str in grouper(result_bool_and, 4)]


def get_id_by_binary_ip_mask(binary_ip, binary_mask):
    """
    Метод проходит по октетам в обратном порядке и выполняет пересечение маски и ip
    Если октет маски != 00000000, то двоичные элементы ip адреса над не нулевыми элементами маски - искомый id

    Пример
    00000000  10000110  00000101 - ip
    00001111  11000000  00000000 - маска
    Первый с конца октет неравной 00000000 = 11000000, элементы ip адреса над единицами маски = 10 - это искомый id

    :param binary_ip:
    :type binary_ip:
    :param binary_mask:
    :type binary_mask:
    :return:
    :rtype:
    """
    result = ''
    for octet_8 in reversed(range(len(binary_mask))):
        for id_bin_mask, num_bin_mask in enumerate(binary_mask[octet_8]):
            if num_bin_mask != '0':
                result += binary_ip[octet_8][id_bin_mask]
        if result != '':
            break
    return result


def get_id_net_by_ip_mask(ip_addr, ip_mask):
    """
    По ip адресу узла и маске определяет ID подсети
    :param ip_addr:
    :type ip_addr:
    :return:
    :rtype:
    """
    binary_ip = get_binary_ip(ip_addr)
    # print(binary_ip)
    binary_mask = get_binary_ip(ip_mask)
    # print(binary_mask)
    id_net = get_id_by_binary_ip_mask(binary_ip, binary_mask)
    return id_net, int(id_net, 2)


# print(get_class_ip('198.65.12.67'))
# print(get_adress_net('198.65.12.67'))
print(get_id_net_by_ip_mask('198.65.12.67', '255.255.255.240'))
print(get_id_net_by_ip_mask('129.64.134.5', '255.255.192.0'))
print(get_id_net_by_ip_mask('129.44.204.1', '255.255.252.0'))
print(get_id_net_by_ip_mask('129.44.12.1', '255.255.252.0'))
print(get_id_net_by_ip_mask('129.44.99.254', '255.255.252.0'))
print(get_id_net_by_ip_mask('164.21.174.19', '255.255.248.0'))
print(get_id_net_by_ip_mask('192.168.210.97', '255.255.255.224'))
print(get_id_net_by_ip_mask('192.168.210.97', '255.255.255.224'))

"""
('0100', 4)
('10', 2)
('110011', 51)
('000011', 3)
('011000', 24)
('10101', 21)
('011', 3)
('011', 3)"""