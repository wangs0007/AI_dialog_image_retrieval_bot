class Model_Loader(object):
    __instance = None
    __first_init = True

    def __init__(self, ip):
        if self.__first_init:
            self.fuc()
            self.ip = ip
            self.__first_init = False

    def __new__(cls, *args, **kwargs):
        if not cls.__instance:
            cls.__instance = object.__new__(cls)
        return cls.__instance

    @staticmethod
    def fuc():
        print('fuc')


obj1 = Model_Loader(1)
obj2 = Model_Loader(2)
# print(obj1.ip, obj2.ip)
