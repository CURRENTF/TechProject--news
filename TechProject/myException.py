class MyException(Exception):
    def __init__(self, error_code, message):
        super(MyException, self).__init__(message)
        self.error_info = message
        self.error_dict = {
            "code": error_code,
            "message": message
        }

    def __str__(self):
        return self.error_info

    def get_error_dict(self):
        return self.error_dict
