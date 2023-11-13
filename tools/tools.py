import os
import requests
from urllib.parse import urlparse


def download_image(url, path):
    if not os.path.isdir(path):
        os.makedirs(path)

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        url_path = urlparse(url)
        filename = os.path.basename(url_path.path)
        file_path = os.path.join(path, filename)

        with open(file_path, 'wb') as f:
            f.write(response.content)

        return file_path
    else:
        print("Failed to get image at {}".format(url))
        return None


def print_separator(debugging):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if debugging:
                print("********** Start {} **********".format(func.__name__))
            result = func(*args, **kwargs)
            if debugging:
                print("********** End {} **********\n".format(func.__name__))
            return result
        return wrapper
    return decorator
