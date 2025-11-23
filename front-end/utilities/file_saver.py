import os

from werkzeug.utils import secure_filename


ALLOWED_EXTENSIONS = ["jpg", "gif", "jpeg", "png"]


def save_image(image_file, folder=None):
    """
    Save the provided image file to the specified folder.

    :param image_file: werkzeug.datastructures.FileStorage - The image file
        to be saved.

    :param folder: str, optional - The folder where the image will be saved. If
        not provided, the image will not be saved.

    :return (filename | None): str, optional - The full path to the saved image
        if successful, None otherwise.
    """
    if image_file and folder:
        if not os.path.exists(folder):
            os.makedirs(folder)

        filename = secure_filename(image_file.filename)
        image_path = os.path.join(folder, filename)
        image_file.save(image_path)

        return filename

    return None


def save_file(file, folder=None):
    """
    Save the provided file to the specified folder.

    :param file: werkzeug.datastructures.FileStorage - The file to be saved.

    :param folder: str, optional - The folder where the file will be saved.
        If not provided, the file will not be saved.

    :return filename: str | None - The full path to the saved file if
        successful, None otherwise.
    """
    if file and folder:
        if not os.path.exists(folder):
            os.makedirs(folder)

        filename = secure_filename(file.filename)
        file_path = os.path.join(folder, filename)
        file.save(file_path)

        return file_path

    return None


def allowed_file(filename, allowed_extensions=set()):
    """
    Checks if a filename has an allowed file extension.

    :param filename: str - The name of the file to be checked.
    :param allowed_extensions: set - A set of allowed extensions.

    :return: bool - True if the filename has an allowed extension, False
        otherwise.
    """
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in allowed_extensions
    )


def delete_file(path):
    """
    Deletes the file in the given path.

    :param path: str - Path of the file to be deleted.
    """
    if os.path.isfile(path):
        os.remove(path)
