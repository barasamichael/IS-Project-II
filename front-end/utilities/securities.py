import os
import json
import flask

import hashlib

def get_gravatar_hash(emailAddress = None):
    """
    Returns the Gravatar hash based on the provided email address.

    :param emailAddress: The email address used to generate the Gravatar hash. 
        If not provided, the function returns the hash for an empty string.
    :type emailAddress: str, optional

    :return: The Gravatar hash for the provided email address.
    :rtype: str
    """
    return hashlib.md5(emailAddress.lower().encode("utf-8")).hexdigest()

def load_deleted_accounts():
    DELETED_ACCOUNTS_FILE = flask.current_app.config["DELETED_ACCOUNTS_FILE"] 
    if not os.path.exists(DELETED_ACCOUNTS_FILE):
        return []
    with open(DELETED_ACCOUNTS_FILE, "r") as file:
        return json.load(file)

def save_deleted_account(email):
    DELETED_ACCOUNTS_FILE = flask.current_app.config["DELETED_ACCOUNTS_FILE"] 
    deleted_accounts = load_deleted_accounts()
    if email not in deleted_accounts:
        deleted_accounts.append(email)
        with open(DELETED_ACCOUNTS_FILE, "w") as file:
            json.dump(deleted_accounts, file)
