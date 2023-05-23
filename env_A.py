user = 'pagel_2187'
password = 'Fd9gHEmyRFZNH6qGh9rSzxcksXhZ4wz6'
host_name = 'data.codeup.com'

def get_db_url(database, user=user, password=password, host_name=host_name):
    """
    Generates a MySQL database URL based on the provided parameters.

    Parameters:
        database (str): The name of the database.
        user (str): The username for accessing the database. Defaults to a variable named `user`.
        password (str): The password for accessing the database. Defaults to a variable named `password`.
        host_name (str): The host name or IP address of the database server. Defaults to a variable named `host_name`.

    Returns:
        str: The generated MySQL database URL.
    """
    url = f'mysql+pymysql://{user}:{password}@{host_name}/{database}'
    return url
