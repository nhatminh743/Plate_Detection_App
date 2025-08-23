import os

def ROOT_DIR():
    return os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__)))
            )
        )
    )

print(ROOT_DIR())