"""
File operations.
"""

import filetype


def is_video(fpath: str) -> bool:
    kind = filetype.guess(fpath)
    try:
        if kind.mime.split('/')[0] == 'video':
            return True
    except (AttributeError, IndexError):
        pass

    return False

