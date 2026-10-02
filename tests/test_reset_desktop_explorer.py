"""资源管理器清场只能关闭文件夹窗口，不能关闭桌面外壳。"""

from __future__ import annotations

import ctypes
import sys
from types import SimpleNamespace

import pytest

from tasks import reset_desktop


@pytest.mark.skipif(sys.platform != "win32", reason="Win32 窗口回调仅在 Windows 可用")
def test_close_explorer_windows_only_targets_folder_windows(monkeypatch):
    closed = []

    class Api:
        def __init__(self, call):
            self.call = call
            self.argtypes = None

        def __call__(self, *args):
            return self.call(*args)

    classes = {11: "CabinetWClass", 22: "Progman", 33: "ExploreWClass"}

    def get_class_name(handle, buffer, _length):
        buffer.value = classes[handle]
        return len(buffer.value)

    def enum_windows(callback, _context):
        for handle in classes:
            callback(handle, None)
        return True

    def post_message(handle, message, _wparam, _lparam):
        closed.append((handle, message))
        return True

    user32 = SimpleNamespace(
        EnumWindows=Api(enum_windows),
        GetClassNameW=Api(get_class_name),
        PostMessageW=Api(post_message),
    )
    monkeypatch.setattr(ctypes.windll, "user32", user32)

    assert reset_desktop.close_explorer_windows() == "已请求关闭 2 个资源管理器文件夹窗口"
    assert closed == [(11, 0x0010), (33, 0x0010)]
