# my tools

[TOC]


## 1 utils
### 1.1 options.py
- parse_options()
  - Args:
  - Return:
    - opt: 通过 yml 读取的选项
    - args: 传入的参数
  - 备注:
    - 传入参数时
      - --opt: 传入 yml 文件的路径
      - --force_yml: Force to update yml files. Examples: train:ema_decay=0.999
- dict2str(opt, indent_level=1)
  - Args:
    - opt (dict): Option dict.
    - indent_level (int): Indent level. Default: 1.
  - Return:
    - (str): Option string for printing.
  - 备注:

### 1.2 misc.py

#### 1.2.1 get
- get_from_dict(dict, key, replace_none = None)
  - Args:
    - dict (dict):
    - key (int, str, ...):
    - replace_none: the value to replace the return if the original value is None
  - Return:
    - (bool): whethe the key exists
    - (value): value
  - Note:
    - get value from dictionary
#### 1.2.2 文件操作
- is_file_exit(file_path, rename=False, delete=False)
  - Args:
    - file_path (str): file path
    - rename (bool): if path exists, whether to rename it (rename it with timestamp)
    - delete (bool): if path exists, whether to delete it
  - Return:
    - (bool): whether the file exit
  - Note:
    - 检查某文件是否存在
- make_dir(path, rename=False, delete=False):
  - Args:
    - path (str): Folder path.
    - rename (bool): if path exists, whether to rename it (rename it with timestamp and create a new one.)
    - delete (bool): if path exists, whether to delete it
    - keep (bool): if path exists, whether to keep it and do not make a new one
  - Returns:
  - Note:
    - mkdirs
- mkdir_and_rename(path)
  - Args:
    - path (str): Folder path.
  - Returns:
  - Note:
    - mkdirs. If path exists, rename it with timestamp and create a new one.
- scandir(dir_path, suffix=None, recursive=False, full_path=False)
  - Args:
    - dir_path (str): Path of the directory.
    - suffix (str | tuple(str), optional): File suffix that we are interested in. Default: None.
    - recursive (bool, optional): If set to True, recursively scan the directory. Default: False.
    - full_path (bool, optional): If set to True, include the dir_path. Default: False.
  - Returns:
    - A generator for all the interested files with relative paths.
  - Note:
    - Scan a directory to find the interested files.
    - 不读取以 `'.'` 开头的文件

#### 1.2.3 else
- get_time_str()
  - Args:
  - Returns:
    - (str): 以 %Y%m%d_%H%M%S 为形式的当前时间
  - Note:
- sizeof_fmt(size, suffix='B')
  - Args:
    - size (int): File size.
    - suffix (str): Suffix. Default: 'B'.
  - Return:
    - (str): Formatted file size.
  - Note:
    - Get human readable file size.

### 1.3 registry.py
先在 `registry.py` 中定义一个 registry 如 `IMG_PROCESS_REGISTRY = Registry('img_process')`

### 1.4 logger.py
- create_logger(filepath=None, is_console=False, change_stdout_stderr=False)
- get_logger(logger_name='base', log_level=logging.DEBUG, log_file=None, is_console=False, change_stdout_stderr=False)
  - Args:
    - logger_name (str): logger_name
  - Return: logger 如果已存在，则直接返回，若不存在，则先创建再返回