from pathlib import Path


class Dir:
    """
    결과는 VAULT-cake-fdname에 저장된다.
        - _get_save_dir()는 위의 경로를 지장한다.
        - get_save_path()는 위의 경로 내에서 파일들이 저장될 경로를 반환한다.
    메타데이터는 VAULT-meta에 mode_name.csv로 저장한다.
    로그는 VAULT-log에 log_fdname.txt로 저장한다.
    """
    def __init__(self, fdname: str):
        self.fdname = fdname
        self.save_dir = self._get_save_dir()

    def get_save_path(self, sname: str, suf: str, sdir: Path = None) -> str:
        save_dir = self.save_dir if sdir is None else sdir
        save_path = str(save_dir / (sname + f'.{suf}'))
        return save_path

    def _get_save_dir(self) -> Path:
        root_save_dir = Dir.get_root_dir(return_type='save')
        save_dir = root_save_dir / self.fdname
        save_dir.mkdir(parents=True, exist_ok=True)

        return save_dir

    @staticmethod
    def get_root_dir(return_type: str = None) -> Path:
        """필요한 절대 경로를 반환한다."""
        assert return_type in [None, 'data', 'save', 'meta', 'log'], \
            ("Parameter 'return_type' must be one of three: "
             "'None' for root, 'data' for dataset', 'save' for savings,"
             "'meta' for metadata, 'log' for log.")

        """
        __file__은 현재(i.e. 이 함수가 정의된) 스크립트 파일의 위치를 제공한다.
        그러므로 함수가 임의의 경로에서 실행된다 하더라도 동일한 값을 반환한다.
        resolve()는 경로를 절대 경로의 형태로 명확하게 만든다.
        파일 경로에 심볼릭 링크가 포함되어도 그 링크를 따라가 파일의 실제 절대 경로를 찾는다.
        """
        ROOT_DIR = Path(__file__).resolve().parents[1]

        if return_type is None:
            return ROOT_DIR
        elif return_type == 'data':
            DATA_DIR = ROOT_DIR / 'dataset'
            return DATA_DIR
        elif return_type == 'save':
            SAVE_DIR = ROOT_DIR / 'cake'
            SAVE_DIR.mkdir(parents=True, exist_ok=True)
            return SAVE_DIR
        elif return_type == 'meta':
            META_DIR = ROOT_DIR / 'meta'
            META_DIR.mkdir(parents=True, exist_ok=True)
            return META_DIR
        elif return_type == 'log':
            LOG_DIR = ROOT_DIR / 'log'
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            return LOG_DIR