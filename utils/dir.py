from pathlib import Path
from utils.utils import check_mode


class Dir:
    """
    결과는 VAULT - cake - group - fdname에 저장된다.
        _get_save_dir()는 위의 경로를 지정한다.
        get_save_path()는 해당 경로 내의 결과 파일들의 저장 경로를 반환한다.

    메타데이터는 VAULT - meta에 저장된다.
        get_meta_save_path()는 메타데이터 파일용 저장 경로를 반환한다.
    """
    def __init__(self, group: str, fdname: str):
        self.group = group
        self.fdname = fdname
        self.save_dir = self._get_save_dir()

    def get_save_path(self, sname: str, suf: str) -> str:
        save_path = str(self.save_dir / (sname + f'.{suf}'))
        return save_path

    @staticmethod
    def get_meta_save_path(mode: str) -> str:
        check_mode(mode)
        save_path = str(Dir.get_root_dir() / 'meta' / f'{mode}_meta.csv')
        return save_path

    @staticmethod
    def get_root_dir(return_type: str = None) -> Path:
        """필요한 절대 경로를 반환한다."""
        assert return_type in [None, 'data', 'save'], \
            ("Parameter 'return_type' must be one of three: "
             "'None' for root, 'data' for dataset', 'save' for savings.")

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
            DATA_DIR = ROOT_DIR.parent / 'dataset'
            return DATA_DIR
        elif return_type == 'save':
            SAVE_DIR = ROOT_DIR / 'cake'
            SAVE_DIR.mkdir(parents=True, exist_ok=True)
            return SAVE_DIR

    def _get_save_dir(self) -> Path:
        root_save_dir = Dir.get_root_dir(return_type='save')
        save_dir = root_save_dir / f'{self.group}' / self.fdname
        save_dir.mkdir(parents=True, exist_ok=False)

        return save_dir