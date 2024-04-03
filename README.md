### Module Dependency
레벨 n에서는 그보다 하위 레벨의 모듈을 임포트할 수 없다. <br>
e.g. config 모듈에서는 utils 모듈 임포트 불가능

1. **config**
   - argparse.py <br>
     main.py에서 불러 올 범용적인 하이퍼파라미터를 지정한다.
   - components.py <br>
     범용적이지 않은 하이퍼파라미터를 지정한다. 
2. **utils**
   - dir.py <br>
     필요한 경로를 손쉽기 반환하기 위한 라이브러리
   - scheduler.py <br>
     골라 먹는 스케줄러
   - logger.py <br>
     로거 세팅을 위한 라이브러리
   - utils.py <br>
     범용적으로 쓸 수 있는 유틸리티용 라이브러리
   - loader.py <br>
     이미지 데이터를 손쉽게 불러오기 위한 라이브러리
3. **composer**
   - scaler.py <br>
     전처리에 사용하는 커스텀 스케일링을 위한 라이브러리 <br>
     loader.py의 Loader 클래스를 상속한 Scaler 클래스
   - metaloader.py <br>
     메타데이터를 생성하기 위한 라이브러리 <br>
     loader.py의 Loader 클래스를 상속한 MetaLoader 클래스
4. **models**
5. **core**
   - trainer.py
   - predictor.py
6. **main.py**

공용 인터페이스는 절대 경로를 사용하고 위의 의존성을 지켜 정의하자. <br> 
상위 레벨의 모듈을 하위 레벨의 모듈에서 임포트할 때는 공용 인터페이스만 사용하자. <br>
모듈끼리의 임포트 순서는 역방향으로, 모듈 내에서의 임포트 순서는 정방향으로 하자. <br>
같은 모듈 내의 라이브러리끼리는 절대 경로로 임포트하자. <br>
   