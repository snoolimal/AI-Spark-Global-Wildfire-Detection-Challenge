### 모듈 의존성
레벨 n에서는 그보다 하위 레벨의 모듈을 임포트할 수 없다. <br>
e.g. config 모듈에서는 utils 모듈 임포트 불가능

1. **config**
   - argparser.py
   - components.py
   - logger.py
2. **utils**
   1. utils.py <br>
      범용적으로 쓸 수 있는 유틸리티용 라이브러리
   2. dir.py <br>
      필요한 경로를 손쉽기 반환하기 위한 라이브러리
   3. loader.py <br>
      데이터를 손쉽게 불러오기 위한 라이브러리
3. **composer** and **model**
   - composer
     1. scaler.py <br>
        전처리에 사용하는 커스텀 스케일링을 위한 라이브러리 <br>
        loader.py의 Loader 클래스를 상속한 Scaler 클래스
     2. metaloader.py <br>
        메타데이터를 생성하기 위한 라이브러리 <br> 
        scaler.py의 Scaler 클래스를 상속한 MetaLoader 클래스
     3. dataset.py <br>
        PyTorch의 DataLoader 사용을 위한 준비 끝! 
   - model
     - unet2p.py: Attention-Based UNet++
     - utils.py: UNet3+++ 직접 정의하기 위해 UNet3+++의 구조를 블록화한 라이브러리
       - unet3pl.py: Lightweighted UNet+++ <br>
         utils.py의 UNUtils 클래스를 상속한 UN3PL 클래스
       - unet3p.py: UNet+++ <br>
         utils.py의 UNUtils 클래스를 상속한 UN3P 클래스
4. **core**
   - trainer.py
   - predictor.py
5. **main**

같은 모듈 내의 라이브러리끼리는 절대 경로로 임포트하자. <br>
공용 인터페이스는 절대 경로를 사용하고 위의 의존성을 지켜 정의하자. <br> 
상위 레벨의 모듈을 하위 레벨의 모듈에서 임포트할 때는 공용 인터페이스만 사용하자.
