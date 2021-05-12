# Colorize_your_old_pics

파일만들어 놓은 것은 코드 분석하고, 계획잡기위해 긁어옴
    - cycle_gan_rashed폴더에 있는 모델을 변경하여 새로운 py파일에 모델생성
    - resnet 소스파일 분석하여 어떻게 취할지 고민. (최대한 imagenet weight 활용하는 방안으로)
    - toy.py는 그냥 확인하는 playground파일이니까 무시하쎄영
  

1. preprocessing
    - input image(b&w)를 RGB에서 LAB형식, 여기서 'L'만 뽑음 (즉, L만가지고 AB를 예측하는 방식)
    - 256x256으로 resize
    - 정규화

2. generator(G) : 흑백>>칼라
    - UNET -> resnetV2로 코드 변경
    - in : 256x256x1 >> out : 256x256x3
    - 최대한 convolution조정만으로 모델 생성
    - Loss : L1 or MSE (yhat - ytrue)

3. generator(F) : 칼라>>흑백
    - UNET -> resnetV2로 코드 변경
    - in : 256x256x3 >> out : 256x256x1
    - 최대한 convolution조정만으로 모델 생성
    - Loss : L1 or MSE (xhat - xtrue)


4. discriminator(Dx) : 칼라(transslated) vs 칼라(ground true)
    - CNN -> PatchGAN으로 변경

5. discriminator(Dy) : 흑백(reconstructed) vs 흑백(ground true)
    - CNN -> PatchGAN으로 변경


6. train
    - sample interval별로 샘플이미지 출력(ex:에폭 20단위로 샘플이미지 출력, 잘되는지 확인 + 연구자료확보)
    - 1에폭당 배치사이즈별로 훈련시킨뒤 업데이트, 그 뒤에 다음 에폭
    - 훈련은 Genrator(G,F)와 Discriminator(Dx,Dy) 각각 진행
    - Loss : 
      - GAN Loss(Dx) +      // G가 생성한 이미지 Fake,True 판별
      - GAN Loss (Dy) +     // F가 생성한 이미지 Fake,True 판별
      - Cycle Loss +        // X와 F(G(X)), Y와 G(F(X))의 손실
      - Generator Loss(G) + // G의 손실, Yhat과 Ytrue
      - Generator Loss(F)   // F의 손실, Xhat과 Xtrue
      - Identity Loss(G,F)  // 차원이 달라서 사용할 수 없을 것 같다는 생각.. 고민해봐요

7. predict
    - Generator만 작동
    - 영상이 들어올 경우 openCV를 통해서 capture하여 동영상 재생성
    - train의 경우 out이 256x256x3, Predict는 원본사이즈로 resize필요 (G 이후 resize)
    - 