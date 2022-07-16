# TRANSFORMER-BASED ACOUSTIC MODELING  FOR HYBRID SPEECH RECOGNITION

### 1. Data
- Librispeech data (960h for training, {dev, test}-{clean, other})
- 80-dimensional MFCC with a 10ms frame shift
- bootstrap HMM-GMM system using Kaldi Librispeech recipe
- Speed perturbation & SpecAugment

 train_clean_100 데이터를 이용해 monophone HMM 및 triphone HMM 학습을 진행하고, 적은 데이터로 학습을 시도하기 위해 train_clean_100, dev_clean, test_clean에 대해서만 label을 생성했습니다. [Kaldi Librispeech recipe의 run.sh](https://github.com/kaldi-asr/kaldi/blob/master/egs/librispeech/s5/run.sh)를 stage 13까지 실행하여 HMM을 학습하고 alignment를 진행했습니다.
 
![image](https://user-images.githubusercontent.com/55790232/179365632-309b0ca2-b4d0-4461-9284-c6e2f70b684f.png)


### 2. Model Implement
  [fairseq library](https://github.com/pytorch/fairseq/tree/master/examples/speech_recognition)를 이용하였고, 구현한 코드는 vgg_transformer.py 파일로 첨부하였습니다.
1) Positional Embedding: Convolution – VGG blocks : fairseq.modules의 VGGBlock class에 pooling stride 변수를 추가하여 VGG block을 구현하였습니다. 
2) Transformer Layer : fairseq.modules의 MultiHeadAttention class를 활용하여 Transformer Layer class를 구현하였습니다.

![image](https://user-images.githubusercontent.com/55790232/179365664-cdc5bee6-e347-4fe9-a405-61e5e911f313.png)
3) Iterated Loss와 Right Context Limit은 구현하지 못했습니다.

### 3. Train & Test
- [Pytorch-Kaldi](https://github.com/mravanelli/pytorch-kaldi) 을 참고하여 학습을 진행하려 했으나 입력 데이터를 tensor로 변환할 때 모델의 입력 tensor와 shape이 일치하지 않아 진행하지 못했습니다.
- scp, ark 파일을 tensor로 변환해주는 torchaudio library를 사용하거나  [SpeechBrain toolkit](https://github.com/speechbrain/speechbrain)을 활용하면 해결할 수 있을 것 같습니다.
